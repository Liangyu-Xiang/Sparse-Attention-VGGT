# GPU speed-optimization audit — 2026-11-03

## Scope

All tests use 8× NVIDIA RTX 5090 D host `14.152.92.142`, with a CPU-only `scripts/run_gpu_lease.py --strict-contexts` monitor, one CUDA application per physical GPU. Original source and datasets are not modified. The test worktree is `/tmp/gpt6-speed-opt-worktree-20261103`; the shared project `main` is unchanged. Real 7Scenes heads, uniformly sampled 128 frames; exact VGGT checkpoint and 20% attention budget.

Persistent raw results:
`/mnt/rfs/users/xiangliangyu/benchmarks/gpt6_speed_opt_20261103/`

The performance-only tests do **not** produce paper-quality results; geometry measurements are stated only where explicitly performed. The different-GPU, parallel window-group benchmark is useful for trend exploration but does not constitute matched single-device timing.

## 1. Evidence extraction: no safe new winner

The opt-in `fast_evidence` proof-of-concept removed per-tile selected-Probe LSE accumulation, recomputing selected-LSE from already materialized FP32 raw logits. It preserved the fixed-QKV Attention output exactly and differed in selected-LSE by at most about 2e-6, but was inconsistent in speed:

| Layer | 1-probe fused (ms) | 1-probe raw-only (ms) | 4-probe fused (ms) | 4-probe raw-only (ms) |
|---|---:|---:|---:|---:|
| 0 | 2.478 | 2.304 | 2.638 | 3.809 |
| 12 | 2.270 | 2.313 | 2.497 | 3.729 |
| 23 | 2.198 | 2.261 | 2.457 | 3.684 |

Therefore **the experimental kernel path was reverted**, and the original `reuse_light_attention.py` remains the inference implementation. The associated unsuccessful test script was removed from the branch. Raw benchmark: `rawonly.json`.

## 2. Greedy budget-repair optimization

Original two-pass repair sorts every cluster's best state-upgrade ratio twice. On real tensors of shape [64,26009,4], the baseline takes about 1.26–1.31 ms per call and occurs 1518 times per 128-frame forward.

- Replacing stable sort with unstable sort or sorting via `topk(F)` saved virtually nothing or was slower; changing tie behavior is not a justified production change.
- Wrapping the same repair in CUDA Graph reproduced the selection exactly but had almost unchanged time (~1.26 ms per call including graph inputs).
- **One-pass repair** reduced per-call cost to ~0.66 ms and produced selected-state match fractions mostly >99.99% on the sampled real windows. However, even a small change in historical edge selection propagated through the model and affected geometry dramatically.
- **Second pass top-64**: The first pass remains identical. On the tested 128-frame forward, maximum per-group slack at the beginning of the second pass was 29–37 keys (different runs), and using the top 64 upgrade ratios produced exactly the same selected states as the original full stable sort on all 1518 repair calls. Packing positive float32 ratios and original cluster-index tie priority into int64 sort keys reproduced stable tie ordering. This reduced isolated repair cost from about 1.26 ms to about 1.09 ms per call.
- **Production-safe large-slack guard**: An experimental option `budget_repair_second_pass_topk=64` first checks whether any group's available budget exceeds 64 and falls back to full stable sort if so. The check requires a per-window host-side synchronization with current PyTorch execution; **full Forward became slower**, 15.0835 s vs paired original 14.5618 s. This guarded option is thus **disabled by default** (topk=0).

### Full-forward and quality checks

| Version | 128-frame median Forward | Scope |
|---|---:|---|
| Original 2-pass (GPU5) | 14.5631 s | Three warmed full forwards |
| One-pass (GPU4) | 13.4011 s | Three warmed full forwards |
| Guarded Top-64 2-pass (GPU2) | 15.0835 s | Three warmed full forwards |
| Paired original 2-pass (GPU3) | 14.5618 s | Three warmed full forwards |

One-pass quality (128-frame 7Scenes heads, ETH3D courtyard all 38 frames) using the same evaluation protocol as historical reference:

| Scene / metric | Original 2-pass | One-pass |
|---|---:|---:|
| 7Scenes heads AUC@3 | 48.7820% | 51.0704% |
| 7Scenes heads F1 | 82.9602% | 83.5877% |
| ETH3D courtyard AUC@3 | **17.6387%** | **8.9616%** |
| ETH3D courtyard F1 | 40.7328% | 64.5256% |

The 7Scenes scene improved but the ETH3D pose AUC@3 decreased sharply. These are **individual scenes, not dataset-wide averages**. There is no basis to declare the one-pass variant quality-preserving; it is an opt-in diagnostic only.

Code changes in this branch: `LightweightConfig.budget_repair_passes` defaults to 2; `LightweightConfig.budget_repair_second_pass_topk` defaults to 0. Experimental configs:
- `configs/evaluation/reuse_light_single_probe_repaired_onepass.json`
- `configs/evaluation/reuse_light_single_probe_repaired_fast2.json`

Four CPU unit tests, including tied candidates and fallback for large slack, passed. No new speed option was enabled for the existing reference configuration.

## 3. Window-group size ablation

The 32-query shared-mask decision is unchanged. Only the number of such groups processed together in one window is varied. Each method used one warmup and three measured full forwards; runs were placed on distinct GPUs simultaneously and each GPU was exclusively leased.

| window_groups | GPU | Full-forward median (s) | Result |
|---:|---:|---:|---|
| 16 | 6 | 28.1033 | Worse |
| 32 | 7 | 17.5721 | Worse |
| **64 (existing)** | 5 | **14.7759** | Best in this ablation |
| 128 | 0 | 15.2806 | Worse |
| 256 | 1 | 15.9705 | Worse |
| 512 | 4 | 16.7476 | Worse |

Neither smaller nor larger processing windows improved the current pipeline, so the existing `window_groups=64` remains unchanged.

## 4. Recommendation and remaining work

**Keep the current production/default inference math and schedule unchanged.** None of the tested rewrites simultaneously offered a demonstrated full-forward speed improvement and a sufficiently stable quality/semantics result.

The 1.95-s two-pass repair remains a conceptual and engineering problem because it was introduced to repair discrete budget underfill after changing the cluster-state objective. The next substantial advance should return to the original weighted cosine/Value distortion model with **matching History/Rep execution semantics**, eliminate the need for post-hoc sorting, and precompute cluster representation statistics once per layer. That is a modeling comparison and requires full geometry evaluation; this audit does not claim to have completed it.

The original raw GPU kernel baselines and Split-K evidence measurements are available separately at `/mnt/rfs/users/xiangliangyu/benchmarks/gpt6_attention_20261103/`.
