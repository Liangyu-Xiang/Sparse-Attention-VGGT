# Fixed-real-QKV GPU kernel ablations — 2026-11-03

## Scope and environment

GPU: RTX 5090 D, physical GPU 0. Existing independent evaluation worktree is based on `bb59688b`. The separate microbenchmark scripts are in this fix branch:
- `scripts/benchmark_light_attention_evidence.py`
- `scripts/benchmark_light_partition_ablation.py`

Method configuration: `configs/evaluation/reuse_light_single_probe_repaired.json`; real 128-frame heads input:
`outputs/fastest_single_probe_snapshot_20261103/timing/input.json`.
Observations at global layers 0, 12 and 23, first fixed window per layer.
Real Q [16, 2048, 64], full K/V [16, 133248, 64], 64 query groups (32 queries/group), 26,009 indexed patch keys and 640 special keys. BF16 QKV, FP32 probe logits. Fixed QKV, selected support and bias for each paired sparse comparison.

For each kernel configuration: 2 warmups, 7 CUDA-event timed repetitions, median GPU time (ms). Native dense reference uses PyTorch forced FLASH_ATTENTION with the same full Q/K/V but without the sparse population compensation bias; it is a **performance-only** control and its output is not expected to match sparse. No reconstruction GT and no end-to-end speedup claimed.

The remote `run_gpu_lease.py --strict-contexts --gpu 0` monitors reported `foreign_gpu_pids: []`, `same_gpu_overlap_detected: false`, and successful returncode for both runs. The separate lease reservation context runs no kernels; only one measurement program ran on GPU 0 at a time.

## A: Sparse QK/AV versus historical evidence and dense FlashAttention

| GPU ms per 2048-query window | Layer 0 | Layer 12 | Layer 23 |
|---|---:|---:|---:|
| Dense FlashAttention | 6.48150 | 6.51888 | 6.53178 |
| Sparse no evidence, 8 parts | 1.61414 | 1.62490 | 1.59498 |
| Sparse + 1 probe, 8 parts | 2.46819 | 2.27098 | 2.22477 |
| Sparse + 4 probes, 8 parts | 2.64122 | 2.49754 | 2.47146 |

The 1-probe evidence adds +0.85405, +0.64608 and +0.62979 ms per window (respectively +52.9%, +39.8%, +39.5% relative to pure sparse). The 4-probe evidence adds +1.02707, +0.87264, +0.87648 ms. The sparse output relative L2 is exactly zero between no evidence / 1 probe / 4 probes for each fixed mask in these runs. Pure sparse is ~4.0x faster than native dense FlashAttention on these windows; 1-probe sparse is ~2.6–2.9x faster.

This is **fused evidence-export overhead**, separate from the `light_history_producer` stage. It does not imply an end-to-end model speedup.

## B: Split-K parts 1/2/4/8 on same QKV and mask

| Layer | Evidence | 1 part | 2 parts | 4 parts | 8 parts |
|---|---|---:|---:|---:|---:|
| 0 | No | 1.816 | 1.722 | 1.627 | **1.609** |
| 0 | 1 probe | 3.256 | 2.954 | 2.600 | **2.475** |
| 12 | No | 1.822 | 1.761 | 1.643 | **1.632** |
| 12 | 1 probe | 2.649 | 2.680 | 2.374 | **2.290** |
| 23 | No | 1.795 | 1.707 | 1.615 | **1.605** |
| 23 | 1 probe | 2.610 | 2.568 | 2.318 | **2.240** |

All parts produce nearly identical outputs (worst reported relative L2 versus 8-part reference approximately 0.00087, attributable to different accumulation order). For this 26,009-key case, 8 parts is the fastest among tested partitions, so removing Split-K is **not** the current optimization priority. This does not establish the optimal partition count for other support sizes.

## Interpretation and next experiments

1. Pure indexed Sparse Attention is not slower than native FlashAttention for these tested real windows, despite random key gathers.
2. Fused evidence extraction is significant and needs a separate-probe kernel comparison or more efficient statistics design. A per-window increase of roughly 0.63–0.85 ms can accumulate across many windows, but an end-to-end savings claim must be measured independently.
3. Keep 8 parts for this support size pending broader tuning; do not conflate 32-query mask sharing with GPU Split-K scheduling.
4. The previously measured ~1.95s budget repair per forward is independent of these kernel experiments. Evaluate the original, semantically matched distortion-based state model without the repair before optimizing or retaining the repair.

## Persistent raw measurements

`/mnt/rfs/users/xiangliangyu/benchmarks/gpt6_attention_20261103/`

- `evidence_result.json` and `evidence_monitor.json`
- `partition_result.json` and `partition_monitor.json`

The results were copied to the persistent filesystem, not committed to the shared `main` worktree. No algorithm, checkpoint, or dataset file was modified.
