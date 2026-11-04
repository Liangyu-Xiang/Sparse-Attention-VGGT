# Shared Value State Loss — Six-Sequence GPU Evaluation

**Date:** 2026-11-04  
**GPU server:** `14.152.92.142:5948`, eight RTX 5090 D GPUs.  
**Experimental branch:** `codex/shared-query-state-loss-20261104`.  
**Baseline code:** `codex/sampled-history-integration-20261104`.

## Experimental scope

- **7Scenes:** heads, chess, stairs; 128 frames per scene.
- **ETH3D:** courtyard (38 frames), kicker (31), playground (32).
- Same checkpoint (`checkpoints/vggt/model_tracker_fixed_e20.pt`), frame selection (uniform, seed 0), pose and unposed reconstruction evaluation, 20% key retention, 20% clusters, sampled history every 256 queries.
- **New method:** precompute per-layer Query-independent Cluster–Grid Drop and Rep distortion, apply per-Query-Group grid probability and sparse history correction, History loss = Drop loss minus retained history contribution; remove nearest-history owner and attention population bias.
- **Old method:** same sampled256 reference with original Drop moment, Rep/History distortion and population compensation.
- **Performance measurements:** one warmup plus three warmed full-forward runs per scene. Each old/new scene pair ran on the same RTX 5090 D GPU, with independent exclusive-GPU leases (GPU IDs 2–7). No overlaps recorded.
- **Quality controls:** Dense/SAF3R/old sampled32/old sampled256 values re-averaged from the **same three scenes** of archived `outputs/sampled_history_quality_20261104/per_scene.csv`, rather than the original full-dataset averages.
- Archived controls are historical quality runs; only new and old sampled256 were rerun in this campaign. No paired timing of Dense or SAF3R is claimed.

## Six-metric geometry quality

Values are equal-scene means; AUC/F1 percentages; ACC, Comp, CD in metres, lower is better.

| Dataset | Method | AUC@3 ↑ | AUC@30 ↑ | ACC ↓ | Comp ↓ | CD ↓ | F1 ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| 7Scenes | Dense | 25.6698 | 87.8046 | 0.1317 | 0.1994 | 0.1655 | 45.4214 |
| 7Scenes | SAF3R | 26.4531 | 88.5495 | 0.1250 | 0.1987 | 0.1618 | 46.6945 |
| 7Scenes | sampled256 | 24.5188 | 87.5161 | 0.1317 | 0.1979 | 0.1648 | 45.5579 |
| 7Scenes | sampled32 | 25.0615 | 87.6001 | 0.1276 | 0.1961 | 0.1619 | 45.4552 |
| 7Scenes | **Shared Value** | **21.4170** | **85.4537** | **0.1171** | **0.2206** | **0.1689** | **46.7210** |
| ETH3D | Dense | 36.3852 | 87.1533 | 0.2206 | 0.2927 | 0.2567 | 77.1050 |
| ETH3D | SAF3R | 28.5950 | 84.1486 | 0.1799 | 0.2988 | 0.2393 | 81.5893 |
| ETH3D | sampled256 | 27.9954 | 81.8695 | 0.2575 | 0.3822 | 0.3199 | 64.8859 |
| ETH3D | sampled32 | 28.2634 | 82.4701 | 0.2327 | 0.3749 | 0.3038 | 67.3244 |
| ETH3D | **Shared Value** | **20.5813** | **78.2049** | **0.3375** | **0.5396** | **0.4385** | **54.1470** |

## Paired complete-forward timing

| Dataset | Scene | Frames | Old sampled256 (s) | Shared Value (s) | Speedup |
|---|---|---:|---:|---:|---:|
| 7Scenes | heads | 128 | 13.7284 | 12.0658 | 1.138× |
| 7Scenes | chess | 128 | 14.1368 | 12.3283 | 1.147× |
| 7Scenes | stairs | 128 | 14.1810 | 12.2376 | 1.159× |
| **7Scenes mean** | 3 scenes | — | **14.0154** | **12.2106** | **1.148×** |
| ETH3D | courtyard | 38 | 2.7090 | 2.7551 | 0.983× |
| ETH3D | kicker | 31 | 2.3198 | 2.5519 | 0.909× |
| ETH3D | playground | 32 | 2.3796 | 2.2663 | 1.050× |
| **ETH3D mean** | 3 scenes | — | **2.4695** | **2.5244** | **0.978×** |

These are paired *full-forward* timings, not isolated State Loss kernel times. Concurrent experiments were on different GPUs; CPU resource contention can affect absolute full-forward time.

## Validation and interpretation

- CUDA Triton State Loss and zero-bias CUDA Graph consistency tests **passed** on RTX 5090 D.
- End-to-end 2-frame diagnostic **passed** after fixing a legacy geometry-memory statistic that still assumed pairwise distance buffers.
- All six new sequences and all six old sampled256 controls completed pose/reconstruction metrics; no exclusive-GPU overlap was recorded.
- **7Scenes:** new F1 46.72% versus old sampled256 45.56%; however AUC@3 drops 24.52% → 21.42% and CD slightly worsens. Full-forward speed improves about 14.8%.
- **ETH3D:** new F1 54.15% versus old sampled256 64.89%, CD 0.4385 versus 0.3199, AUC@3 20.58% versus 28.00%; no full-forward speedup.
- **Decision:** the shared-statistic approach is promising for 128-frame runtime, **but the tested configuration is not quality-equivalent**. The experiment also removes probability compensation and changes History's representational semantics; conduct a controlled ablation before attributing regressions to the shared loss or geometry computations alone.

## Raw outputs

On the GPU server, the complete experiment logs, GPU isolation monitors, per-scene `result.json`, `forward.json`, `layers.json`, and frozen source snapshots are preserved at:

`/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs/shared_value_loss_six_20261104/`

Sub-reports: `paired_macro_report.md`, `paired_macro_summary.json`, `paired_speed_report.md`, `paired_speed.json`.

**No change has been made to the user's existing `main` branch worktree.**
