# Dual Iteration Sweep — Shared Value Loss (7Scenes heads, 128 frames)

**Date:** 2026-11-04. **Setup:** RTX 5090 D, 10% flat Value Clusters, 20% Key Retention, 256-query sampled history; all other parameters held constant. This experiment only changes `dual_iterations` from the original 16.

Four configurations `dual_iterations=1,2,4,8` were evaluated on separate physically exclusive GPUs using one warmup and two measured full-forward passes. The 16-iteration reference used the existing same-scene run (one warmup and three measured passes); thus the timing comparison is **not a strict matched-GPU microbenchmark**.

| Dual iterations | Average selected patch Keys across 23 dynamic layers | Usage of 26,009-Key patch budget | Forward median (s) | AUC@3° (%) | F1 (%) | CD ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 101.7 | 0.39% | 10.869 | 0.0041 | 0.0029 | 1.5201 |
| 2 | 725.4 | 2.79% | 11.003 | 0.0000 | 2.4555 | 0.3888 |
| 4 | 4767.8 | 18.33% | 10.951 | 0.2789 | 45.8727 | 0.0778 |
| 8 | 16795.2 | 64.57% | 11.233 | 15.7029 | 62.1247 | 0.0555 |
| 16 | 25903.9 | 99.60% | 11.203 | 36.5732 | 69.4481 | 0.0452 |

In the current `saf3r/triton/reuse_compact.py::budget`, each dual iteration runs `_choose` over every Group–Cluster–State followed by `_dual`; a final `_choose` uses the feasible upper penalty bound. The 16-iteration configuration thus invokes 17 choose kernels and 16 dual kernels per selection window; 1-iteration invokes two choose kernels and one dual kernel. Crucially the starting upper bound is based on maximum Drop loss and **one bisection iteration stays far from the optimal budget-using dual threshold**. It favors Drop across most Clusters rather than using the permitted 20% Key budget.

Empirical verdict: switching 16→1 by changing only `dual_iterations` produces at most a small whole-forward speed gain (~3% in this non-strict comparison) while essentially destroying geometry reconstruction. Changing the number of dual iterations alone does NOT make the mask nearly free. To target one pass, a **different budget-aware allocation rule or well-calibrated warm-start dual penalty** is necessary; keep 16 iterations until that rule has proven budget feasibility and full quality.

Full result JSONs, layer diagnostics and exclusive-GPU monitors persisted on the original server at:

`/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs/shared_value_dual_sweep_20261104/`

No experiment branch source code was modified and no change was made to `main`.
