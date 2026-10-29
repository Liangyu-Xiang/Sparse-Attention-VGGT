# Complete combined revision: compact acceleration

7Scenes all seven scenes, uniform128 frames, identical checkpoint/input/preprocessing/pose/recon_unposed protocol.
AUC/F1 in percent; ACC/Comp/CD unsquared metres. CD=(ACC+Comp)/2. F1 threshold0.05m; voxel0.0078125m.
Only rep_compact is newly evaluated. All prior methods, including Dense/SAF3R and the unoptimized combined implementation, are reused as requested.
Time ratios use stored matched same-input/GPU/warmup/repeat protocol medians; they are NOT contemporaneously remeasured control comparisons. No parameters selected using test truth.
SAF3R DINO per-frame Top-64 uses the ETH3D calibrated profile, distinct from lazy first-correspondence Top-4.

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Full-forward seconds | Peak GiB | Ratio vs stored Dense |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 23.6689 | 84.6389 | 0.1537 | 0.1256 | 0.1397 | 48.0128 | 12.2893 | 14.360 | 1.0000x |
| saf3r_dino64 | 24.0620 | 84.6844 | 0.1426 | 0.1266 | 0.1346 | 46.5900 | 7.0419 | 18.912 | 1.7452x (stored) |
| original | 23.2483 | 84.2760 | 0.1268 | 0.1151 | 0.1209 | 50.2582 | 81.6545 | 15.497 | N/A (historical cohort) |
| cosine_distortion | 8.0258 | 75.5553 | 0.3321 | 0.1245 | 0.2283 | 18.9966 | 80.3477 | 15.499 | N/A (historical cohort) |
| rep_combined | 15.8646 | 80.8353 | 0.1346 | 0.1248 | 0.1297 | 46.6509 | 110.4322 | 15.902 | 0.1113x (stored) |
| rep_compact | 16.7733 | 81.1725 | 0.1415 | 0.1295 | 0.1355 | 46.0568 | 19.5138 | 15.903 | 0.6298x (stored) |

Stored unoptimized combined median mean / compact median mean = 5.6592x. This is a stored-reference ratio, not a fresh paired control measurement.
Quality differences against unoptimized combined: AUC@3 +0.90868pp, AUC@30 +0.33716pp, ACC +0.006853m, Comp +0.004679m, CD +0.005766m, F1 -0.59412pp.
Floating-point reduction/atomic order and larger execution batching can cause discrete state/network drift despite identical state definitions. No claim of bitwise identical full-network output.
All complete-forward costs, geometry preparation, history/routing, sixteen-step solve, packing, transfer and diagnostic summaries count. The existing second evidence QK pass remains included.

Per-scene raw result/input/config/source snapshots are under scene/rep_compact/. Detailed six-metric scene table: report.md. Numerical checks and instrumented profile are separate diagnostic-only artifacts; their overhead is not a warmed timing measurement.
