# Matched seven-scene ablation

No compensation means all selected real Key weights are 1; padding remains masked; history evidence uses unit weight. Coarse pooling population correction, state allocation and the original weight-transfer calculation remain. This is an execution-weight ablation, not removal of all population information.

| Metric | Dense | rep_compact | no compensation | Change vs rep_compact |
|---|---:|---:|---:|---:|
| AUC@3 (%) | 23.6689 | 16.0111 | 20.8615 | +4.8504 |
| AUC@30 (%) | 84.6389 | 80.7025 | 83.6746 | +2.9721 |
| ACC (m) | 0.1537 | 0.1425 | 0.1302 | -0.0124 |
| Comp (m) | 0.1256 | 0.1271 | 0.1431 | +0.0160 |
| CD (m) | 0.1397 | 0.1348 | 0.1366 | +0.0018 |
| F1 (%) | 48.0128 | 46.5220 | 44.8734 | -1.6486 |

- Warm median full-forward mean across scenes: Dense 12.295s (1.000x), rep_compact 19.623s (0.627x), no compensation 19.621s (0.627x).
- Mean peak allocation: 14.360, 15.902, 15.902 GiB respectively.
- No compensation vs local Dense: AUC@3 -2.8075 pp, AUC@30 -0.9643 pp, F1 -3.1393 pp.
- Local Dense vs published Dense: AUC@3 +0.0089 pp, AUC@30 -0.0011 pp. Published values are references, not local measurements.

Seven scenes, one DA3-Bench sequence per scene, uniform 128 frames. F1 threshold 0.05m, voxel 0.0078125m; CD=(ACC+Comp)/2, unsquared. One warmup and 3 repeats per method on same physical GPU. The two-frame diagnostic verified unit selected-key weights across 24 layers; it is not a quality run.
