# Unified branch vs four existing baselines: 7Scenes 128 frames

Branch `refactor/unified-reuse-exploration`, pinned upstream `46bd65b5af355ba6d383947e6f12c93fd847e62d`.
Only the new branch is evaluated. Four old methods are read from the frozen seven-scene results, NOT rerun.
Exact checkpoint, 128-frame lists, preprocessing, pose protocol and unposed reconstruction settings are checked.
AUC/F1 percent; distances metres; CD=(ACC+Comp)/2 unsquared.
F1 threshold .05m; downsample .0078125m. One warmup and three whole forwards, including all selector/history costs.
SAF3R: lazy first-correspondence-head Top-4, ETH3D calibration profile; not paper DINO Top-K reproduction.
Old timings/speedups belong to their original measurement phase; new runtime is shown separately.
New vs Dense end-to-end speedup is N/A: no contemporaneous same-UUID Dense timing, per user no-rerun instruction.
No parameter search against test truth; no Dense fallback or silent budget/frame reductions.

| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Forward s | Peak allocated GiB | Speedup¹ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | VGGT Dense (historical) | 51.6281 | 94.0071 | 0.0341 | 0.0308 | 0.0324 | 85.1957 | 17.437 | 14.359 | 1.000x |
| heads | SAF3R lazy Top-4 (historical) | 50.9719 | 93.8501 | 0.0325 | 0.0300 | 0.0313 | 85.0901 | 8.940 | 14.612 | 1.950x |
| heads | History80 + population (historical) | 47.1088 | 93.5486 | 0.0352 | 0.0331 | 0.0341 | 81.5735 | 12.643 | 14.948 | 1.379x |
| heads | Original reuse + residual (historical) | 42.8437 | 92.5956 | 0.0366 | 0.0331 | 0.0349 | 82.0629 | 117.713 | 15.495 | 0.148x |
| heads | Unified refactor (new) | 48.1709 | 93.6467 | 0.0354 | 0.0324 | 0.0339 | 82.7774 | 71.417 | 15.701 | N/A |
| chess | VGGT Dense (historical) | 20.4273 | 88.4662 | 0.1421 | 0.0883 | 0.1152 | 35.2924 | 17.072 | 14.359 | 1.000x |
| chess | SAF3R lazy Top-4 (historical) | 20.1649 | 88.3793 | 0.1442 | 0.0807 | 0.1124 | 37.7565 | 8.739 | 14.612 | 1.954x |
| chess | History80 + population (historical) | 21.1286 | 88.7381 | 0.1137 | 0.0792 | 0.0964 | 47.4284 | 12.420 | 14.948 | 1.375x |
| chess | Original reuse + residual (historical) | 22.5230 | 89.1318 | 0.1332 | 0.0864 | 0.1098 | 37.0950 | 115.802 | 15.495 | 0.147x |
| chess | Unified refactor (new) | 20.7841 | 88.5724 | 0.1170 | 0.0839 | 0.1005 | 42.5166 | 70.412 | 15.699 | N/A |
| stairs | VGGT Dense (historical) | 4.8187 | 80.9416 | 0.2200 | 0.4811 | 0.3506 | 15.1577 | 17.224 | 14.359 | 1.000x |
| stairs | SAF3R lazy Top-4 (historical) | 8.5261 | 83.5519 | 0.1968 | 0.4862 | 0.3415 | 16.4587 | 8.862 | 14.612 | 1.944x |
| stairs | History80 + population (historical) | 4.3143 | 79.0687 | 0.2546 | 0.4942 | 0.3744 | 11.2720 | 12.557 | 14.947 | 1.372x |
| stairs | Original reuse + residual (historical) | 5.5733 | 76.9373 | 0.2232 | 0.4340 | 0.3286 | 15.0778 | 117.534 | 15.498 | 0.147x |
| stairs | Unified refactor (new) | 6.5945 | 81.5244 | 0.2274 | 0.4879 | 0.3576 | 14.7030 | 69.546 | 15.705 | N/A |
| fire | VGGT Dense (historical) | 55.1181 | 94.5230 | 0.0490 | 0.0512 | 0.0501 | 66.5909 | 17.055 | 14.359 | 1.000x |
| fire | SAF3R lazy Top-4 (historical) | 56.4961 | 94.5837 | 0.0492 | 0.0548 | 0.0520 | 67.0276 | 8.703 | 14.612 | 1.960x |
| fire | History80 + population (historical) | 52.2597 | 93.9940 | 0.0505 | 0.0521 | 0.0513 | 65.3476 | 12.414 | 14.948 | 1.374x |
| fire | Original reuse + residual (historical) | 52.9815 | 93.6606 | 0.0409 | 0.0487 | 0.0448 | 72.9849 | 116.814 | 15.496 | 0.146x |
| fire | Unified refactor (new) | 52.6493 | 94.0769 | 0.0494 | 0.0532 | 0.0513 | 65.1306 | 70.382 | 15.702 | N/A |
| office | VGGT Dense (historical) | 14.1527 | 77.1342 | 0.0472 | 0.0524 | 0.0498 | 60.9704 | 16.952 | 14.359 | 1.000x |
| office | SAF3R lazy Top-4 (historical) | 13.4391 | 75.8325 | 0.0478 | 0.0569 | 0.0523 | 58.4845 | 8.739 | 14.612 | 1.940x |
| office | History80 + population (historical) | 12.4057 | 77.3766 | 0.0516 | 0.0595 | 0.0556 | 57.4769 | 12.523 | 14.948 | 1.354x |
| office | Original reuse + residual (historical) | 14.5341 | 78.0787 | 0.0544 | 0.0556 | 0.0550 | 54.0928 | 116.154 | 15.495 | 0.146x |
| office | Unified refactor (new) | 13.1275 | 77.3757 | 0.0549 | 0.0603 | 0.0576 | 51.2630 | 71.677 | 15.704 | N/A |
| pumpkin | VGGT Dense (historical) | 6.7052 | 75.2083 | 0.4639 | 0.0847 | 0.2743 | 26.3906 | 17.285 | 14.359 | 1.000x |
| pumpkin | SAF3R lazy Top-4 (historical) | 7.1194 | 74.7347 | 0.3577 | 0.0752 | 0.2164 | 37.0315 | 8.711 | 14.612 | 1.984x |
| pumpkin | History80 + population (historical) | 5.9178 | 74.2114 | 0.3594 | 0.0819 | 0.2206 | 32.7615 | 12.453 | 14.947 | 1.388x |
| pumpkin | Original reuse + residual (historical) | 5.8686 | 74.5058 | 0.2943 | 0.0762 | 0.1853 | 31.2863 | 116.407 | 15.499 | 0.148x |
| pumpkin | Unified refactor (new) | 6.5002 | 75.3859 | 0.3668 | 0.0835 | 0.2251 | 27.6897 | 70.476 | 15.712 | N/A |
| redkitchen | VGGT Dense (historical) | 8.4441 | 79.7683 | 0.0800 | 0.0820 | 0.0810 | 46.7784 | 16.798 | 14.359 | 1.000x |
| redkitchen | SAF3R lazy Top-4 (historical) | 8.5015 | 80.0853 | 0.0801 | 0.0860 | 0.0831 | 42.3157 | 8.647 | 14.612 | 1.943x |
| redkitchen | History80 + population (historical) | 8.4482 | 79.8171 | 0.0715 | 0.0719 | 0.0717 | 50.4583 | 12.417 | 14.948 | 1.353x |
| redkitchen | Original reuse + residual (historical) | 11.1179 | 81.9000 | 0.0703 | 0.0563 | 0.0633 | 60.4989 | 116.770 | 15.496 | 0.144x |
| redkitchen | Unified refactor (new) | 8.3292 | 79.6658 | 0.0793 | 0.0873 | 0.0833 | 44.7796 | 69.455 | 15.704 | N/A |

## Equal-scene macro: only all seven complete scenes qualify

| Method | Complete | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Mean scene median s | Speedup¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VGGT Dense (historical) | 7/7 | 23.0420 | 84.2927 | 0.1480 | 0.1244 | 0.1362 | 48.0537 | 17.118 | 1.000x |
| SAF3R lazy Top-4 (historical) | 7/7 | 23.6027 | 84.4311 | 0.1298 | 0.1243 | 0.1270 | 49.1664 | 8.763 | 1.953x |
| History80 + population (historical) | 7/7 | 21.6547 | 83.8221 | 0.1338 | 0.1245 | 0.1292 | 49.4740 | 12.490 | 1.371x |
| Original reuse + residual (historical) | 7/7 | 22.2060 | 83.8300 | 0.1218 | 0.1129 | 0.1174 | 50.4427 | 116.742 | 0.147x |
| Unified refactor (new) | 7/7 | 22.3080 | 84.3211 | 0.1329 | 0.1269 | 0.1299 | 46.9800 | 70.481 | N/A |

¹ Historical speedups are retained only within their original phase. Do not rank new/old runtime as a matched speed comparison.
Published paper scores/H100 times are not mixed into this local RTX4090 comparison.
