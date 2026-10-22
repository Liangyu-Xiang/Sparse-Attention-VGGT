# Reuse + residual exploration: matched7Scenes128-frame evaluation

Same checkpoint, exact per-scene uniform128-frame list, preprocessing, pose and recon_unposed protocol.
AUC/F1 in percent, ACC/Comp/CD in metres; CD=(ACC+Comp)/2, unsquared.
Reconstruction threshold0.05m, downsampling0.0078125m. No parameters tuned to7Scenes truth.
SAF3R is repository lazy first-correspondence-head Top-4, ETH3D calibration profile, NOT a paper DINO reproduction.
New method is an output-proxy reference prototype. Its second selected-QK evidence pass and all selection/history costs count in time.
Budget20% is an upper bound; indivisible state choices may leave unused budget, recorded in layers.json.

| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm median s | Peak allocated GiB | Speedup vs Dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | dense | 51.6281 | 94.0071 | 0.0341 | 0.0308 | 0.0324 | 85.1957 | 17.437 | 14.359 | 1.000x |
| heads | saf3r | 50.9719 | 93.8501 | 0.0325 | 0.0300 | 0.0313 | 85.0901 | 8.940 | 14.612 | 1.950x |
| heads | history80 | 47.1088 | 93.5486 | 0.0352 | 0.0331 | 0.0341 | 81.5735 | 12.643 | 14.946 | 1.379x |
| heads | reuse_explore | 42.8437 | 92.5956 | 0.0366 | 0.0331 | 0.0349 | 82.0629 | 117.713 | 15.494 | 0.148x |
| chess | dense | 20.4273 | 88.4662 | 0.1421 | 0.0883 | 0.1152 | 35.2924 | 17.072 | 14.359 | 1.000x |
| chess | saf3r | 20.1649 | 88.3793 | 0.1442 | 0.0807 | 0.1124 | 37.7565 | 8.739 | 14.612 | 1.954x |
| chess | history80 | 21.1286 | 88.7381 | 0.1137 | 0.0792 | 0.0964 | 47.4284 | 12.420 | 14.946 | 1.375x |
| chess | reuse_explore | 22.5230 | 89.1318 | 0.1332 | 0.0864 | 0.1098 | 37.0950 | 115.802 | 15.494 | 0.147x |
| stairs | dense | 4.8187 | 80.9416 | 0.2200 | 0.4811 | 0.3506 | 15.1577 | 17.224 | 14.359 | 1.000x |
| stairs | saf3r | 8.5261 | 83.5519 | 0.1968 | 0.4862 | 0.3415 | 16.4587 | 8.862 | 14.612 | 1.944x |
| stairs | history80 | 4.3143 | 79.0687 | 0.2546 | 0.4942 | 0.3744 | 11.2720 | 12.557 | 14.946 | 1.372x |
| stairs | reuse_explore | 5.5733 | 76.9373 | 0.2232 | 0.4340 | 0.3286 | 15.0778 | 117.534 | 15.498 | 0.147x |
| fire | dense | 55.1181 | 94.5230 | 0.0490 | 0.0512 | 0.0501 | 66.5909 | 17.055 | 14.359 | 1.000x |
| fire | saf3r | 56.4961 | 94.5837 | 0.0492 | 0.0548 | 0.0520 | 67.0276 | 8.703 | 14.612 | 1.960x |
| fire | history80 | 52.2597 | 93.9940 | 0.0505 | 0.0521 | 0.0513 | 65.3476 | 12.414 | 14.946 | 1.374x |
| fire | reuse_explore | 52.9815 | 93.6606 | 0.0409 | 0.0487 | 0.0448 | 72.9849 | 116.814 | 15.495 | 0.146x |
| office | dense | 14.1527 | 77.1342 | 0.0472 | 0.0524 | 0.0498 | 60.9704 | 16.952 | 14.359 | 1.000x |
| office | saf3r | 13.4391 | 75.8325 | 0.0478 | 0.0569 | 0.0523 | 58.4845 | 8.739 | 14.612 | 1.940x |
| office | history80 | 12.4057 | 77.3766 | 0.0516 | 0.0595 | 0.0556 | 57.4769 | 12.523 | 14.946 | 1.354x |
| office | reuse_explore | 14.5341 | 78.0787 | 0.0544 | 0.0556 | 0.0550 | 54.0928 | 116.154 | 15.494 | 0.146x |
| pumpkin | dense | 6.7052 | 75.2083 | 0.4639 | 0.0847 | 0.2743 | 26.3906 | 17.285 | 14.359 | 1.000x |
| pumpkin | saf3r | 7.1194 | 74.7347 | 0.3577 | 0.0752 | 0.2164 | 37.0315 | 8.711 | 14.612 | 1.984x |
| pumpkin | history80 | 5.9178 | 74.2114 | 0.3594 | 0.0819 | 0.2206 | 32.7615 | 12.453 | 14.946 | 1.388x |
| pumpkin | reuse_explore | 5.8686 | 74.5058 | 0.2943 | 0.0762 | 0.1853 | 31.2863 | 116.407 | 15.496 | 0.148x |
| redkitchen | dense | 8.4441 | 79.7683 | 0.0800 | 0.0820 | 0.0810 | 46.7784 | 16.798 | 14.359 | 1.000x |
| redkitchen | saf3r | 8.5015 | 80.0853 | 0.0801 | 0.0860 | 0.0831 | 42.3157 | 8.647 | 14.612 | 1.943x |
| redkitchen | history80 | 8.4482 | 79.8171 | 0.0715 | 0.0719 | 0.0717 | 50.4583 | 12.417 | 14.946 | 1.353x |
| redkitchen | reuse_explore | 11.1179 | 81.9000 | 0.0703 | 0.0563 | 0.0633 | 60.4989 | 116.770 | 15.495 | 0.144x |

## Equal-scene macro averages (only when all seven quality evaluations finish)

| Method | Complete scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 23.0420 | 84.2927 | 0.1480 | 0.1244 | 0.1362 | 48.0537 | 1.000x |
| saf3r | 7/7 | 23.6027 | 84.4311 | 0.1298 | 0.1243 | 0.1270 | 49.1664 | 1.953x |
| history80 | 7/7 | 21.6547 | 83.8221 | 0.1338 | 0.1245 | 0.1292 | 49.4740 | 1.371x |
| reuse_explore | 7/7 | 22.2060 | 83.8300 | 0.1218 | 0.1129 | 0.1174 | 50.4427 | 0.147x |

¹ Ratio of equal-scene mean warmed median full-forward times. Below1x is slower.
Quality, speed and memory gates are separate. Depth AbsRel is not evaluated here, so complete historical acceptance cannot be claimed.
Missing/OOM/interrupted runs retain six N/A fields with a concrete reason; no silent frame/budget reductions.
Per-method input/config/source snapshots and raw metrics are stored under scene/method/.
