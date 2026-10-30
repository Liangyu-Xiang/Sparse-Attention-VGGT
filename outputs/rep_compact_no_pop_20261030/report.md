# 7Scenes: rep_compact without attention compensation

Seven scenes, 128 uniformly sampled frames, matched same scene/GPU Dense. AUC and F1 in percent; distances in metres. CD=(ACC+Comp)/2. F1 threshold 0.05 m; voxel 0.0078125 m.

| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Median s | Peak GiB | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | dense | 51.5789 | 93.9944 | 0.0343 | 0.0308 | 0.0326 | 84.9976 | 12.209 | 14.360 | 1.000x |
| heads | rep_compact | 29.0559 | 88.3813 | 0.0405 | 0.0368 | 0.0387 | 77.6539 | 19.660 | 15.903 | 0.621x |
| heads | rep_compact_no_pop | 39.7884 | 91.2713 | 0.0359 | 0.0380 | 0.0369 | 78.8865 | 19.455 | 15.903 | 0.628x |
| chess | dense | 20.5135 | 88.4658 | 0.1399 | 0.0877 | 0.1138 | 35.9524 | 12.248 | 14.360 | 1.000x |
| chess | rep_compact | 12.7666 | 83.9612 | 0.1262 | 0.0827 | 0.1045 | 35.1684 | 19.540 | 15.902 | 0.627x |
| chess | rep_compact_no_pop | 20.9318 | 88.6909 | 0.1130 | 0.0784 | 0.0957 | 47.2442 | 19.578 | 15.902 | 0.626x |
| stairs | dense | 4.9172 | 80.9535 | 0.2208 | 0.4797 | 0.3502 | 15.3143 | 12.189 | 14.360 | 1.000x |
| stairs | rep_compact | 6.4715 | 78.6639 | 0.2208 | 0.4913 | 0.3560 | 14.2580 | 19.455 | 15.903 | 0.627x |
| stairs | rep_compact_no_pop | 3.8714 | 76.6371 | 0.2234 | 0.5615 | 0.3924 | 12.1814 | 19.467 | 15.902 | 0.626x |
| fire | dense | 55.3150 | 94.5579 | 0.0489 | 0.0513 | 0.0501 | 66.7010 | 12.383 | 14.360 | 1.000x |
| fire | rep_compact | 41.8184 | 92.0924 | 0.0400 | 0.0446 | 0.0423 | 71.6360 | 19.941 | 15.904 | 0.621x |
| fire | rep_compact_no_pop | 47.5271 | 93.1939 | 0.0592 | 0.0614 | 0.0603 | 58.2236 | 19.803 | 15.903 | 0.625x |
| office | dense | 14.2758 | 77.1457 | 0.0471 | 0.0521 | 0.0496 | 61.0877 | 12.256 | 14.360 | 1.000x |
| office | rep_compact | 11.9464 | 74.7379 | 0.0517 | 0.0523 | 0.0520 | 58.1697 | 19.514 | 15.900 | 0.628x |
| office | rep_compact_no_pop | 10.8268 | 76.9509 | 0.0498 | 0.0548 | 0.0523 | 57.9614 | 19.612 | 15.900 | 0.625x |
| pumpkin | dense | 10.6176 | 77.5980 | 0.5057 | 0.0965 | 0.3011 | 25.0739 | 12.473 | 14.360 | 1.000x |
| pumpkin | rep_compact | 5.7374 | 73.7049 | 0.4351 | 0.1061 | 0.2706 | 19.9008 | 19.701 | 15.901 | 0.633x |
| pumpkin | rep_compact_no_pop | 11.1302 | 76.8151 | 0.3420 | 0.0957 | 0.2188 | 19.3684 | 19.713 | 15.901 | 0.633x |
| redkitchen | dense | 8.4646 | 79.7568 | 0.0795 | 0.0812 | 0.0803 | 46.9624 | 12.308 | 14.360 | 1.000x |
| redkitchen | rep_compact | 4.2815 | 73.3756 | 0.0834 | 0.0761 | 0.0797 | 48.8672 | 19.553 | 15.902 | 0.629x |
| redkitchen | rep_compact_no_pop | 11.9546 | 82.1629 | 0.0879 | 0.1119 | 0.0999 | 40.2484 | 19.720 | 15.905 | 0.624x |

## Equal-scene averages (only all seven scenes)

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Median mean s | Peak mean GiB | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 23.6689 | 84.6389 | 0.1537 | 0.1256 | 0.1397 | 48.0128 | 12.295 | 14.360 | 1.000x |
| rep_compact | 16.0111 | 80.7025 | 0.1425 | 0.1271 | 0.1348 | 46.5220 | 19.623 | 15.902 | 0.627x |
| rep_compact_no_pop | 20.8615 | 83.6746 | 0.1302 | 0.1431 | 0.1366 | 44.8734 | 19.621 | 15.902 | 0.627x |

One warmup + three timed full model forwards, per scene and same physical GPU. Population compensation removed only from executed attention weights and associated history normalization; allocation and weight-transfer computations remain. Historical prior and coarse pooling remain unchanged.
