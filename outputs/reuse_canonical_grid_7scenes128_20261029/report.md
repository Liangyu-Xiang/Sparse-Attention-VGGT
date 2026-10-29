# Reuse + residual exploration: matched7Scenes128-frame evaluation

Same checkpoint, exact per-scene uniform128-frame list, preprocessing, pose and recon_unposed protocol.
AUC/F1 in percent, ACC/Comp/CD in metres; CD=(ACC+Comp)/2, unsquared.
Reconstruction threshold0.05m, downsampling0.0078125m. No parameters tuned to7Scenes truth.
Grid-only variants: original 2x2, grid4 4x4, grid8 8x8 per frame; all other method settings unchanged.
Single-QK evidence saves FP32 logits during attention, then normalizes/reduces them without QK recomputation; extra scratch counts in peak memory. Shift8 reprojects group-mean histories by query overlap. All costs count.
Budget20% is an upper bound; indivisible state choices may leave unused budget, recorded in layers.json.

| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm median s | Peak allocated GiB | Speedup vs Dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | dense | 51.5789 | 93.9944 | 0.0343 | 0.0308 | 0.0326 | 84.9976 | 12.180 | 14.360 | 1.000x |
| heads | original | 45.7103 | 92.9548 | 0.0355 | 0.0321 | 0.0338 | 83.0603 | 84.826 | 15.497 | 0.144x |
| heads | grid4 | 45.6160 | 93.1033 | 0.0342 | 0.0317 | 0.0330 | 83.2200 | 81.273 | 15.525 | 0.150x |
| heads | grid8 | 43.8935 | 92.8736 | 0.0341 | 0.0315 | 0.0328 | 83.5402 | 84.685 | 15.621 | 0.144x |
| chess | dense | 20.5135 | 88.4658 | 0.1399 | 0.0877 | 0.1138 | 35.9524 | 12.241 | 14.360 | 1.000x |
| chess | original | 22.2318 | 88.9522 | 0.1416 | 0.0872 | 0.1144 | 35.8581 | 81.347 | 15.497 | 0.150x |
| chess | grid4 | 22.7444 | 89.0395 | 0.1355 | 0.0824 | 0.1089 | 38.8558 | 81.375 | 15.525 | 0.150x |
| chess | grid8 | 21.8053 | 88.7902 | 0.1389 | 0.0798 | 0.1093 | 38.3142 | 82.520 | 15.616 | 0.148x |
| stairs | dense | 4.9172 | 80.9535 | 0.2208 | 0.4797 | 0.3502 | 15.3143 | 12.186 | 14.360 | 1.000x |
| stairs | original | 5.2411 | 77.1252 | 0.2266 | 0.4371 | 0.3318 | 14.5182 | 81.131 | 15.495 | 0.150x |
| stairs | grid4 | 6.0285 | 79.6916 | 0.2333 | 0.4383 | 0.3358 | 15.0965 | 81.971 | 15.526 | 0.149x |
| stairs | grid8 | 7.3450 | 80.0685 | 0.2174 | 0.4391 | 0.3283 | 16.4697 | 82.532 | 15.616 | 0.148x |
| fire | dense | 55.3150 | 94.5579 | 0.0489 | 0.0513 | 0.0501 | 66.7010 | 12.382 | 14.360 | 1.000x |
| fire | original | 53.6417 | 93.6274 | 0.0388 | 0.0457 | 0.0422 | 75.1424 | 82.455 | 15.497 | 0.150x |
| fire | grid4 | 57.1071 | 94.2606 | 0.0407 | 0.0456 | 0.0431 | 74.5401 | 82.514 | 15.526 | 0.150x |
| fire | grid8 | 56.8118 | 94.3069 | 0.0440 | 0.0475 | 0.0458 | 72.1484 | 83.269 | 15.616 | 0.149x |
| office | dense | 14.2758 | 77.1457 | 0.0471 | 0.0521 | 0.0496 | 61.0877 | 12.253 | 14.360 | 1.000x |
| office | original | 14.3414 | 77.8461 | 0.0534 | 0.0544 | 0.0539 | 55.1947 | 81.901 | 15.497 | 0.150x |
| office | grid4 | 13.5745 | 76.4981 | 0.0548 | 0.0561 | 0.0555 | 50.4696 | 80.793 | 15.524 | 0.152x |
| office | grid8 | 13.3079 | 76.8984 | 0.0500 | 0.0537 | 0.0519 | 57.9688 | 81.484 | 15.614 | 0.150x |
| pumpkin | dense | 10.6176 | 77.5980 | 0.5057 | 0.0965 | 0.3011 | 25.0739 | 12.438 | 14.360 | 1.000x |
| pumpkin | original | 9.8425 | 77.0837 | 0.3306 | 0.0910 | 0.2108 | 27.6190 | 83.262 | 15.497 | 0.149x |
| pumpkin | grid4 | 9.7236 | 77.1243 | 0.3841 | 0.1030 | 0.2435 | 24.0783 | 82.313 | 15.526 | 0.151x |
| pumpkin | grid8 | 9.7810 | 76.9558 | 0.4381 | 0.0901 | 0.2641 | 26.8689 | 86.443 | 15.618 | 0.144x |
| redkitchen | dense | 8.4646 | 79.7568 | 0.0795 | 0.0812 | 0.0803 | 46.9624 | 12.297 | 14.360 | 1.000x |
| redkitchen | original | 10.7940 | 81.7323 | 0.0723 | 0.0626 | 0.0674 | 56.8111 | 81.569 | 15.497 | 0.151x |
| redkitchen | grid4 | 11.7741 | 82.1998 | 0.0725 | 0.0646 | 0.0685 | 54.9771 | 82.445 | 15.524 | 0.149x |
| redkitchen | grid8 | 10.4577 | 81.4493 | 0.0705 | 0.0613 | 0.0659 | 56.8318 | 82.897 | 15.616 | 0.148x |

## Equal-scene macro averages (only when all seven quality evaluations finish)

| Method | Complete scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 23.6689 | 84.6389 | 0.1537 | 0.1256 | 0.1397 | 48.0128 | 1.000x |
| original | 7/7 | 23.1147 | 84.1888 | 0.1284 | 0.1157 | 0.1221 | 49.7434 | 0.149x |
| grid4 | 7/7 | 23.7955 | 84.5596 | 0.1364 | 0.1174 | 0.1269 | 48.7482 | 0.150x |
| grid8 | 7/7 | 23.3432 | 84.4775 | 0.1419 | 0.1147 | 0.1283 | 50.3060 | 0.147x |

¹ Ratio of equal-scene mean warmed median full-forward times. Below1x is slower.
Quality, speed and memory gates are separate. Depth AbsRel is not evaluated here, so complete historical acceptance cannot be claimed.
Missing/OOM/interrupted runs retain six N/A fields with a concrete reason; no silent frame/budget reductions.
Per-method input/config/source snapshots and raw metrics are stored under scene/method/.

## Single-change differences against original

AUC/F1 differences in percentage points; distance differences in metres. Runtime ratio = original / variant; >1 means faster.

| Variant | dAUC@3 pp | dAUC@30 pp | dACC m | dComp m | dCD m | dF1 pp | Original/variant time |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid4 | +0.680774 | +0.370793 | +0.008026 | +0.001636 | +0.004831 | -0.995182 | 1.0066x |
| grid8 | +0.228487 | +0.288714 | +0.013466 | -0.001011 | +0.006228 | +0.562630 | 0.9874x |
