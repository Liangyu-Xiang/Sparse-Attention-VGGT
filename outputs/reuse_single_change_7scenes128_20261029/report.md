# Reuse + residual exploration: matched7Scenes128-frame evaluation

Same checkpoint, exact per-scene uniform128-frame list, preprocessing, pose and recon_unposed protocol.
AUC/F1 in percent, ACC/Comp/CD in metres; CD=(ACC+Comp)/2, unsquared.
Reconstruction threshold0.05m, downsampling0.0078125m. No parameters tuned to7Scenes truth.
Four independent reuse variants: original, cyclic query shift8, max_error state error objective, saved-logit single-QK evidence. No SAF3R method in this ablation.
Single-QK evidence saves FP32 logits during attention, then normalizes/reduces them without QK recomputation; extra scratch counts in peak memory. Shift8 reprojects group-mean histories by query overlap. All costs count.
Budget20% is an upper bound; indivisible state choices may leave unused budget, recorded in layers.json.

| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm median s | Peak allocated GiB | Speedup vs Dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | dense | 51.5789 | 93.9944 | 0.0343 | 0.0308 | 0.0326 | 84.9976 | 12.180 | 14.360 | 1.000x |
| heads | original | 45.7103 | 92.9548 | 0.0355 | 0.0321 | 0.0338 | 83.0603 | 84.826 | 15.497 | 0.144x |
| heads | shift8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| heads | max_error | 0.0164 | 7.9921 | 0.2145 | 0.2219 | 0.2182 | 15.7898 | 82.426 | 15.499 | 0.148x |
| heads | single_qk | 47.4163 | 93.2735 | 0.0353 | 0.0317 | 0.0335 | 83.2590 | 85.591 | 15.499 | 0.142x |
| chess | dense | 20.5135 | 88.4658 | 0.1399 | 0.0877 | 0.1138 | 35.9524 | 12.241 | 14.360 | 1.000x |
| chess | original | 22.2318 | 88.9522 | 0.1416 | 0.0872 | 0.1144 | 35.8581 | 81.347 | 15.497 | 0.150x |
| chess | shift8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chess | max_error | 0.0000 | 4.6752 | 0.7605 | 0.5072 | 0.6339 | 7.7021 | 81.713 | 15.498 | 0.150x |
| chess | single_qk | 22.3220 | 89.0240 | 0.1352 | 0.0856 | 0.1104 | 37.0686 | 85.080 | 15.497 | 0.144x |
| stairs | dense | 4.9172 | 80.9535 | 0.2208 | 0.4797 | 0.3502 | 15.3143 | 12.186 | 14.360 | 1.000x |
| stairs | original | 5.2411 | 77.1252 | 0.2266 | 0.4371 | 0.3318 | 14.5182 | 81.131 | 15.495 | 0.150x |
| stairs | shift8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| stairs | max_error | 0.0000 | 7.9175 | 0.4886 | 1.4860 | 0.9873 | 0.9020 | 80.816 | 15.497 | 0.151x |
| stairs | single_qk | 5.9219 | 77.1375 | 0.2266 | 0.4376 | 0.3321 | 14.6384 | 85.180 | 15.499 | 0.143x |
| fire | dense | 55.3150 | 94.5579 | 0.0489 | 0.0513 | 0.0501 | 66.7010 | 12.382 | 14.360 | 1.000x |
| fire | original | 53.6417 | 93.6274 | 0.0388 | 0.0457 | 0.0422 | 75.1424 | 82.455 | 15.497 | 0.150x |
| fire | shift8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| fire | max_error | 0.0000 | 1.7532 | 0.2059 | 0.3795 | 0.2927 | 21.0425 | 83.278 | 15.499 | 0.149x |
| fire | single_qk | 54.4619 | 93.7939 | 0.0410 | 0.0476 | 0.0443 | 73.9562 | 85.921 | 15.497 | 0.144x |
| office | dense | 14.2758 | 77.1457 | 0.0471 | 0.0521 | 0.0496 | 61.0877 | 12.253 | 14.360 | 1.000x |
| office | original | 14.3414 | 77.8461 | 0.0534 | 0.0544 | 0.0539 | 55.1947 | 81.901 | 15.497 | 0.150x |
| office | shift8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| office | max_error | 0.0000 | 1.7717 | 1.3339 | 1.5370 | 1.4354 | 0.4084 | 81.515 | 15.498 | 0.150x |
| office | single_qk | 14.4439 | 77.9696 | 0.0527 | 0.0551 | 0.0539 | 55.9563 | 85.467 | 15.498 | 0.143x |
| pumpkin | dense | 10.6176 | 77.5980 | 0.5057 | 0.0965 | 0.3011 | 25.0739 | 12.438 | 14.360 | 1.000x |
| pumpkin | original | 9.8425 | 77.0837 | 0.3306 | 0.0910 | 0.2108 | 27.6190 | 83.262 | 15.497 | 0.149x |
| pumpkin | shift8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| pumpkin | max_error | 0.0000 | 1.0536 | 0.9987 | 1.7081 | 1.3534 | 0.0000 | 82.123 | 15.499 | 0.151x |
| pumpkin | single_qk | 9.9081 | 77.0969 | 0.3347 | 0.0986 | 0.2166 | 26.5083 | 86.069 | 15.497 | 0.145x |
| redkitchen | dense | 8.4646 | 79.7568 | 0.0795 | 0.0812 | 0.0803 | 46.9624 | 12.297 | 14.360 | 1.000x |
| redkitchen | original | 10.7940 | 81.7323 | 0.0723 | 0.0626 | 0.0674 | 56.8111 | 81.569 | 15.497 | 0.151x |
| redkitchen | shift8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| redkitchen | max_error | 0.0000 | 4.2565 | 0.1761 | 0.5150 | 0.3455 | 16.5266 | 81.547 | 15.495 | 0.151x |
| redkitchen | single_qk | 10.9334 | 81.9062 | 0.0726 | 0.0621 | 0.0673 | 57.6867 | 86.206 | 15.497 | 0.143x |

## Equal-scene macro averages (only when all seven quality evaluations finish)

| Method | Complete scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 23.6689 | 84.6389 | 0.1537 | 0.1256 | 0.1397 | 48.0128 | 1.000x |
| original | 7/7 | 23.1147 | 84.1888 | 0.1284 | 0.1157 | 0.1221 | 49.7434 | 0.149x |
| shift8 | 0/7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| max_error | 7/7 | 0.0023 | 4.2028 | 0.5969 | 0.9078 | 0.7524 | 8.9102 | 0.150x |
| single_qk | 7/7 | 23.6297 | 84.3145 | 0.1283 | 0.1169 | 0.1226 | 49.8677 | 0.143x |

¹ Ratio of equal-scene mean warmed median full-forward times. Below1x is slower.
Quality, speed and memory gates are separate. Depth AbsRel is not evaluated here, so complete historical acceptance cannot be claimed.
Missing/OOM/interrupted runs retain six N/A fields with a concrete reason; no silent frame/budget reductions.
Per-method input/config/source snapshots and raw metrics are stored under scene/method/.

## Single-change differences against original

AUC/F1 differences in percentage points; distance differences in metres. Runtime ratio = original / variant; >1 means faster.

| Variant | dAUC@3 pp | dAUC@30 pp | dACC m | dComp m | dCD m | dF1 pp | Original/variant time |
|---|---:|---:|---:|---:|---:|---:|---:|
| max_error | -23.112345 | -79.985998 | +0.468476 | +0.792079 | +0.630277 | -40.833186 | 1.0054x |
| single_qk | +0.514975 | +0.125668 | -0.000097 | +0.001161 | +0.000532 | +0.124272 | 0.9616x |
