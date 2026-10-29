# Staged Rep-baseline revision comparisons

Original and previous cosine QUALITY results are reused, with identical input/protocol. Their historical timings are not contemporaneous comparisons. A/B/C are cumulative; D changes execution relative to C.
AUC/F1 are percentages; ACC/Comp/CD are unsquared metres. CD=(ACC+Comp)/2.
7Scenes all seven scenes, uniform 128 frames, F1 threshold 0.05 m, voxel 0.0078125 m.
SAF3R uses DINO per-frame Top-64 with the ETH3D calibrated profile, distinct from lazy first-correspondence Top-4.

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm full-forward s | Peak GiB | Dense speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 23.66892 | 84.63887 | 0.15374 | 0.12562 | 0.13968 | 48.01277 | 12.289 | 14.360 | 1.000x |
| saf3r_dino64 | 24.06203 | 84.68439 | 0.14263 | 0.12660 | 0.13461 | 46.58996 | 7.042 | 18.912 | 1.745x |
| original | 23.24827 | 84.27599 | 0.12682 | 0.11506 | 0.12094 | 50.25823 | 81.654 | 15.497 | N/A (historical) |
| cosine_distortion | 8.02575 | 75.55528 | 0.33211 | 0.12453 | 0.22832 | 18.99663 | 80.348 | 15.499 | N/A (historical) |
| rep_a | 3.93818 | 69.60208 | 0.33409 | 0.13153 | 0.23281 | 19.06140 | 102.207 | 15.500 | 0.120x |

## Each stage against its immediate reference

| Stage | Reference | dAUC@3 pp | dAUC@30 pp | dCD m | dF1 pp | Reference/stage time |
|---|---|---:|---:|---:|---:|---:|
| rep_a | cosine_distortion | -4.08757 | -5.95320 | +0.00449 | +0.06477 | N/A (historical) |

## State diagnostics

Equal-scene/equal-layer averages; seed layer included. History is selected non-Rep history state edges, not the full observed bank. Exact edge fraction uses budget as denominator.

| Method | History / cluster | Drop % | Rep / clusters % | Exact edges / budget % | Unused budget | Observed previous keys | Candidate keys | Rep overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 0.08548 | 22.23379 | 69.98057 | 21.36135 | 28.59552 | N/A | N/A | N/A |
| cosine_distortion | 0.00174 | 0.61919 | 99.22900 | 0.57049 | 6.81319 | N/A | N/A | N/A |
| rep_a | 0.00365 | 1.29989 | 98.35764 | 1.23592 | 10.80007 | 416.36 | 304.60 | 63.85 |
