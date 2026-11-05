# Paired 7Scenes speed

1 warmup +3 complete forwards; includes reset/clustering/routing/history/reordering and SAF3R Top-K recomputation; validation synchronizations retained.

Ratio of equal-scene means of warmed per-scene medians; per-scene ratios also reported.

| Method | Mean scene median (s) | Speedup vs original Dense | Maximum allocated GiB |
|---|---:|---:|---:|
| dense | 17.504436 | 1.000x | 14.358575 |
| saf3r | 9.002889 | 1.944x | 14.614327 |
| fast_repair | 9.589701 | 1.825x | 15.366582 |

Method / SAF3R time: 1.065180. Gates: {'no_slowdown_macro': False, 'within_original_10percent_macro': True, 'within_10percent_every_scene': True}.

SAF3R uses repository profile eth3d-train-fltr-calib_cmpmse.json and lazy first-correspondence-head Top-4 per frame (512 selected indices at128frames), not paper DINO Top-K. Control runs have all six quality metrics N/A because only timing was requested; no control quality scores were invented. ETH3D speed not evaluated.

| Scene | Method | Seconds | Dense speedup | Method / SAF3R time |
|---|---|---:|---:|---:|
| chess | dense | 17.631991 | 1.000x | 1.911664 |
| chess | saf3r | 9.223373 | 1.912x | 1.000000 |
| chess | fast_repair | 9.528906 | 1.850x | 1.033126 |
| fire | dense | 17.336228 | 1.000x | 1.950673 |
| fire | saf3r | 8.887305 | 1.951x | 1.000000 |
| fire | fast_repair | 9.536902 | 1.818x | 1.073093 |
| heads | dense | 17.635503 | 1.000x | 1.943405 |
| heads | saf3r | 9.074539 | 1.943x | 1.000000 |
| heads | fast_repair | 9.627265 | 1.832x | 1.060910 |
| office | dense | 17.303251 | 1.000x | 1.945616 |
| office | saf3r | 8.893458 | 1.946x | 1.000000 |
| office | fast_repair | 9.555860 | 1.811x | 1.074482 |
| pumpkin | dense | 17.549000 | 1.000x | 1.961454 |
| pumpkin | saf3r | 8.946934 | 1.961x | 1.000000 |
| pumpkin | fast_repair | 9.667831 | 1.815x | 1.080575 |
| redkitchen | dense | 17.394334 | 1.000x | 1.944512 |
| redkitchen | saf3r | 8.945346 | 1.945x | 1.000000 |
| redkitchen | fast_repair | 9.539977 | 1.823x | 1.066474 |
| stairs | dense | 17.680747 | 1.000x | 1.953831 |
| stairs | saf3r | 9.049271 | 1.954x | 1.000000 |
| stairs | fast_repair | 9.671166 | 1.828x | 1.068723 |
