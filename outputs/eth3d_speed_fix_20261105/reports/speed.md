# Paired 7Scenes speed

1 warmup +3 complete forwards; includes reset/clustering/routing/history/reordering and SAF3R Top-K recomputation; validation synchronizations retained.

Ratio of equal-scene means of warmed per-scene medians; per-scene ratios also reported.

| Method | Mean scene median (s) | Speedup vs original Dense | Maximum allocated GiB |
|---|---:|---:|---:|
| dense | 19.674792 | 1.000x | 14.358575 |
| saf3r | 10.821222 | 1.818x | 14.614327 |
| grouped_saf3r | 7.906875 | 2.488x | 14.716035 |

Method / SAF3R time: 0.730682. Gates: {'no_slowdown_macro': True, 'within_original_10percent_macro': True, 'within_10percent_every_scene': True}.

SAF3R uses repository profile eth3d-train-fltr-calib_cmpmse.json and lazy first-correspondence-head Top-4 per frame (512 selected indices at128frames), not paper DINO Top-K. Control runs have all six quality metrics N/A because only timing was requested; no control quality scores were invented. ETH3D speed not evaluated.

| Scene | Method | Seconds | Dense speedup | Method / SAF3R time |
|---|---|---:|---:|---:|
| chess | dense | 21.411971 | 1.000x | 1.607770 |
| chess | saf3r | 13.317804 | 1.608x | 1.000000 |
| chess | grouped_saf3r | 7.908299 | 2.708x | 0.593814 |
| fire | dense | 21.427315 | 1.000x | 1.613587 |
| fire | saf3r | 13.279301 | 1.614x | 1.000000 |
| fire | grouped_saf3r | 7.881755 | 2.719x | 0.593537 |
| heads | dense | 21.188182 | 1.000x | 2.227731 |
| heads | saf3r | 9.511103 | 2.228x | 1.000000 |
| heads | grouped_saf3r | 7.967501 | 2.659x | 0.837705 |
| office | dense | 21.987855 | 1.000x | 1.630758 |
| office | saf3r | 13.483213 | 1.631x | 1.000000 |
| office | grouped_saf3r | 7.856702 | 2.799x | 0.582703 |
| pumpkin | dense | 17.221432 | 1.000x | 1.974471 |
| pumpkin | saf3r | 8.722047 | 1.974x | 1.000000 |
| pumpkin | grouped_saf3r | 7.893878 | 2.182x | 0.905049 |
| redkitchen | dense | 17.090498 | 1.000x | 1.978529 |
| redkitchen | saf3r | 8.637980 | 1.979x | 1.000000 |
| redkitchen | grouped_saf3r | 7.867456 | 2.172x | 0.910798 |
| stairs | dense | 17.396290 | 1.000x | 1.977501 |
| stairs | saf3r | 8.797109 | 1.978x | 1.000000 |
| stairs | grouped_saf3r | 7.972532 | 2.182x | 0.906267 |
