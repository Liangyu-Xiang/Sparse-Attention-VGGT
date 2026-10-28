# Joint History-conditioned optimization:7Scenes128 evaluation



Evaluated upstream06c5e0c760b34b34c3f1a715a99417dd0406fd76. Current default joint_history_exploration=true;20% retention; all24 global layers; local/frame unchanged. No production modifications or7Scenes test-truth tuning.

The fixed relation-candidate-slot / adaptive-fresh-QK proposal remains a DOCUMENTED PLAN, not the implementation tested here.

Original checkpoint and exact input lists/preprocessing/pose+recon_unposed protocols matched. AUC/F1 percent, distances metres, CD=(ACC+Comp)/2 unsquared; threshold.05m,downsample.0078125m.



## Seven-scene equal macro quality

| method | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Dense quality (historical) | 23.04204 | 84.29269 | 0.14804 | 0.12436 | 0.13620 | 48.05374 |
| SAF3R quality (historical) | 23.60271 | 84.43107 | 0.12976 | 0.12427 | 0.12701 | 49.16638 |
| History80 (historical) | 21.65472 | 83.82206 | 0.13378 | 0.12454 | 0.12916 | 49.47402 |
| Old reuse+residual (historical) | 22.20601 | 83.82997 | 0.12183 | 0.11292 | 0.11737 | 50.44266 |
| 57bdb78 | 22.30034 | 84.30616 | 0.13782 | 0.12741 | 0.13262 | 46.80339 |
| Physical H/Rep baseline | 22.41400 | 84.31466 | 0.14087 | 0.13278 | 0.13682 | 46.20702 |
| Joint History (new) | 22.74149 | 84.12935 | 0.12921 | 0.13401 | 0.13161 | 47.06011 |

Dense quality is reused from frozen matched results. Fresh Dense runs below are TIMING-ONLY, have six explicit N/A quality fields and must not be called new quality evaluations.

SAF3R is frozen local lazy first-correspondence-head per-frame Top4/ETH3D profile, not paper DINO-feature TopK reproduction.



## Fresh paired runtime and peak memory

| scene | device_uuid | dense_seconds | method_seconds | speedup | uncontended | dense_peak_allocated_gib | method_peak_allocated_gib | dense_peak_reserved_gib | method_peak_reserved_gib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heads | GPU-917c95c3-9023-f077-f577-dae0f7f9d717 | 16.88823 | 180.10809 | 0.09377 | True | 14.35857 | 17.42723 | 17.55469 | 21.56836 |
| chess | GPU-f0913c13-448b-e88d-92fe-8d47264d279f | 17.32195 | 209.02414 | 0.08287 | True | 14.35857 | 17.42045 | 17.55469 | 22.05273 |
| stairs | GPU-917c95c3-9023-f077-f577-dae0f7f9d717 | 16.85202 | 161.99881 | 0.10403 | True | 14.35857 | 17.42290 | 17.55469 | 21.56250 |
| fire | GPU-f0913c13-448b-e88d-92fe-8d47264d279f | 17.30942 | 161.14175 | 0.10742 | True | 14.35857 | 17.42239 | 17.55469 | 22.56445 |
| office | GPU-d8d57076-0dc1-2558-282b-9de7fd388284 | 16.77777 | 157.09797 | 0.10680 | True | 14.35857 | 17.42928 | 17.55469 | 22.30859 |
| pumpkin | GPU-8e332fc5-55f4-0d85-1a2d-40bd87342c1f | 16.83514 | 156.89513 | 0.10730 | True | 14.35857 | 17.42287 | 17.55469 | 22.72461 |
| redkitchen | GPU-917c95c3-9023-f077-f577-dae0f7f9d717 | 16.70266 | 158.11456 | 0.10564 | True | 14.35857 | 17.42282 | 17.55469 | 22.32812 |

Mean scene warmed medians:Dense=16.95531s,new=169.19721s; macro end-to-end speedup=0.10021035517095771. Dense reference=1.00x. Ratio is mean Dense scene medians / mean method scene medians, not mean per-scene ratios.

Each pair uses the same UUID,input,checkpoint and timing protocol:1warmup/3whole forwards,capture=False. Includes all initialization,Value updates,selection,history,reordering. No per-layer sync. Foreign GPU processes are monitored; contaminated pairs are markedN/A. Cold samples remain in forward.json.



## Historical latency context (NOT fresh speed comparison)

| method | mean_recorded_seconds |
| --- | --- |
| Dense quality (historical) | 17.11768 |
| SAF3R quality (historical) | 8.76281 |
| History80 (historical) | 12.48963 |
| Old reuse+residual (historical) | 116.74196 |
| 57bdb78 | 72.62345 |
| Physical H/Rep baseline | 159.18326 |
| Joint History (new) | 169.19721 |



## Per-scene six quality metrics

| scene | method | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heads | Dense quality (historical) | 51.62812 | 94.00714 | 0.03407 | 0.03081 | 0.03244 | 85.19571 |
| heads | SAF3R quality (historical) | 50.97195 | 93.85007 | 0.03250 | 0.03005 | 0.03127 | 85.09014 |
| heads | History80 (historical) | 47.10876 | 93.54864 | 0.03516 | 0.03305 | 0.03411 | 81.57346 |
| heads | Old reuse+residual (historical) | 42.84367 | 92.59555 | 0.03657 | 0.03314 | 0.03486 | 82.06291 |
| heads | 57bdb78 | 46.89140 | 93.45554 | 0.03472 | 0.03301 | 0.03387 | 83.50833 |
| heads | Physical H/Rep baseline | 45.21818 | 93.11885 | 0.03412 | 0.03223 | 0.03318 | 83.22430 |
| heads | Joint History (new) | 49.53248 | 93.67372 | 0.03445 | 0.03333 | 0.03389 | 83.45029 |
| chess | Dense quality (historical) | 20.42733 | 88.46621 | 0.14207 | 0.08831 | 0.11519 | 35.29243 |
| chess | SAF3R quality (historical) | 20.16486 | 88.37927 | 0.14415 | 0.08074 | 0.11245 | 37.75651 |
| chess | History80 (historical) | 21.12861 | 88.73811 | 0.11368 | 0.07918 | 0.09643 | 47.42838 |
| chess | Old reuse+residual (historical) | 22.52297 | 89.13181 | 0.13315 | 0.08637 | 0.10976 | 37.09496 |
| chess | 57bdb78 | 21.12451 | 88.55807 | 0.12476 | 0.08920 | 0.10698 | 36.13336 |
| chess | Physical H/Rep baseline | 20.52986 | 88.43914 | 0.12019 | 0.07864 | 0.09941 | 45.41633 |
| chess | Joint History (new) | 20.99738 | 88.61344 | 0.12480 | 0.08700 | 0.10590 | 39.36137 |
| stairs | Dense quality (historical) | 4.81873 | 80.94160 | 0.22003 | 0.48109 | 0.35056 | 15.15768 |
| stairs | SAF3R quality (historical) | 8.52608 | 83.55192 | 0.19683 | 0.48621 | 0.34152 | 16.45867 |
| stairs | History80 (historical) | 4.31430 | 79.06865 | 0.25464 | 0.49419 | 0.37441 | 11.27200 |
| stairs | Old reuse+residual (historical) | 5.57333 | 76.93734 | 0.22320 | 0.43405 | 0.32862 | 15.07784 |
| stairs | 57bdb78 | 7.43930 | 82.06611 | 0.23032 | 0.47873 | 0.35452 | 14.14482 |
| stairs | Physical H/Rep baseline | 8.31693 | 81.97917 | 0.23213 | 0.49510 | 0.36362 | 13.82124 |
| stairs | Joint History (new) | 5.32316 | 80.68734 | 0.23066 | 0.54012 | 0.38539 | 14.27803 |
| fire | Dense quality (historical) | 55.11811 | 94.52305 | 0.04902 | 0.05125 | 0.05013 | 66.59086 |
| fire | SAF3R quality (historical) | 56.49606 | 94.58374 | 0.04923 | 0.05482 | 0.05202 | 67.02756 |
| fire | History80 (historical) | 52.25968 | 93.99401 | 0.05050 | 0.05207 | 0.05128 | 65.34758 |
| fire | Old reuse+residual (historical) | 52.98146 | 93.66060 | 0.04095 | 0.04871 | 0.04483 | 72.98486 |
| fire | 57bdb78 | 52.59596 | 94.08136 | 0.05136 | 0.05243 | 0.05189 | 65.24269 |
| fire | Physical H/Rep baseline | 53.85089 | 94.25320 | 0.04798 | 0.05266 | 0.05032 | 66.56591 |
| fire | Joint History (new) | 53.34236 | 94.15723 | 0.05210 | 0.05496 | 0.05353 | 63.74789 |
| office | Dense quality (historical) | 14.15272 | 77.13419 | 0.04718 | 0.05242 | 0.04980 | 60.97041 |
| office | SAF3R quality (historical) | 13.43914 | 75.83251 | 0.04782 | 0.05687 | 0.05235 | 58.48455 |
| office | History80 (historical) | 12.40568 | 77.37656 | 0.05165 | 0.05947 | 0.05556 | 57.47688 |
| office | Old reuse+residual (historical) | 14.53412 | 78.07866 | 0.05439 | 0.05558 | 0.05498 | 54.09283 |
| office | 57bdb78 | 13.24639 | 76.91929 | 0.05179 | 0.05594 | 0.05387 | 55.11043 |
| office | Physical H/Rep baseline | 13.76312 | 76.91765 | 0.05580 | 0.05822 | 0.05701 | 50.86173 |
| office | Joint History (new) | 14.33317 | 76.50262 | 0.05134 | 0.05679 | 0.05406 | 54.57789 |
| pumpkin | Dense quality (historical) | 6.70522 | 75.20833 | 0.46392 | 0.08467 | 0.27430 | 26.39064 |
| pumpkin | SAF3R quality (historical) | 7.11942 | 74.73466 | 0.35765 | 0.07522 | 0.21643 | 37.03154 |
| pumpkin | History80 (historical) | 5.91781 | 74.21137 | 0.35937 | 0.08193 | 0.22065 | 32.76154 |
| pumpkin | Old reuse+residual (historical) | 5.86860 | 74.50582 | 0.29429 | 0.07621 | 0.18525 | 31.28626 |
| pumpkin | 57bdb78 | 6.42224 | 75.30266 | 0.39347 | 0.09910 | 0.24628 | 27.20287 |
| pumpkin | Physical H/Rep baseline | 6.14337 | 75.22474 | 0.41332 | 0.11559 | 0.26445 | 21.69191 |
| pumpkin | Joint History (new) | 6.82415 | 75.26042 | 0.33295 | 0.08108 | 0.20702 | 28.71077 |
| redkitchen | Dense quality (historical) | 8.44406 | 79.76829 | 0.07995 | 0.08195 | 0.08095 | 46.77843 |
| redkitchen | SAF3R quality (historical) | 8.50148 | 80.08530 | 0.08014 | 0.08598 | 0.08306 | 42.31569 |
| redkitchen | History80 (historical) | 8.44816 | 79.81709 | 0.07151 | 0.07188 | 0.07170 | 50.45830 |
| redkitchen | Old reuse+residual (historical) | 11.11795 | 81.90002 | 0.07026 | 0.05635 | 0.06330 | 60.49893 |
| redkitchen | 57bdb78 | 8.38255 | 79.76009 | 0.07833 | 0.08346 | 0.08089 | 46.28119 |
| redkitchen | Physical H/Rep baseline | 9.07562 | 80.26985 | 0.08252 | 0.09700 | 0.08976 | 41.86773 |
| redkitchen | Joint History (new) | 8.83776 | 80.01066 | 0.07814 | 0.08481 | 0.08147 | 45.29451 |



## Layer behavior

| layer | actual_density | utilization | optimizer_gap | history_fraction | history_count | population_rep_clusters | additive_rep_keys | exact | history_promoted_representatives | history_mib | unused_budget |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.20000 | 1.00000 | 0.00000 | 0.00000 | 0.00000 | 26008.99805 | 26008.99805 | 0.00000 | 0.00000 | 619.70602 | 0.00000 |
| 1 | 0.20000 | 1.00000 | 0.00000 | 0.99999 | 26008.72015 | 26008.72042 | 0.27789 | 0.00000 | 26008.44252 | 619.70602 | 0.00055 |
| 2 | 0.19999 | 0.99995 | 0.00000 | 0.88518 | 23021.55580 | 22427.02483 | 2986.08336 | 0.00000 | 19440.94113 | 619.70602 | 1.35996 |
| 3 | 0.19998 | 0.99990 | 0.00000 | 0.87209 | 22679.80971 | 22569.26646 | 3326.50021 | 0.04494 | 19242.76618 | 619.70602 | 2.64433 |
| 4 | 0.19998 | 0.99990 | 0.00000 | 0.89290 | 23221.06445 | 23293.80720 | 2785.39844 | 0.00000 | 20508.40848 | 619.70602 | 2.53616 |
| 5 | 0.19997 | 0.99989 | 0.00000 | 0.91321 | 23749.23326 | 23465.02762 | 2256.97988 | 0.00000 | 21208.04771 | 619.70602 | 2.78626 |
| 6 | 0.19998 | 0.99991 | 0.00000 | 0.92707 | 24109.96763 | 23509.09766 | 1896.61675 | 0.00000 | 21612.48075 | 619.70602 | 2.41519 |
| 7 | 0.19997 | 0.99989 | 0.00000 | 0.93117 | 24216.17327 | 23573.19866 | 1790.00980 | 0.00000 | 21783.19001 | 619.70602 | 2.81642 |
| 8 | 0.19998 | 0.99991 | 0.00000 | 0.91863 | 23890.35603 | 23655.59626 | 2116.22278 | 0.00000 | 21539.37305 | 619.70602 | 2.42003 |
| 9 | 0.19998 | 0.99994 | 0.00000 | 0.94303 | 24525.74414 | 23230.97154 | 1481.66356 | 0.00034 | 21749.30804 | 619.70602 | 1.59119 |
| 10 | 0.19996 | 0.99984 | 0.00000 | 0.92195 | 23975.07450 | 23600.59096 | 2029.76566 | 0.03595 | 21570.82506 | 619.70602 | 4.12265 |
| 11 | 0.19995 | 0.99974 | 0.00000 | 0.91579 | 23812.72154 | 23791.05859 | 2188.66429 | 0.90291 | 21602.39369 | 619.70602 | 6.71103 |
| 12 | 0.19972 | 0.99861 | 0.00000 | 0.89177 | 23161.88923 | 24080.41936 | 2717.82673 | 93.02913 | 21362.59319 | 619.70602 | 36.25463 |
| 13 | 0.19976 | 0.99878 | 0.00000 | 0.88550 | 23003.00502 | 23887.27595 | 2868.67983 | 105.66306 | 21018.59626 | 619.70602 | 31.65171 |
| 14 | 0.19796 | 0.98960 | 0.00000 | 0.87034 | 22401.40625 | 23133.54576 | 2850.38497 | 486.63865 | 20283.16127 | 619.70602 | 270.56964 |
| 15 | 0.19866 | 0.99317 | 0.00000 | 0.82009 | 21183.92885 | 22555.17048 | 4152.90674 | 494.55790 | 18402.26395 | 619.70602 | 177.60577 |
| 16 | 0.19956 | 0.99778 | 0.00000 | 0.83806 | 21748.70006 | 22804.97210 | 4143.77930 | 58.87172 | 18661.19252 | 619.70602 | 57.64804 |
| 17 | 0.19996 | 0.99981 | 0.00000 | 0.86389 | 22464.71512 | 22853.81780 | 3538.51263 | 0.80575 | 19315.30497 | 619.70602 | 4.96569 |
| 18 | 0.19993 | 0.99967 | 0.00000 | 0.86717 | 22546.70759 | 22536.03209 | 3453.08775 | 0.56704 | 19082.94475 | 619.70602 | 8.63668 |
| 19 | 0.19974 | 0.99870 | 0.00000 | 0.90674 | 23552.67997 | 21930.91797 | 2421.06243 | 1.36545 | 19509.85407 | 619.70602 | 33.89162 |
| 20 | 0.19998 | 0.99995 | 0.00000 | 0.91852 | 23888.37416 | 21849.17662 | 2119.21704 | 0.00062 | 19729.95982 | 619.70602 | 1.40703 |
| 21 | 0.19991 | 0.99957 | 0.00000 | 0.89370 | 23234.29660 | 21819.69475 | 2761.80256 | 1.73017 | 19057.89174 | 619.70602 | 11.16993 |
| 22 | 0.19997 | 0.99989 | 0.00000 | 0.89286 | 23219.85519 | 22360.73186 | 2786.29454 | 0.00048 | 19574.43750 | 619.70602 | 2.84915 |
| 23 | 0.19998 | 0.99994 | 0.00000 | 0.92346 | 24016.86412 | 21662.45285 | 1990.62088 | 0.00000 | 19671.83119 | 619.70602 | 1.51499 |

history_fraction is physical History / patch support; merged population representation is reflected by population_rep_clusters, not only additive_rep_keys.



## Interpretation

New minus previous physical H/Rep baseline:{"auc03": 0.3274981252343434, "auc30": -0.18530886764155374, "acc": -0.01165915979961038, "comp": 0.0012342920327838292, "cd": -0.0052124338834132755, "f1": 0.8530846428759786}; AUC/F1 percentage points, distances metres.

This is a measured tradeoff, not a claim of universal improvement. Inspect AUC@3,AUC@30 and reconstruction together. No planned fresh-probe efficiency benefit may be attributed to this current implementation.

All quality gates, speed and memory must be considered separately; no thresholds or method settings were altered after observing quality.
