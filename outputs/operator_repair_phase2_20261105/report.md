# Operator-loss and partial-Key repair experiments

Research only. The production layout-fixed reuse method is retained. None of the eight frozen candidates has passed AUC, reconstruction and no-speed-sacrifice jointly. This is not an improved full-dataset result.

## Scope and controls

Quality: pipes, courtyard, playground ETH3D calibration scenes, same checkpoint, exact frozen RGB frame lists and preprocessing; pose plus recon_unposed. Equal-scene means below. Full11-scene layout-fixed reference remains unchanged. No7Scenes truth used to select parameters.

Reconstruction: ACC/Comp/CD in metres, CD=(ACC+Comp)/2 unsquared. F1 threshold0.25m, downsampling/TSDF voxel0.0390625m, truncation0.2m,1M mesh points, existing Umeyama/RANSAC alignment and masking. AUC/F1 in percentages in tables; raw JSON stores fractions.

All candidates preserve the original per-layer patch budgets:0.10 for layers0–5/18–23,0.22 for layers6–17 (mean cap0.16); this is a cap, not measured effective density. Actual unique densities and budget errors are separately stored in summary.json. No extra full Dense QK/oracle at inference.

SAF3R timing control: original repository eth3d-train-fltr-calib_cmpmse.json profile, Top-4 lazy first-correspondence head, not paper DINO Top-K. No new Dense/SAF3R quality evaluation.

## What was changed

1. normalized_moment scores each executable support using its weighted probability mass and projected output moment, including the softmax denominator and protected Keys. Equal support+weights gives equal loss, resolving Rep versus singleton History contradiction under estimated probabilities. It remains a local Dense-background approximation, not the true global all-cluster objective.

2. Add Alternate and Pair states. Alternate is a real cluster member farthest from its current representative in a deterministic head-aware Key sketch; Pair retains representative+alternate. Drop/Rep/History/Exact remain available. The shared global budget chooses among all clusters: no one-Key-per-cluster quota. This expands support coverage but cannot express every arbitrary subset.

3. Phase2 calibrated weights multiply population compensation by clipped estimated mass ratio [0.25,4]. The loss uses the same weight that the sparse kernel executes. This avoids scoring one operator and executing another, but depends on probability estimation accuracy.

4. Phase2 fixes CUDA Graph topology churn through fixed-capacity population bins. Zero warmed creations/evictions are verified, separately from prediction quality.

## Calibration macro quality

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| layout_fixed | 14.764 | 78.158 | 0.308703 | 0.508498 | 0.408601 | 55.403 |
| loss_only | 9.095 | 65.676 | 0.376157 | 0.558139 | 0.467148 | 47.337 |
| states_only | 15.529 | 80.725 | 0.359219 | 0.506298 | 0.432758 | 48.070 |
| both | 8.532 | 68.462 | 0.392727 | 0.623373 | 0.508050 | 48.076 |
| both_mass_robust | 8.918 | 67.237 | 0.366639 | 0.497593 | 0.432116 | 48.395 |
| calibrated_loss | 5.166 | 59.915 | 0.364133 | 0.411003 | 0.387568 | 47.645 |
| calibrated_both | 7.957 | 63.818 | 0.398249 | 0.569285 | 0.483767 | 43.414 |
| calibrated_both_robust | 5.197 | 60.983 | 0.455435 | 0.505315 | 0.480375 | 41.850 |
| states_only_fast | 16.398 | 80.473 | 0.341364 | 0.505542 | 0.423453 | 51.927 |

## 7Scenes heads128 warmed full-forward speed

GPU6 RTX4090,1 warmup+3 repeats, median seconds. Includes reset, clustering, routing, history, reordering and graph work. Excludes file/model loading. Controls explicitly authorized for timing only. All speed-only quality fields are N/A; no geometric quality was requested for these runs. This single scene is not a7Scenes macro.

| Method/batch | Seconds | Dense speedup | Time/layout-fixed | Time/SAF3R | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| loss_only / operator_repair_20261105 | 11.499233 | N/A (earlier batch) | N/A (earlier batch) | N/A (earlier batch) | 15.266 |
| states_only / operator_repair_20261105 | 11.276690 | N/A (earlier batch) | N/A (earlier batch) | N/A (earlier batch) | 15.267 |
| both / operator_repair_20261105 | 11.484027 | N/A (earlier batch) | N/A (earlier batch) | N/A (earlier batch) | 15.250 |
| both_mass_robust / operator_repair_20261105 | 11.329215 | N/A (earlier batch) | N/A (earlier batch) | N/A (earlier batch) | 15.253 |
| calibrated_loss / operator_repair_phase2_20261105 | 9.527273 | 1.787x | 1.0230 | 1.1102 | 15.262 |
| calibrated_both / operator_repair_phase2_20261105 | 9.475989 | 1.796x | 1.0175 | 1.1042 | 15.259 |
| calibrated_both_robust / operator_repair_phase2_20261105 | 9.452200 | 1.801x | 1.0150 | 1.1014 | 15.259 |
| states_only_fast / operator_repair_phase2_20261105 | 9.705677 | 1.754x | 1.0422 | 1.1310 | 15.258 |
| layout_fixed / phase2_controls | 9.312878 | 1.828x | 1.0000 | 1.0852 | 15.373 |
| dense / phase2_controls | 17.021049 | 1.000x | 1.8277 | 1.9834 | 14.359 |
| saf3r_top4_lazy_correspondence / phase2_controls | 8.581854 | 1.983x | 0.9215 | 1.0000 | 14.614 |

## Per-scene quality

| Method / scene | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| layout_fixed / pipes | 19.048 | 84.615 | 0.190343 | 0.302177 | 0.246260 | 78.470 |
| layout_fixed / courtyard | 16.643 | 83.381 | 0.328753 | 0.198338 | 0.263545 | 56.690 |
| layout_fixed / playground | 8.602 | 66.478 | 0.407014 | 1.024979 | 0.715997 | 31.047 |
| loss_only / pipes | 16.117 | 80.183 | 0.257996 | 0.245957 | 0.251977 | 67.585 |
| loss_only / courtyard | 9.151 | 78.625 | 0.339714 | 0.212048 | 0.275881 | 54.609 |
| loss_only / playground | 2.016 | 38.219 | 0.530761 | 1.216411 | 0.873586 | 19.818 |
| states_only / pipes | 15.385 | 85.275 | 0.210715 | 0.232043 | 0.221379 | 80.910 |
| states_only / courtyard | 24.751 | 87.553 | 0.502592 | 0.284183 | 0.393387 | 26.218 |
| states_only / playground | 6.452 | 69.348 | 0.364351 | 1.002667 | 0.683509 | 37.084 |
| both / pipes | 13.553 | 76.374 | 0.294186 | 0.273737 | 0.283961 | 67.259 |
| both / courtyard | 8.345 | 76.183 | 0.288710 | 0.198969 | 0.243839 | 65.439 |
| both / playground | 3.696 | 52.829 | 0.595286 | 1.397414 | 0.996350 | 11.531 |
| both_mass_robust / pipes | 13.553 | 80.256 | 0.230492 | 0.206695 | 0.218594 | 74.363 |
| both_mass_robust / courtyard | 12.328 | 76.999 | 0.395517 | 0.204894 | 0.300206 | 46.235 |
| both_mass_robust / playground | 0.874 | 44.456 | 0.473909 | 1.081189 | 0.777549 | 24.586 |
| calibrated_loss / pipes | 7.692 | 69.451 | 0.257213 | 0.308759 | 0.282986 | 59.760 |
| calibrated_loss / courtyard | 6.259 | 73.722 | 0.434325 | 0.197247 | 0.315786 | 41.197 |
| calibrated_loss / playground | 1.546 | 36.573 | 0.400861 | 0.727004 | 0.563933 | 41.979 |
| calibrated_both / pipes | 13.553 | 72.967 | 0.238069 | 0.245390 | 0.241729 | 71.751 |
| calibrated_both / courtyard | 9.578 | 78.103 | 0.455092 | 0.275664 | 0.365378 | 37.117 |
| calibrated_both / playground | 0.739 | 40.383 | 0.501587 | 1.186802 | 0.844195 | 21.373 |
| calibrated_both_robust / pipes | 9.158 | 71.685 | 0.216315 | 0.218312 | 0.217314 | 74.259 |
| calibrated_both_robust / courtyard | 5.358 | 72.077 | 0.655638 | 0.296434 | 0.476036 | 24.787 |
| calibrated_both_robust / playground | 1.075 | 39.187 | 0.494353 | 1.001198 | 0.747775 | 26.505 |
| states_only_fast / pipes | 15.385 | 85.275 | 0.210715 | 0.232043 | 0.221379 | 80.910 |
| states_only_fast / courtyard | 27.359 | 86.761 | 0.443436 | 0.264173 | 0.353804 | 37.314 |
| states_only_fast / playground | 6.452 | 69.382 | 0.369943 | 1.020411 | 0.695177 | 37.556 |

## Interpretation

Fixing the loss formula alone did not stabilize AUC. Current probabilities are coarse-grid estimates plus historical one-Probe evidence, shared across heads/query groups; a4D Value/output sketch and separable cluster objective leave head/query-specific probability error and cross-cluster softmax interactions. Better loss consistency does not establish better true output error. The oracle used true current attention and a broader mask search, which these deployable candidates do not have.

Repeatability limitation: warmed repeated forwards are not bitwise identical even for the unchanged layout-fixed baseline (pose-encoding relative L2 about0.0053); normalized-loss candidates are around0.02–0.027. The existing reductions/clustering plus discrete state decisions amplify numerical differences. Small single-run AUC improvements are exploratory, not a statistically established gain. Full repeat drift is retained in summary.json and the timing artifacts. Fixed population-bin reductions can also alter rounding versus the phase1 padded layout; states_only_fast is separately evaluated, not declared output-identical.

Partial states sometimes improve pose while damaging unposed reconstruction; this is not a successful repair. Mass calibration and additional states likewise must pass measured gates before adoption. No test-set parameter selection or hidden density increase was used. Failed candidates remain documented rather than replacing the production method.

Diagnostics: unit tests cover identical physical-support losses, normalized predicted error, binned statistics and emitted execution weights. Earlier Triton compile failure is retained as a failed diagnostic with all six quality fields N/A. No candidate was promoted to full ETH3D/7Scenes evaluation after failing calibration gates.
