# ETH3D regression repair

FROZEN CANDIDATE: pending complete six-metric quality and paired7Scenes speed gate.

Frozen before quality evaluation. Same checkpoint, exact RGB frames, preprocessing, pose, alignment/fusion and reconstruction settings as before_main. Equal-scene macro averages; all 11 ETH3D and all seven 7Scenes scenes required. No 7Scenes test-truth parameter tuning.

Before_main is the unchanged, complete previous batch; it was not rerun. This is a local measured comparison, not a paper reproduction.

| Dataset / scope | Method | Scenes | AUC@3 | AUC@30 | ACC (m) | Comp (m) | CD (m) | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 7scenes | before_main | 7 | 22.167% | 84.056% | 0.137858 | 0.124579 | 0.131219 | 46.993% |
| 7scenes | fast_repair | 7 | 21.892% | 83.899% | 0.139708 | 0.123243 | 0.131476 | 46.966% |
| eth3d | before_main | 11 | 12.276% | 67.005% | 0.415724 | 1.110314 | 0.763019 | 42.649% |
| eth3d | fast_repair | 11 | 16.659% | 73.369% | 0.374640 | 0.906272 | 0.640456 | 53.262% |

CD=(ACC+Comp)/2 with unsquared distances. AUC/F1 in this report are percentages; raw JSON uses fractions. ETH3D threshold/downsample: 0.25 m / 0.0390625 m; 7Scenes: 0.05 m / 0.0078125 m. Pose + recon_unposed complete for each run. This method does not use SAF3R Top-K or its head profile.

## Diagnosis and intervention

Current Dense quality was reproduced on pipes and playground to check baseline/environment drift; see controls/. Independent Dense numerical trajectories were repeated exactly for every diagnostic batch. These numerical diagnostics have all six geometry metrics marked N/A and are not quality evaluations.

The former calibration compared Dense attention on already corrupted sparse Q/K/V, which misses accumulated feature error. Dense Camera Query correction therefore gives a misleadingly small local error. Smaller Query groups and extra clustering iterations were not consistent cures.

The repair preserves every patch Key in the first coordinate-anchor frame inside the existing total Key budget, removes those patches from clustering, and clusters the remaining rectangular frame domain. It also replaces head-mean Value features with separate-head Value features (see frozen config for full concatenation versus a fixed within-head orthogonal sketch), preventing cross-head cancellation in the selection geometry. Other algorithm parameters and layer retention caps remain frozen. First-frame Dense Query correction was tested and rejected, not enabled in the final method.

The anchor ablation improves independent Dense fidelity; this supports a causal contribution, not a claim that all approximation error is eliminated. Complete six-metric results above determine the actual quality outcome.

## Runtime scope

7Scenes uses one full-forward warmup followed by three complete timed forwards per scene. Authorized original Dense and SAF3R Top-4 timing controls use the same inputs/hardware/protocol; paired results and speed gates are in speed.md and paired_speed.json. Control geometric quality is N/A (timing only). Previous batch timings are historical, not paired comparisons. ETH3D uses no warmup and one forward solely for quality; those cold/instrumented times are not speed benchmarks.

7Scenes equal-scene mean of warmed per-scene medians: 9.589701 s; maximum timed allocated memory: 15.366582 GiB; mean actual global-attention sparsity: 84.163088%.

## Per-scene quality

| Dataset / scope | Method | Scenes | AUC@3 | AUC@30 | ACC (m) | Comp (m) | CD (m) | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 7scenes/chess | before_main | 1 | 19.390% | 88.077% | 0.150999 | 0.088098 | 0.119549 | 34.860% |
| 7scenes/chess | fast_repair | 1 | 20.505% | 88.385% | 0.141444 | 0.086584 | 0.114014 | 35.562% |
| 7scenes/fire | before_main | 1 | 53.355% | 94.141% | 0.049492 | 0.052409 | 0.050951 | 67.245% |
| 7scenes/fire | fast_repair | 1 | 51.575% | 94.045% | 0.048970 | 0.053496 | 0.051233 | 67.291% |
| 7scenes/heads | before_main | 1 | 48.573% | 93.365% | 0.034737 | 0.033055 | 0.033896 | 84.112% |
| 7scenes/heads | fast_repair | 1 | 46.453% | 93.107% | 0.033428 | 0.033621 | 0.033524 | 82.577% |
| 7scenes/office | before_main | 1 | 12.881% | 77.331% | 0.049965 | 0.052204 | 0.051085 | 56.852% |
| 7scenes/office | fast_repair | 1 | 14.099% | 77.543% | 0.048492 | 0.052578 | 0.050535 | 57.706% |
| 7scenes/pumpkin | before_main | 1 | 6.574% | 74.646% | 0.366346 | 0.087678 | 0.227012 | 25.990% |
| 7scenes/pumpkin | fast_repair | 1 | 6.709% | 74.827% | 0.389213 | 0.085717 | 0.237465 | 26.475% |
| 7scenes/redkitchen | before_main | 1 | 8.292% | 79.542% | 0.078872 | 0.084248 | 0.081560 | 46.197% |
| 7scenes/redkitchen | fast_repair | 1 | 7.825% | 79.489% | 0.083714 | 0.089840 | 0.086777 | 44.201% |
| 7scenes/stairs | before_main | 1 | 6.102% | 81.286% | 0.234594 | 0.474361 | 0.354478 | 13.696% |
| 7scenes/stairs | fast_repair | 1 | 6.078% | 79.900% | 0.232695 | 0.460868 | 0.346781 | 14.951% |
| eth3d/courtyard | before_main | 1 | 11.238% | 78.397% | 0.447638 | 0.241046 | 0.344342 | 41.056% |
| eth3d/courtyard | fast_repair | 1 | 7.397% | 74.699% | 0.331437 | 0.168741 | 0.250089 | 54.325% |
| eth3d/electro | before_main | 1 | 22.627% | 88.012% | 0.293715 | 4.877619 | 2.585667 | 53.477% |
| eth3d/electro | fast_repair | 1 | 25.011% | 90.081% | 0.318466 | 4.907537 | 2.613002 | 50.933% |
| eth3d/kicker | before_main | 1 | 40.573% | 92.330% | 0.078650 | 0.070797 | 0.074723 | 98.650% |
| eth3d/kicker | fast_repair | 1 | 49.606% | 93.219% | 0.067761 | 0.059332 | 0.063546 | 98.885% |
| eth3d/pipes | before_main | 1 | 4.396% | 60.842% | 0.331373 | 0.399578 | 0.365475 | 34.401% |
| eth3d/pipes | fast_repair | 1 | 22.711% | 83.590% | 0.313044 | 0.237744 | 0.275394 | 70.305% |
| eth3d/relief | before_main | 1 | 8.382% | 68.363% | 0.734689 | 0.932143 | 0.833416 | 1.408% |
| eth3d/relief | fast_repair | 1 | 3.314% | 64.912% | 0.823328 | 0.887828 | 0.855578 | 1.070% |
| eth3d/delivery_area | before_main | 1 | 7.009% | 73.684% | 0.577886 | 1.270277 | 0.924081 | 26.268% |
| eth3d/delivery_area | fast_repair | 1 | 12.308% | 76.983% | 0.754026 | 1.560876 | 1.157451 | 8.319% |
| eth3d/facade | before_main | 1 | 18.643% | 84.720% | 0.417969 | 0.432425 | 0.425197 | 51.407% |
| eth3d/facade | fast_repair | 1 | 27.111% | 86.793% | 0.335447 | 0.336896 | 0.336172 | 59.892% |
| eth3d/office | before_main | 1 | 18.051% | 76.133% | 0.120005 | 0.109096 | 0.114551 | 97.435% |
| eth3d/office | fast_repair | 1 | 22.051% | 74.215% | 0.058990 | 0.056035 | 0.057513 | 99.285% |
| eth3d/playground | before_main | 1 | 2.487% | 55.538% | 0.514113 | 1.154362 | 0.834238 | 20.621% |
| eth3d/playground | fast_repair | 1 | 5.376% | 57.970% | 0.459892 | 1.075039 | 0.767466 | 31.176% |
| eth3d/relief_2 | before_main | 1 | 1.053% | 36.333% | 0.520484 | 0.597489 | 0.558987 | 32.531% |
| eth3d/relief_2 | fast_repair | 1 | 2.982% | 41.000% | 0.358216 | 0.477743 | 0.417980 | 39.817% |
| eth3d/terrains | before_main | 1 | 0.581% | 22.706% | 0.536441 | 2.128624 | 1.332532 | 11.881% |
| eth3d/terrains | fast_repair | 1 | 5.381% | 63.597% | 0.300429 | 0.201217 | 0.250823 | 71.877% |
