# ETH3D regression repair

REJECTED as final method: quality improves, but the measured 7Scenes cost violates the user requirement not to sacrifice speed. This is a retained research candidate.

Frozen before quality evaluation. Same checkpoint, exact RGB frames, preprocessing, pose, alignment/fusion and reconstruction settings as before_main. Equal-scene macro averages; all 11 ETH3D and all seven 7Scenes scenes required. No 7Scenes test-truth parameter tuning.

Before_main is the unchanged, complete previous batch; it was not rerun. This is a local measured comparison, not a paper reproduction.

| Dataset / scope | Method | Scenes | AUC@3 | AUC@30 | ACC (m) | Comp (m) | CD (m) | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eth3d | before_main | 11 | 12.276% | 67.005% | 0.415724 | 1.110314 | 0.763019 | 42.649% |
| eth3d | anchor_concat | 11 | 17.388% | 74.689% | 0.383356 | 0.927922 | 0.655639 | 51.338% |
| 7scenes | before_main | 7 | 22.167% | 84.056% | 0.137858 | 0.124579 | 0.131219 | 46.993% |
| 7scenes | anchor_concat | 7 | 22.624% | 84.100% | 0.139651 | 0.123345 | 0.131498 | 47.432% |

CD=(ACC+Comp)/2 with unsquared distances. AUC/F1 in this report are percentages; raw JSON uses fractions. ETH3D threshold/downsample: 0.25 m / 0.0390625 m; 7Scenes: 0.05 m / 0.0078125 m. Pose + recon_unposed complete for each run. This method does not use SAF3R Top-K or its head profile.

## Diagnosis and intervention

Current Dense quality was reproduced on pipes and playground to check baseline/environment drift; see controls/. Independent Dense numerical trajectories were repeated exactly for every diagnostic batch. These numerical diagnostics have all six geometry metrics marked N/A and are not quality evaluations.

The former calibration compared Dense attention on already corrupted sparse Q/K/V, which misses accumulated feature error. Dense Camera Query correction therefore gives a misleadingly small local error. Smaller Query groups and extra clustering iterations were not consistent cures.

The repair preserves every patch Key in the first coordinate-anchor frame inside the existing total Key budget, removes those patches from clustering, and clusters the remaining rectangular frame domain. It also replaces head-mean Value features with separate-head Value features (see frozen config for full concatenation versus a fixed within-head orthogonal sketch), preventing cross-head cancellation in the selection geometry. Other algorithm parameters and layer retention caps remain frozen. First-frame Dense Query correction was tested and rejected, not enabled in the final method.

The anchor ablation improves independent Dense fidelity; this supports a causal contribution, not a claim that all approximation error is eliminated. Complete six-metric results above determine the actual quality outcome.

## Runtime scope

7Scenes uses one full-forward warmup followed by three complete timed forwards per scene. End-to-end speedup relative to Dense and the SAF3R +10% target are N/A: unchanged controls were not rerun for timing. Previous batch timings are historical, not paired comparisons. ETH3D uses no warmup and one forward solely for quality; those cold/instrumented times are not speed benchmarks.

7Scenes equal-scene mean of warmed per-scene medians: 12.759013 s; maximum timed allocated memory: 17.013790 GiB; mean actual global-attention sparsity: 84.150466%.

## Per-scene quality

| Dataset / scope | Method | Scenes | AUC@3 | AUC@30 | ACC (m) | Comp (m) | CD (m) | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eth3d/courtyard | before_main | 1 | 11.238% | 78.397% | 0.447638 | 0.241046 | 0.344342 | 41.056% |
| eth3d/courtyard | anchor_concat | 1 | 13.893% | 79.649% | 0.399539 | 0.208942 | 0.304241 | 43.193% |
| eth3d/electro | before_main | 1 | 22.627% | 88.012% | 0.293715 | 4.877619 | 2.585667 | 53.477% |
| eth3d/electro | anchor_concat | 1 | 25.956% | 89.600% | 0.282097 | 4.837286 | 2.559692 | 54.503% |
| eth3d/kicker | before_main | 1 | 40.573% | 92.330% | 0.078650 | 0.070797 | 0.074723 | 98.650% |
| eth3d/kicker | anchor_concat | 1 | 46.380% | 92.932% | 0.068769 | 0.058752 | 0.063761 | 98.681% |
| eth3d/pipes | before_main | 1 | 4.396% | 60.842% | 0.331373 | 0.399578 | 0.365475 | 34.401% |
| eth3d/pipes | anchor_concat | 1 | 18.315% | 86.447% | 0.189013 | 0.319694 | 0.254354 | 77.338% |
| eth3d/relief | before_main | 1 | 8.382% | 68.363% | 0.734689 | 0.932143 | 0.833416 | 1.408% |
| eth3d/relief | anchor_concat | 1 | 15.400% | 73.294% | 0.713676 | 0.811724 | 0.762700 | 2.150% |
| eth3d/delivery_area | before_main | 1 | 7.009% | 73.684% | 0.577886 | 1.270277 | 0.924081 | 26.268% |
| eth3d/delivery_area | anchor_concat | 1 | 11.795% | 70.936% | 0.554290 | 1.167619 | 0.860955 | 29.251% |
| eth3d/facade | before_main | 1 | 18.643% | 84.720% | 0.417969 | 0.432425 | 0.425197 | 51.407% |
| eth3d/facade | anchor_concat | 1 | 20.994% | 86.083% | 0.313358 | 0.277531 | 0.295444 | 61.604% |
| eth3d/office | before_main | 1 | 18.051% | 76.133% | 0.120005 | 0.109096 | 0.114551 | 97.435% |
| eth3d/office | anchor_concat | 1 | 23.795% | 78.462% | 0.094694 | 0.092811 | 0.093752 | 98.798% |
| eth3d/playground | before_main | 1 | 2.487% | 55.538% | 0.514113 | 1.154362 | 0.834238 | 20.621% |
| eth3d/playground | anchor_concat | 1 | 4.839% | 59.577% | 0.460838 | 1.002817 | 0.731827 | 30.140% |
| eth3d/relief_2 | before_main | 1 | 1.053% | 36.333% | 0.520484 | 0.597489 | 0.558987 | 32.531% |
| eth3d/relief_2 | anchor_concat | 1 | 4.561% | 42.105% | 0.803314 | 1.182848 | 0.993081 | 6.647% |
| eth3d/terrains | before_main | 1 | 0.581% | 22.706% | 0.536441 | 2.128624 | 1.332532 | 11.881% |
| eth3d/terrains | anchor_concat | 1 | 5.343% | 62.497% | 0.337328 | 0.247114 | 0.292221 | 62.418% |
| 7scenes/chess | before_main | 1 | 19.390% | 88.077% | 0.150999 | 0.088098 | 0.119549 | 34.860% |
| 7scenes/chess | anchor_concat | 1 | 19.738% | 88.260% | 0.142714 | 0.086328 | 0.114521 | 36.905% |
| 7scenes/fire | before_main | 1 | 53.355% | 94.141% | 0.049492 | 0.052409 | 0.050951 | 67.245% |
| 7scenes/fire | anchor_concat | 1 | 52.731% | 94.030% | 0.050804 | 0.053047 | 0.051926 | 67.727% |
| 7scenes/heads | before_main | 1 | 48.573% | 93.365% | 0.034737 | 0.033055 | 0.033896 | 84.112% |
| 7scenes/heads | anchor_concat | 1 | 50.262% | 93.618% | 0.034563 | 0.033319 | 0.033941 | 83.206% |
| 7scenes/office | before_main | 1 | 12.881% | 77.331% | 0.049965 | 0.052204 | 0.051085 | 56.852% |
| 7scenes/office | anchor_concat | 1 | 14.001% | 77.692% | 0.049117 | 0.052333 | 0.050725 | 57.907% |
| 7scenes/pumpkin | before_main | 1 | 6.574% | 74.646% | 0.366346 | 0.087678 | 0.227012 | 25.990% |
| 7scenes/pumpkin | anchor_concat | 1 | 6.795% | 74.954% | 0.386643 | 0.079125 | 0.232884 | 28.380% |
| 7scenes/redkitchen | before_main | 1 | 8.292% | 79.542% | 0.078872 | 0.084248 | 0.081560 | 46.197% |
| 7scenes/redkitchen | anchor_concat | 1 | 7.956% | 79.525% | 0.079163 | 0.087760 | 0.083462 | 44.389% |
| 7scenes/stairs | before_main | 1 | 6.102% | 81.286% | 0.234594 | 0.474361 | 0.354478 | 13.696% |
| 7scenes/stairs | anchor_concat | 1 | 6.886% | 80.620% | 0.234556 | 0.471499 | 0.353028 | 13.505% |
