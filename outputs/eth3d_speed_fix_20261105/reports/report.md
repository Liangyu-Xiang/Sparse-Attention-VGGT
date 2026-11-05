# ETH3D regression repair: grouped SAF3R

Frozen before dataset quality. 7Scenes: all seven scenes,128frames each. ETH3D: all11scenes, up to128frames (all available selected valid frames). Same checkpoint SHA-256, exact input paths/indices, preprocessing, pose protocol, alignment/fusion, downsampling and thresholds as before_main. Dataset rows are equal-scene macro averages. No7Scenes test-truth parameter selection.

| Scope | Method | AUC@3 | AUC@30 | ACC (m) | Comp (m) | CD (m) | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 7scenes (7/7, equal-scene) | before_main | 22.167% | 84.056% | 0.137858 | 0.124579 | 0.131219 | 46.993% |
| 7scenes (7/7, equal-scene) | grouped_saf3r | 23.563% | 84.418% | 0.129600 | 0.124892 | 0.127246 | 49.206% |
| eth3d (11/11, equal-scene) | before_main | 12.276% | 67.005% | 0.415724 | 1.110314 | 0.763019 | 42.649% |
| eth3d (11/11, equal-scene) | grouped_saf3r | 23.226% | 80.189% | 0.303194 | 0.821517 | 0.562355 | 63.556% |

AUC/F1 are percentages in this table and fractions in raw JSON. Distances are metres. CD=(ACC+Comp)/2, unsquared distances. 7Scenes reconstruction threshold/downsample:0.05m /0.0078125m. ETH3D:0.25m /0.0390625m. Every new quality run includes pose and recon_unposed.

## Adopted implementation and diagnosis

The final implementation uses the original frozen SAF3R per-head routes, batches identical static patterns, and retains the allocator cache instead of purging it after every block. Dynamic query-probe heads and lazy correspondence heads use their original scalar implementation. Image-dependent correspondence Top-K is reset/recomputed inside every complete-forward timer. Static geometry-index banks cache positions only, never image content.

The failed shared-Value method used a fixed10% /22% /10% layer schedule and one support bank across heads. That schedule was calibrated at128frames and did not reproduce the original SAF3R masks on shorter ETH3D sequences. On pipes, its measured global retained density was15.963%, while the original SAF3R profile retains20.201%; late stride-4 heads also retain the anchor/special-token union. This is a real budget/layout difference, not an evaluation-score adjustment. The final masks have exactly the original SAF3R logical QK counts on every scene, independently checked from the profile.

A second diagnostic issue was local calibration on already corrupted sparse Q/K/V. Tiny local Camera Query error hid accumulated independent-Dense feature error. Protecting anchor Keys and preserving separate-head Value features improved the approximation, but those research variants either remained too slow or retained significant geometric loss. They were not promoted. The final route change is explicit: the shared-Value selector is not part of the adopted inference path.

Current Dense quality was reproduced on pipes and playground in the regression-diagnosis batch; baseline drift did not explain the severe loss. The grouped implementation was also checked against original SAF3R on a disjoint RGB128-frame calibration: pose, full depth and depth-confidence outputs were elementwise equal, including after allocator-cache reuse. Those numerical/timing diagnostics have six geometric fields N/A; they are not dataset quality evaluations.

SAF3R configuration: repository eth3d-train-fltr-calib_cmpmse.json, lazy first-correspondence-head Top-4 per frame (512 selected patch indices at128frames). This is not the paper DINO-feature Top-K, not Top-128, and not a claimed paper reproduction. No published/H100 reference is used as a local timing baseline.

## Paired speed

See [speed.md](speed.md) and paired_speed.json. Original VGGT Dense, original SAF3R and grouped SAF3R share exact inputs, physical GPU, CPU affinity, checkpoint and1warmup+3complete-forward timing per scene. All model stages, routing/Top-K resets, grouping/gathers/scatters and cache misses are included. Loading checkpoint/images is outside full-forward timing for all methods. Original controls retain their original per-block cache purge; only the explicit grouped candidate changes that allocator policy.

The14control runs were explicitly timing-only. All six quality values remain N/A with the concrete reason; no control quality scores are fabricated. ETH3D cold quality-forward times are not a speed benchmark. Earlier batches and noisy development timings are not used as paired speed baselines. Peak allocated and reserved memory are recorded separately in raw timings; allocator-cache reuse can increase reserved memory.

## Per-scene quality

| Scope | Method | AUC@3 | AUC@30 | ACC (m) | Comp (m) | CD (m) | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 7scenes/chess | before_main | 19.390% | 88.077% | 0.150999 | 0.088098 | 0.119549 | 34.860% |
| 7scenes/chess | grouped_saf3r | 20.095% | 88.358% | 0.142800 | 0.081005 | 0.111902 | 37.806% |
| 7scenes/fire | before_main | 53.355% | 94.141% | 0.049492 | 0.052409 | 0.050951 | 67.245% |
| 7scenes/fire | grouped_saf3r | 56.344% | 94.571% | 0.048995 | 0.054636 | 0.051816 | 66.939% |
| 7scenes/heads | before_main | 48.573% | 93.365% | 0.034737 | 0.033055 | 0.033896 | 84.112% |
| 7scenes/heads | grouped_saf3r | 51.161% | 93.897% | 0.032713 | 0.029945 | 0.031329 | 85.105% |
| 7scenes/office | before_main | 12.881% | 77.331% | 0.049965 | 0.052204 | 0.051085 | 56.852% |
| 7scenes/office | grouped_saf3r | 13.324% | 75.779% | 0.047581 | 0.056709 | 0.052145 | 58.721% |
| 7scenes/pumpkin | before_main | 6.574% | 74.646% | 0.366346 | 0.087678 | 0.227012 | 25.990% |
| 7scenes/pumpkin | grouped_saf3r | 7.140% | 74.765% | 0.356595 | 0.075745 | 0.216170 | 37.054% |
| 7scenes/redkitchen | before_main | 8.292% | 79.542% | 0.078872 | 0.084248 | 0.081560 | 46.197% |
| 7scenes/redkitchen | grouped_saf3r | 8.493% | 80.106% | 0.080361 | 0.086195 | 0.083278 | 42.337% |
| 7scenes/stairs | before_main | 6.102% | 81.286% | 0.234594 | 0.474361 | 0.354478 | 13.696% |
| 7scenes/stairs | grouped_saf3r | 8.387% | 83.450% | 0.198156 | 0.490010 | 0.344083 | 16.477% |
| eth3d/courtyard | before_main | 11.238% | 78.397% | 0.447638 | 0.241046 | 0.344342 | 41.056% |
| eth3d/courtyard | grouped_saf3r | 18.587% | 84.509% | 0.192665 | 0.149764 | 0.171214 | 75.184% |
| eth3d/electro | before_main | 22.627% | 88.012% | 0.293715 | 4.877619 | 2.585667 | 53.477% |
| eth3d/electro | grouped_saf3r | 24.471% | 90.229% | 0.339880 | 4.900288 | 2.620084 | 44.457% |
| eth3d/kicker | before_main | 40.573% | 92.330% | 0.078650 | 0.070797 | 0.074723 | 98.650% |
| eth3d/kicker | grouped_saf3r | 51.828% | 94.882% | 0.060529 | 0.055513 | 0.058021 | 99.464% |
| eth3d/pipes | before_main | 4.396% | 60.842% | 0.331373 | 0.399578 | 0.365475 | 34.401% |
| eth3d/pipes | grouped_saf3r | 35.897% | 92.564% | 0.213776 | 0.214880 | 0.214328 | 81.654% |
| eth3d/relief | before_main | 8.382% | 68.363% | 0.734689 | 0.932143 | 0.833416 | 1.408% |
| eth3d/relief | grouped_saf3r | 21.637% | 83.314% | 0.534322 | 0.550075 | 0.542199 | 27.693% |
| eth3d/delivery_area | before_main | 7.009% | 73.684% | 0.577886 | 1.270277 | 0.924081 | 26.268% |
| eth3d/delivery_area | grouped_saf3r | 15.641% | 80.607% | 0.533845 | 1.292645 | 0.913245 | 34.492% |
| eth3d/facade | before_main | 18.643% | 84.720% | 0.417969 | 0.432425 | 0.425197 | 51.407% |
| eth3d/facade | grouped_saf3r | 32.608% | 89.551% | 0.302568 | 0.282734 | 0.292651 | 58.521% |
| eth3d/office | before_main | 18.051% | 76.133% | 0.120005 | 0.109096 | 0.114551 | 97.435% |
| eth3d/office | grouped_saf3r | 25.744% | 76.851% | 0.098608 | 0.101761 | 0.100185 | 98.774% |
| eth3d/playground | before_main | 2.487% | 55.538% | 0.514113 | 1.154362 | 0.834238 | 20.621% |
| eth3d/playground | grouped_saf3r | 15.188% | 72.681% | 0.277195 | 0.724574 | 0.500885 | 69.893% |
| eth3d/relief_2 | before_main | 1.053% | 36.333% | 0.520484 | 0.597489 | 0.558987 | 32.531% |
| eth3d/relief_2 | grouped_saf3r | 5.789% | 51.123% | 0.404297 | 0.548866 | 0.476581 | 41.244% |
| eth3d/terrains | before_main | 0.581% | 22.706% | 0.536441 | 2.128624 | 1.332532 | 11.881% |
| eth3d/terrains | grouped_saf3r | 8.091% | 65.768% | 0.377449 | 0.215584 | 0.296517 | 67.738% |

## Sparsity

| Dataset / scene | Frames | Global main-attention sparsity | Including unchanged frame attention |
|---|---:|---:|---:|
| 7scenes/chess | 128 | 84.1468% | 83.4945% |
| 7scenes/fire | 128 | 84.1468% | 83.4945% |
| 7scenes/heads | 128 | 84.1468% | 83.4945% |
| 7scenes/office | 128 | 84.1468% | 83.4945% |
| 7scenes/pumpkin | 128 | 84.1468% | 83.4945% |
| 7scenes/redkitchen | 128 | 84.1468% | 83.4945% |
| 7scenes/stairs | 128 | 84.1468% | 83.4945% |
| eth3d/courtyard | 38 | 82.8707% | 80.7458% |
| eth3d/electro | 39 | 82.9170% | 80.8441% |
| eth3d/kicker | 31 | 82.4667% | 79.8896% |
| eth3d/pipes | 14 | 79.7991% | 74.4792% |
| eth3d/relief | 19 | 81.0795% | 77.0255% |
| eth3d/delivery_area | 40 | 82.9605% | 80.9370% |
| eth3d/facade | 76 | 83.7664% | 82.6785% |
| eth3d/office | 26 | 82.0441% | 79.0054% |
| eth3d/playground | 32 | 82.5351% | 80.0341% |
| eth3d/relief_2 | 20 | 81.2583% | 77.3889% |
| eth3d/terrains | 42 | 83.0413% | 81.1101% |

These are logical attention-QK sparsities, not whole-model FLOP reductions. Query-probe and lazy correspondence selection costs are fully timed and separately listed in sparsity.csv/summary.json.

All18input/protocol/source/CPU/GPU/density audits passed. Raw per-run six-metric tables, metrics, source snapshots, input lists and GPU monitors remain under final_runs/. The earlier full-head and compact-Value campaigns remain separate research artifacts.
