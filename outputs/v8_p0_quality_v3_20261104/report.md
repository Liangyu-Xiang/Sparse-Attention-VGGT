# Corrected v8 P0 execution prototype

Evaluation complete: **True**. Default unchanged. No test-truth tuning. Only corrected P0 geometry is rerun; Dense/SAF3R/v8 quality is reused. Dense timing rerun is explicitly authorized by the later user request to compare original forward/core.

All7 7Scenes scenes128frames and all11 ETH3D scenes14–76frames; exact frame list, shape, checkpoint, preprocessing and pose+recon_unposed protocol audited. Equal-scene macro, AUC/F1 percentages below, JSON fractions; unsquared distances metres, CD=(ACC+Comp)/2. Threshold/voxel7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. SAF3R local lazy first-correspondence-head Top4/ETH3D profile,not paper DINO Top-K.

| dataset | method | scenes | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense | 7 | True | 23.6689% | 84.6389% | 0.153741 | 0.125616 | 0.139678 | 48.0128% |
| 7scenes | saf3r | 7 | True | 23.9349% | 84.7131% | 0.143263 | 0.127883 | 0.135573 | 47.3154% |
| 7scenes | v8 | 7 | True | 21.9102% | 84.2025% | 0.144386 | 0.121930 | 0.133158 | 46.8951% |
| 7scenes | p0_v2 | 7 | True | 22.7485% | 84.5766% | 0.146029 | 0.120645 | 0.133337 | 47.1591% |
| eth3d | dense | 11 | True | 29.1835% | 81.8162% | 0.313193 | 0.848854 | 0.581024 | 62.5731% |
| eth3d | saf3r | 11 | True | 23.3322% | 80.1747% | 0.303106 | 0.857720 | 0.580413 | 62.8304% |
| eth3d | v8 | 11 | True | 21.9385% | 78.2346% | 0.308629 | 0.872234 | 0.590431 | 59.5813% |
| eth3d | p0_v2 | 11 | True | 22.8875% | 79.2574% | 0.332203 | 0.867187 | 0.599695 | 58.7390% |

Independent matched heads128 timing: {"dense_seconds": 12.642996637616307, "p0_v2_seconds": 21.134578619152308, "dense_speedup": 0.5982138023872939, "scope": "Separate paired heads128 benchmark;not per-scene quality timings", "p0_samples": [{"seconds": 21.134578619152308, "peak_allocated_gib": 18.26276969909668, "peak_reserved_gib": 25.16015625}, {"seconds": 20.684243813622743, "peak_allocated_gib": 18.49738645553589, "peak_reserved_gib": 26.0}, {"seconds": 21.212137987371534, "peak_allocated_gib": 18.50277280807495, "peak_reserved_gib": 26.265625}]}. Each quality scene has one warmup and one measured output, recorded separately and not presented as warmed3-sample timing. Dataset timing speedups N/A; complete-forward paired speedup is only defined for heads128.

The first prototype changed cluster tile size while sharing geometry across2 Query Groups; small loss rounding differences produced a routing mismatch at L4. Its quality queue was stopped without discarding completed scene scores; it is not an accepted equivalent implementation. The corrected kernel preserves original cluster tile/reduction layout while reusing query-independent geometry. Captured-vs-original-eager routing matches Keys/padding/state counts in the first64-group window of all23 changed layers. Maximum weight relativeL2 5.802890612471856e-8; maximum absolute6.103515625e-5 on large weights, within the unchanged rtol1e-5 gate. L0 seed unchanged. Original v8 Graph-vs-eager passed the same check.

The stronger ALL-window routing oracle FAILED at L1 beyond the sampled window. The corrected prototype is NOT accepted as globally equivalent; small floating-point loss differences can change discrete budget decisions. Failure and first-mismatch investigation are preserved. The complete quality evaluation measures a rejected candidate, not an equivalence success.

Direct packing uses per-cluster counts/prefix and direct member writes, then sorts ONLY the retained Key list to preserve original ascending token order. It eliminates full-Patch flags/prefix, but sorting can offset the saving. This initial execution prototype does not implement Cluster–Grid sufficient statistics, does not change beta or probability definition, and does not claim a successful speed optimization.

Frozen-source unchanged: True. Completed audits: 18. Missing scenes: []. Full source archive, hashes, failed diagnostics, numerical checks, six-metric per-scene records and GPU exclusivity monitors are preserved.

Six-metric relative changes versus historical matched v8 (percent): [{"dataset": "7scenes", "auc03_relative_change_percent": 3.826407829295686, "auc30_relative_change_percent": 0.44425550153210924, "acc_relative_change_percent": 1.13744600832566, "comp_relative_change_percent": -1.0531970880958696, "cd_relative_change_percent": 0.1344866524816135, "f1_relative_change_percent": 0.5630992880454988}, {"dataset": "eth3d", "auc03_relative_change_percent": 4.325855583404503, "auc30_relative_change_percent": 1.3073889551866724, "acc_relative_change_percent": 7.638368198966661, "comp_relative_change_percent": -0.5786594257762179, "cd_relative_change_percent": 1.5689319401727353, "f1_relative_change_percent": -1.4137034519980718}]. Differences in floating-point execution can propagate through later discrete decisions; no quality gate was relaxed and no result was selected by repeated successful quality runs.
