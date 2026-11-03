# Historical Rep / budget repair: GPU evaluation

Requested remote branch: fix/gpt6-single-probe-history-20261103; tested source commit `bb59688bf13c5fa104e658f4ec6a9825ca634fdf`. Complete: **True**. No model formula/configuration was changed during evaluation. The only evaluator change archives the newly added repair module in each source manifest.

Evaluation completeness is separate from method acceptance. All-six1% relative-degradation check versus grid_local across both datasets: {'single_probe_repaired': False, 'full_repaired': False}. A recovery from the old collapsed result is not proof of quality equivalence.

Only the two changed repaired methods ran. Dense, SAF3R, grid_local, previous optimized v8 and old single/full quality are reused historical controls (original batch paths and raw results preserved in references/). No paper scores are mixed in. SAF3R is lazy first-correspondence-head Top4, ETH3D profile, not paper DINO feature Top-K.

7Scenes: all seven scenes,128 uniform frames. ETH3D: all11 scenes,14–76 uniform frames. Same checkpoint, exact image list/frame indices, preprocessing and pose/unposed reconstruction protocol audited per scene. AUC/F1 are percentages below; ACC/Comp/CD are unsquared metres, CD=(ACC+Comp)/2. JSON/CSV AUC/F1 values are fractions. Reconstruction threshold/voxel:7Scenes0.05m/0.0078125m; ETH3D0.25m/0.0390625m. Equal-scene macro averages.

| dataset | method | scenes | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense | 7 | True | 23.6689% | 84.6389% | 0.153741 | 0.125616 | 0.139678 | 48.0128% |
| 7scenes | saf3r | 7 | True | 23.9349% | 84.7131% | 0.143263 | 0.127883 | 0.135573 | 47.3154% |
| 7scenes | grid_local | 7 | True | 22.1234% | 84.4171% | 0.146727 | 0.124312 | 0.135520 | 45.2177% |
| 7scenes | previous_optimized_v8 | 7 | True | 21.9102% | 84.2025% | 0.144386 | 0.121930 | 0.133158 | 46.8951% |
| 7scenes | old_single_probe | 7 | True | 0.0375% | 11.3659% | 3.792835 | 0.532620 | 2.162728 | 1.9536% |
| 7scenes | old_full | 7 | True | 6.4129% | 66.3552% | 0.186813 | 0.232534 | 0.209674 | 28.1502% |
| 7scenes | single_probe_repaired | 7 | True | 21.6869% | 83.7570% | 0.145132 | 0.128173 | 0.136652 | 45.3525% |
| 7scenes | full_repaired | 7 | True | 20.8773% | 83.6029% | 0.142365 | 0.130925 | 0.136645 | 44.6388% |
| eth3d | dense | 11 | True | 29.1835% | 81.8162% | 0.313193 | 0.848854 | 0.581024 | 62.5731% |
| eth3d | saf3r | 11 | True | 23.3322% | 80.1747% | 0.303106 | 0.857720 | 0.580413 | 62.8304% |
| eth3d | grid_local | 11 | True | 23.9534% | 79.2698% | 0.326859 | 0.875317 | 0.601088 | 56.8682% |
| eth3d | previous_optimized_v8 | 11 | True | 21.9385% | 78.2346% | 0.308629 | 0.872234 | 0.590431 | 59.5813% |
| eth3d | old_single_probe | 11 | True | 0.2127% | 12.7805% | 1.424884 | 5.403268 | 3.414076 | 12.6518% |
| eth3d | old_full | 11 | True | 7.7689% | 63.5083% | 0.486222 | 1.104045 | 0.795134 | 43.7542% |
| eth3d | single_probe_repaired | 11 | True | 18.6899% | 75.3391% | 0.408199 | 0.974275 | 0.691237 | 45.8899% |
| eth3d | full_repaired | 11 | True | 20.8233% | 76.1835% | 0.421680 | 0.990915 | 0.706297 | 46.6610% |

## Actual global attention retention

The20% cap, Drop/Rep/History/Exact states and original clustering configuration are retained. This is separate from the earlier layered no-Drop/Rep-always method. Retention below counts global attention edges including special keys, averaged equally over scenes and layers; it does not describe whole-model FLOPs.

| dataset | method | global_attention_retained_fraction | global_attention_sparse_fraction | mean_patch_budget_fill |
| --- | --- | --- | --- | --- |
| 7scenes | single_probe_repaired | 0.199983 | 0.800017 | 0.999938 |
| eth3d | single_probe_repaired | 0.199935 | 0.800065 | 0.999665 |
| 7scenes | full_repaired | 0.199983 | 0.800017 | 0.999934 |
| eth3d | full_repaired | 0.199932 | 0.800068 | 0.999649 |

## Full-forward runtime and memory

Timing: fixed 7Scenes heads128, RTX5090 D GPU7, one warmup and median of three complete forwards. Includes reset, initialization within forward, clustering, routing, budget repair sorting, history, reordering and output heads; excludes model/image loading and GT evaluation. All GPU4–7 jobs have CPU-only locking/monitoring; one CUDA process per physical GPU, held until process exit. Quality-run one-sample timings are retained separately and are not benchmark medians. Dense speedup is N/A: unchanged Dense was not rerun. Dense identity is1.00x. Historical times from other batches are not paired speedups.

| method | median_seconds | cold_seconds | peak_allocated_gib | gpu | dense_speedup | exclusive_monitor_passed |
| --- | --- | --- | --- | --- | --- | --- |
| single_probe_repaired | 14.634211 | 21.453653 | 15.778574 | 7 | N/A | True |
| full_repaired | 15.532334 | 16.686047 | 15.777915 | 7 | N/A | True |

## Disjoint stage timing

Independent event profile after the three benchmark samples; insertion overhead makes its wall time different from the warmed median. Percentages use the profile wall time. Budget includes both the16-step solver and the new two-pass greedy repair/sort. Unattributed includes the other model layers, CPU launch gaps and uninstrumented work; it is not all mask selection. History producer is charged once.

| method | operation | seconds | percent | profile_wall_seconds |
| --- | --- | --- | --- | --- |
| single_probe_repaired | light_qkv | 0.520812 | 3.471739 | 15.001462 |
| single_probe_repaired | light_clustering | 1.189922 | 7.932039 | 15.001462 |
| single_probe_repaired | light_geometry | 0.070021 | 0.466764 | 15.001462 |
| single_probe_repaired | light_observation | 0.125415 | 0.836020 | 15.001462 |
| single_probe_repaired | light_attention | 3.378901 | 22.523809 | 15.001462 |
| single_probe_repaired | light_history_producer | 0.344029 | 2.293301 | 15.001462 |
| single_probe_repaired | light_prior | 0.290492 | 1.936422 | 15.001462 |
| single_probe_repaired | light_exact_qualification | 0.006046 | 0.040300 | 15.001462 |
| single_probe_repaired | light_state_loss | 1.459793 | 9.731006 | 15.001462 |
| single_probe_repaired | light_budget | 2.218390 | 14.787828 | 15.001462 |
| single_probe_repaired | light_pack | 0.411138 | 2.740651 | 15.001462 |
| single_probe_repaired | unattributed | 4.986504 | 33.240121 | 15.001462 |
| full_repaired | light_qkv | 0.521563 | 3.387055 | 15.398721 |
| full_repaired | light_clustering | 1.169233 | 7.593053 | 15.398721 |
| full_repaired | light_geometry | 0.069794 | 0.453245 | 15.398721 |
| full_repaired | light_observation | 0.103850 | 0.674404 | 15.398721 |
| full_repaired | light_attention | 3.745167 | 24.321289 | 15.398721 |
| full_repaired | light_history_producer | 0.681271 | 4.424206 | 15.398721 |
| full_repaired | light_prior | 0.307250 | 1.995295 | 15.398721 |
| full_repaired | light_exact_qualification | 0.005965 | 0.038736 | 15.398721 |
| full_repaired | light_state_loss | 1.467140 | 9.527672 | 15.398721 |
| full_repaired | light_budget | 2.216744 | 14.395640 | 15.398721 |
| full_repaired | light_pack | 0.409756 | 2.660972 | 15.398721 |
| full_repaired | unattributed | 4.700988 | 30.528434 | 15.398721 |

A separate diagnostic splits solver/repair with temporary function event wrappers, restored after profiling; model math and files are unchanged. Nested children replace their budget parent in the disjoint sum. This independent sample must not be added to the earlier profile table. Detailed raw result: detailed_budget_profile/result.json.

| operation | seconds |
| --- | --- |
| detail_budget_solver | 0.308975 |
| detail_budget_repair | 1.947667 |
| detail_budget_other | 0.025436 |

## Recovery comparisons

relative_changes.csv separates changes against old collapsed variants, previous optimized v8, grid_local and matched local Dense. The recorded1% relative-degradation check is descriptive; passing it does not prove mathematical equivalence. No parameter was tuned against test truth.

- 7scenes / single_probe_repaired vs dense: all-six degradation within1% = False; AUC@3 -1.9820 pp, AUC@30 -0.8819 pp, CD -2.167%, F1 -2.6602 pp.
- 7scenes / single_probe_repaired vs grid_local: all-six degradation within1% = False; AUC@3 -0.4365 pp, AUC@30 -0.6601 pp, CD +0.836%, F1 +0.1348 pp.
- 7scenes / single_probe_repaired vs previous_optimized_v8: all-six degradation within1% = False; AUC@3 -0.2232 pp, AUC@30 -0.4455 pp, CD +2.624%, F1 -1.5425 pp.
- 7scenes / single_probe_repaired vs old_single_probe: all-six degradation within1% = True; AUC@3 +21.6494 pp, AUC@30 +72.3911 pp, CD -93.681%, F1 +43.3989 pp.
- 7scenes / full_repaired vs dense: all-six degradation within1% = False; AUC@3 -2.7916 pp, AUC@30 -1.0360 pp, CD -2.172%, F1 -3.3739 pp.
- 7scenes / full_repaired vs grid_local: all-six degradation within1% = False; AUC@3 -1.2461 pp, AUC@30 -0.8142 pp, CD +0.830%, F1 -0.5789 pp.
- 7scenes / full_repaired vs previous_optimized_v8: all-six degradation within1% = False; AUC@3 -1.0329 pp, AUC@30 -0.5996 pp, CD +2.619%, F1 -2.2562 pp.
- 7scenes / full_repaired vs old_full: all-six degradation within1% = True; AUC@3 +14.4644 pp, AUC@30 +17.2477 pp, CD -34.830%, F1 +16.4886 pp.
- eth3d / single_probe_repaired vs dense: all-six degradation within1% = False; AUC@3 -10.4935 pp, AUC@30 -6.4771 pp, CD +18.969%, F1 -16.6833 pp.
- eth3d / single_probe_repaired vs grid_local: all-six degradation within1% = False; AUC@3 -5.2635 pp, AUC@30 -3.9307 pp, CD +14.998%, F1 -10.9783 pp.
- eth3d / single_probe_repaired vs previous_optimized_v8: all-six degradation within1% = False; AUC@3 -3.2485 pp, AUC@30 -2.8955 pp, CD +17.073%, F1 -13.6914 pp.
- eth3d / single_probe_repaired vs old_single_probe: all-six degradation within1% = True; AUC@3 +18.4772 pp, AUC@30 +62.5586 pp, CD -79.753%, F1 +33.2381 pp.
- eth3d / full_repaired vs dense: all-six degradation within1% = False; AUC@3 -8.3601 pp, AUC@30 -5.6327 pp, CD +21.561%, F1 -15.9122 pp.
- eth3d / full_repaired vs grid_local: all-six degradation within1% = False; AUC@3 -3.1300 pp, AUC@30 -3.0862 pp, CD +17.503%, F1 -10.2072 pp.
- eth3d / full_repaired vs previous_optimized_v8: all-six degradation within1% = False; AUC@3 -1.1151 pp, AUC@30 -2.0510 pp, CD +19.624%, F1 -12.9203 pp.
- eth3d / full_repaired vs old_full: all-six degradation within1% = True; AUC@3 +13.0545 pp, AUC@30 +12.6753 pp, CD -11.173%, F1 +2.9068 pp.

Per-scene raw metrics and six-metric Markdown tables: quality/. Layer counts, state usage and actual global attention density: per_layer.csv. Source snapshot and SHA256 are frozen before validation; unchanged: True. Numerical-only and CPU-only tests report all six quality fields unavailable with concrete reasons. Missing artifacts: [].

Reproduce the summary using `python scripts/report_history_fix_evaluation.py --output-dir outputs/gpt6_history_fix_eval_20261103`. A complete reproduction of GPU runs needs external data/checkpoints and a fresh output directory.
