# Beta + current Grid mass: changed-method experiment

Evaluation complete: **True**. **Not accepted: severe geometry quality regression.** This models actual compensated sparse Softmax P as a cross-layer prior; it is not an equivalent v8 optimization. Config and source were frozen before quality results; no parameter/gate/budget was tuned against test truth.

Only beta_gridmass ran. Dense/SAF3R/grid_local/v8 rows are reused complete historical controls; raw results and original batch paths are preserved in references/.7Scenes all7 scenes128frames; ETH3D all11 scenes14–76frames. Same exact inputs/checkpoint/preprocessing and pose/recon_unposed protocol audited. AUC/F1 percentages below, distances unsquared metres, CD=(ACC+Comp)/2; JSON/CSV use fractions. Threshold/voxel:7Scenes0.05m/0.0078125m;ETH3D0.25m/0.0390625m. SAF3R is lazy first-correspondence-head Top4/ETH3D profile, not paper DINO Top-K.

| dataset | method | scenes | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense | 7 | True | 23.6689% | 84.6389% | 0.153741 | 0.125616 | 0.139678 | 48.0128% |
| 7scenes | saf3r | 7 | True | 23.9349% | 84.7131% | 0.143263 | 0.127883 | 0.135573 | 47.3154% |
| 7scenes | grid_local | 7 | True | 22.1234% | 84.4171% | 0.146727 | 0.124312 | 0.135520 | 45.2177% |
| 7scenes | v8 | 7 | True | 21.9102% | 84.2025% | 0.144386 | 0.121930 | 0.133158 | 46.8951% |
| 7scenes | beta_gridmass | 7 | True | 1.4553% | 40.4058% | 0.669175 | 0.554980 | 0.612078 | 10.9585% |
| eth3d | dense | 11 | True | 29.1835% | 81.8162% | 0.313193 | 0.848854 | 0.581024 | 62.5731% |
| eth3d | saf3r | 11 | True | 23.3322% | 80.1747% | 0.303106 | 0.857720 | 0.580413 | 62.8304% |
| eth3d | grid_local | 11 | True | 23.9534% | 79.2698% | 0.326859 | 0.875317 | 0.601088 | 56.8682% |
| eth3d | v8 | 11 | True | 21.9385% | 78.2346% | 0.308629 | 0.872234 | 0.590431 | 59.5813% |
| eth3d | beta_gridmass | 11 | True | 0.0000% | 1.9610% | 1.859754 | 7.197541 | 4.528647 | 1.9670% |

## Runtime

Fixed heads128 on RTX5090 D GPU7, one warmup+three complete-forward samples, then separate profile. Includes initialization within forward, clustering/routing/history/repair/reordering/heads. Excludes checkpoint/image loading and GT quality. GPU4–7 have CPU-only monitors, exactly one CUDA application per GPU until full process exit. Dense speedup N/A: no unchanged Dense rerun. Historical v8 heads12819.981s belongs to a different batch, not a paired speed comparison.

Changed-method warmed median **19.553523152s**, samples [19.560943017248064, 19.25325431721285, 19.55352315166965], maximum sample peak **18.829412GiB**.

Independent detailed event profile (parents are exclusive, not added twice); this profile wall is not the warmed benchmark median.

| operation | seconds | percent_full |
| --- | --- | --- |
| global_attention | 0.217881 | 1.162544 |
| qkv_rope_prepare | 0.496463 | 2.648967 |
| value_update | 0.028595 | 0.152574 |
| cluster_centers_from_labels | 0.011204 | 0.059780 |
| cluster_full_assignment | 0.169290 | 0.903277 |
| cluster_repair_empty | 0.016182 | 0.086341 |
| pooling | 0.086020 | 0.458976 |
| m5c_coarse_observation | 0.103100 | 0.550107 |
| execution_and_evidence | 0.037867 | 0.202045 |
| attention_wrapper | 0.837655 | 4.469452 |
| attention_core_fused_grid | 6.404190 | 34.170672 |
| attention_grid_normalize_reduce | 0.135894 | 0.725086 |
| m5c_evidence_and_host_copy | 0.002160 | 0.011526 |
| actual_sparse_probability_export | 0.445496 | 2.377023 |
| beta_counts_and_coarse_cache | 0.045187 | 0.241105 |
| history_summary | 0.040511 | 0.216151 |
| output_projection | 0.029711 | 0.158528 |
| output_dropout | 0.000018 | 0.000096 |
| cluster_center_graph | 0.454986 | 2.427656 |
| cluster_local_assignment | 0.489374 | 2.611141 |
| coarse_observation | 0.228956 | 1.221632 |
| residual_fine_probe | 0.354443 | 1.891194 |
| reuse_and_budget | 3.741345 | 19.962600 |
| consume_beta_history | 0.004182 | 0.022313 |
| beta_grid_calibration | 0.261341 | 1.394433 |
| beta_dense_prior_materialization | 0.262618 | 1.401242 |
| outside_global | 3.837104 | 20.473538 |

The former Grid-normalizer kernel is absent. This first implementation still exports large raw probe logits, and adds actual-P head reduction/cache, Grid-mass accumulation and current-layer compensation. Removing one kernel is not a guarantee of the same total saving. Differences in selected supports and other stages must also be considered.

## Validation and limits

Numerical export, formal AV, bias/Special/tail/padding, FP16-u14 storage and per-Grid compensation were checked on FP32/BF16. Numerical Grid-mass conservation passes, but this does not establish geometric quality. Model layer records preserve actual density, underfill, prior patch-mass error and cache rounding excess. Actual sparse P includes compensation bias and normalizes over selected keys, unlike old v8 conditionals. This semantic difference and across-layer distribution prediction are plausible error sources; this experiment does not uniquely isolate their contributions.

Actual recorded diagnostics: {"max_prior_patch_mass_error": 1.7881393432617188e-07, "max_quantized_sparse_cache_mass_excess": 0.000244140625, "global_attention_density": {"7scenes": 0.19708204987858022, "eth3d": 0.19840600798753175}}. seed_normalization_max_error in this new method records excess of the quantized actual-sparse patch mass over1; it is not a geometric score and is distinct from the current Dense prior mass-conservation error.

The first numerical diagnostic used an inappropriate unquantized FP32 absolute tolerance for a FP16 cache; v2 checks the quantized reference. Both diagnostics and the failed v1 trace are preserved. This did not change the model or geometric quality gates. No successful geometry run was repeated to select a score.

Source unchanged: True; audited scene runs: 18; missing: []. Source archive and per-run source hashes, numerical diagnostics, monitors, raw six-metric quality JSON/Markdown, layer tables and historical controls are published. Timing-only diagnostics have all six quality fields unavailable with concrete reasons.

The existing v8 quality/time controls are historical, and neither quality equivalence nor a paired Dense speedup is claimed. Only the changed method was rerun.
