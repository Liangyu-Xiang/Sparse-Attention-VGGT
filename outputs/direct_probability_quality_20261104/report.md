# Direct Sparse probability history

Campaign complete: **True**. Opt-in only; default unchanged. No test-truth tuning or unchanged-control reruns.

## Method

Removed legacy all-query selected-QK/Grid observations and the old four-probe conditional-cache QK. Main AV is unchanged Direct compensated Sparse. One independent four-probe export uses final sparse LSE; it computes log(mean_h(exp(raw_h - LSE_comp,h))), algebraically the shared-bias-removed actual sparse probability. A zero assignment weight is handled by the raw/LSE limiting score, rather than dividing two floating-point zeros. No per-head edge bank or FP32 raw-logit bank. Four-probe QK still exists; this is not a zero-additional-QK claim.

Missing-token score is logmeanexp_h(pooled_grid_logit_h - LSE_comp,h), requiring no new Grid QK beyond the existing coarse observer. Selected and coarse missing scores jointly form positive per-Grid conditional distributions via compact log-sum-exp reductions. Signed FP16 log conditionals are cached to avoid FP16 probability underflow. Next layer applies the unchanged M5c beta(1,2) confidence, blending conditionals with uniform, and uses current coarse Grid mass to complete unselected tokens. No hard cap that assigns the entire Grid mass to selected tokens. This intentionally changes the estimator: ratio of head-mean scores, instead of the mean of per-head Grid conditionals; residual exploration uses four-probe means instead of all32 Queries. It is not mathematically equivalent to v8 nor exact Dense probability recovery.

Retention20%,adaptive-local clusters,four probes,Drop/Rep/History/Exact,16-round budget,state losses,compensation and all geometry protocols remain frozen.

## Six-metric quality

All7 7Scenes scenes128frames; all11 ETH3D scenes14–76frames; equal-scene macros. AUC/F1 are percentages here and fractions in raw JSON. ACC/Comp/CD in metres; unsquared CD=(ACC+Comp)/2. Threshold/voxel:7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. Same checkpoint,exact image list,preprocessing,pose and reconstruction protocols audited against preceding Direct run. Dense/SAF3R/v8/Direct-legacy are historical local controls. SAF3R is lazy first-correspondence-head Top4 with ETH3D profile,not paper DINO Top-K.

| dataset | method | scenes | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense | 7 | True | 23.6689% | 84.6389% | 0.153741 | 0.125616 | 0.139678 | 48.0128% |
| 7scenes | saf3r | 7 | True | 23.9349% | 84.7131% | 0.143263 | 0.127883 | 0.135573 | 47.3154% |
| 7scenes | v8 | 7 | True | 21.9102% | 84.2025% | 0.144386 | 0.121930 | 0.133158 | 46.8951% |
| 7scenes | direct_legacy | 7 | True | 22.6091% | 84.4353% | 0.145531 | 0.120729 | 0.133130 | 46.4245% |
| 7scenes | probability_history | 7 | True | 22.5517% | 84.4984% | 0.142468 | 0.124380 | 0.133424 | 44.9711% |
| eth3d | dense | 11 | True | 29.1835% | 81.8162% | 0.313193 | 0.848854 | 0.581024 | 62.5731% |
| eth3d | saf3r | 11 | True | 23.3322% | 80.1747% | 0.303106 | 0.857720 | 0.580413 | 62.8304% |
| eth3d | v8 | 11 | True | 21.9385% | 78.2346% | 0.308629 | 0.872234 | 0.590431 | 59.5813% |
| eth3d | direct_legacy | 11 | True | 22.4769% | 77.9733% | 0.320937 | 0.859388 | 0.590162 | 60.9808% |
| eth3d | probability_history | 11 | True | 22.5800% | 79.3415% | 0.342087 | 0.882130 | 0.612108 | 54.7835% |

## Complete-forward timing

heads128,GPU7 RTX5090 D,onewarmup+three warmed full forwards plus SEPARATE profiled forward. Includes initialization inside forward,clustering,routing,history,reordering and prediction heads. CPU-only leases,one CUDA application per physical GPU,held through full process exit.

Warmed median **15.641640002s**,samples [15.225101436022669, 15.641640001907945, 15.71937146410346],max allocated **18.794601GiB**.

End-to-end Dense speedup **N/A**: no unchanged Dense rerun in this campaign; historical Dense timings cannot be used as paired speed comparisons. Dense identity1.00x. Preceding Direct18.999162s is historical from direct_sparse_quality_20261104,not a contemporaneous speed comparison.

Profile percentages use the separate profile wall. execution_and_evidence includes both AV and four-probe export; nested Direct scope timings separate those operations and must not be added twice. outside_named_stages includes shared base-model work PLUS uninstrumented host/allocator gaps; it is not all unavoidable Dense cost.

| operation | seconds | percent_profile |
| --- | --- | --- |
| value_update | 1.075847 | 6.519897 |
| pooling | 0.080976 | 0.490736 |
| m5c_coarse_observation | 0.133795 | 0.810829 |
| execution_and_evidence | 3.738445 | 22.655896 |
| sparse_probe_grid_summary | 0.417437 | 2.529769 |
| m5c_evidence_and_host_copy | 0.398454 | 2.414731 |
| history_summary | 0.036679 | 0.222281 |
| coarse_observation | 0.030088 | 0.182339 |
| residual_fine_probe | 0.351190 | 2.128297 |
| reuse_and_budget | 4.051142 | 24.550919 |
| outside_named_stages | 6.186928 | 37.494304 |

| operation | seconds | scope |
| --- | --- | --- |
| direct_av | 2.842266 | nested within execution_and_evidence; do not add again |
| direct_probability_export | 0.887521 | nested within execution_and_evidence; do not add again |

Audits=18,source unchanged=True,missing=[],diagnostics={'max_prior_mass_error': 3.170967102050781e-05, 'max_seed_error': 0.0}.

## Per-scene results

| dataset | scene | frames | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | heads | 128 | True | 52.7518% | 94.1265% | 0.036521 | 0.034290 | 0.035405 | 83.2679% |
| 7scenes | chess | 128 | True | 15.9695% | 87.2174% | 0.142258 | 0.090951 | 0.116604 | 34.4755% |
| 7scenes | stairs | 128 | True | 6.7298% | 80.8850% | 0.219630 | 0.464367 | 0.341998 | 15.3839% |
| 7scenes | fire | 128 | True | 53.0758% | 94.2762% | 0.056302 | 0.054350 | 0.055326 | 53.2358% |
| 7scenes | office | 128 | True | 11.9177% | 78.0393% | 0.051605 | 0.053964 | 0.052785 | 56.1918% |
| 7scenes | pumpkin | 128 | True | 10.1255% | 77.5668% | 0.413556 | 0.097447 | 0.255501 | 23.0933% |
| 7scenes | redkitchen | 128 | True | 7.2917% | 79.3775% | 0.077404 | 0.075290 | 0.076347 | 49.1498% |
| eth3d | courtyard | 38 | True | 32.3850% | 87.6197% | 0.288079 | 0.206005 | 0.247042 | 51.7030% |
| eth3d | delivery_area | 40 | True | 25.4274% | 85.6709% | 0.610418 | 1.371638 | 0.991028 | 18.8133% |
| eth3d | electro | 39 | True | 29.1048% | 91.6599% | 0.144571 | 5.011316 | 2.577943 | 66.8232% |
| eth3d | facade | 76 | True | 28.0000% | 89.1041% | 0.300552 | 0.272819 | 0.286686 | 57.9461% |
| eth3d | kicker | 31 | True | 41.2903% | 92.9606% | 0.056863 | 0.051545 | 0.054204 | 99.5157% |
| eth3d | office | 26 | True | 21.3333% | 78.2359% | 0.087149 | 0.086810 | 0.086979 | 99.2898% |
| eth3d | pipes | 14 | True | 25.6410% | 88.2051% | 0.233261 | 0.263359 | 0.248310 | 75.5974% |
| eth3d | playground | 32 | True | 12.6344% | 70.1344% | 0.339138 | 0.802944 | 0.571041 | 59.9624% |
| eth3d | relief | 19 | True | 19.1033% | 79.6101% | 0.691247 | 0.681246 | 0.686247 | 13.4724% |
| eth3d | relief_2 | 20 | True | 6.4912% | 52.3158% | 0.584510 | 0.690562 | 0.637536 | 9.4723% |
| eth3d | terrains | 42 | True | 6.9686% | 57.2396% | 0.427162 | 0.265189 | 0.346176 | 50.0225% |

Raw metrics,per-layer records,source snapshot/hashes,monitor logs and diagnostic failures are retained. Numerical tests cover extreme bias,zero assignments,FP16 underflow,arbitrary support order/holes,all/none/partial Grids,tail probes and fresh CUDA Graph replay; six geometric fields N/A for these diagnostics. Numerical correctness does not establish geometric quality.

## Interpretation and measurement limits

Full-forward warmed median15.641640s; the separate profile wall16.500981s is a
different forward and must not be summed into the warmed median. Its disjoint
three-way grouping is Direct AV2.842266s(17.22%),outside_named_stages6.186928s
(37.49%,shared model work PLUS uninstrumented gaps),algorithm7.471787s(45.28%).
The shared/residual category is not a measurement of unavoidable Dense cost.
Four-probe export0.887521s is part of algorithm overhead,not the AV core.

Against the historical matched preceding Direct campaign,7Scenes F1 decreases
1.453357percentage points;ETH3D decreases6.197351points. ETH3D ACC worsens6.589944%
and CD3.718644%,while pose AUC@30 increases1.368213points. This is not a quality
equivalent replacement. The new estimator remains an explicit experiment.

The inherited per-layer field `seed_normalization_max_error` is used by this
backend as a finite-denominator guard(zero when finite),not a measured floating
point normalization error. Actual FP16-log-cache normalization is checked by
`prior_patch_mass_max_error` and the numerical diagnostic. The largest complete
campaign prior patch-mass error is recorded in summary.json. Do not interpret
the finite guard's zero as exact probability normalization.

Reproduce report generation from repository root with
`python -m scripts.report_probability_history` (or set PYTHONPATH=.).
Campaign source hashes and archive are independently verified in verification.json;
all18 scene audits and19 exclusive GPU monitors pass. No code changes were made
after the quality campaign's source freeze.
