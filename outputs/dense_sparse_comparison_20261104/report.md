# Dense versus v8/P0/beta: matched runtime and regression diagnosis

All methods: identical heads128 frame paths, checkpoint, preprocessing, BF16, TF32 disabled, RTX5090 D GPU7, one CUDA process per physical GPU with CPU-only locks/monitors. One cold/warmup + three complete-forward measurements, followed by a separate instrumented forward. GT quality and checkpoint/image loading excluded. This rerun of unchanged Dense/v8 is explicitly authorized by the user request to compare with the original model. GPU4–6 may run independent processes; physical GPU7 remains exclusive. No clock locking is claimed. Samples and ranges are preserved, including beta timing variation.

| method | warmed_median_s | sample_min_s | sample_max_s | dense_speedup | peak_allocated_gib | peak_reserved_gib |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 12.480868 | 12.458180 | 12.556859 | 1.000000 | 14.359551 | 17.566406 |
| v8 | 20.525772 | 20.381290 | 20.968127 | 0.608058 | 18.742395 | 26.187500 |
| p0 | 20.806060 | 20.735421 | 21.211980 | 0.599867 | 18.222847 | 25.414062 |
| beta | 19.798294 | 19.378833 | 22.849094 | 0.630401 | 18.866563 | 25.662109 |

## Shared operations and added sparse work

Seconds below are from the separate profile, not partitions of the warmed median. Each global layer is decomposed into named QKV/Norm/RoPE/output projection, core, and remaining global work. outside_global includes the unchanged backbone, frame Attention, ALL FFNs including global block FFNs, norms/residuals, heads, allocator/host gaps and any uninstrumented work. It is an accounting residual, not an extra sparse stage and not an identical fixed constant across methods.

| method | profile_full_s | common_qkv_norm_rope_proj_s | attention_core_s | other_global_s | outside_global_s |
| --- | --- | --- | --- | --- | --- |
| dense | 12.877114 | 0.519506 | 7.562792 | 0.008928 | 4.785889 |
| v8 | 20.854166 | 0.519980 | 6.462369 | 9.362625 | 4.509193 |
| p0 | 21.574514 | 0.519352 | 6.454262 | 9.684550 | 4.916350 |
| beta | 19.659594 | 0.527864 | 6.454044 | 8.068577 | 4.609109 |

| operation | dense_s | v8_s | p0_s | beta_s |
| --- | --- | --- | --- | --- |
| image_backbone | 0.604419 | 0.607932 | 0.609719 | 0.607904 |
| frame_attention | 0.758853 | 0.674788 | 0.684494 | 0.657302 |
| ffn | 0.559640 | 0.550449 | 0.568514 | 0.554266 |
| block_norm | 0.112884 | 0.069750 | 0.069709 | 0.069992 |
| global_qkv | 0.099898 | 0.100130 | 0.100129 | 0.100049 |
| global_qk_norm | 0.157757 | 0.158068 | 0.158007 | 0.157943 |
| global_output_projection | 0.031105 | 0.029864 | 0.029976 | 0.038793 |
| camera_head | 0.022186 | 0.008972 | 0.009113 | 0.008971 |
| depth_head | 0.999464 | 1.011358 | 1.010547 | 1.010144 |
| point_head | 1.008466 | 1.020332 | 1.021164 | 1.017497 |

The v8 added stages are Value/cluster updates, geometry and pooling, coarse/history observation, history cache production/consumption, eligibility gates, state losses, budget solve, Key packing, mass transfer, diagnostics and execution preparation. None is needed by original Dense SDPA. sparse_execution_wrapper exclusive time covers sorting/gathers/offset preparation and launch/allocator gaps, while sparse_grid_reduce is an added per-Grid statistics reduction. The sparse core also writes probes and Grid evidence, so it is not a pure QK/Softmax/AV-only kernel. Tiny reshapes/residual framework work exists on both paths and is not proven wholly sparse-only. Exclusive accounting avoids double counting.

Shared tensor operations were not found to be uniformly slower. Reported named times measure both paths directly; no historical Dense-minus-current-residual inference is used. Differences in allocation, layout, launch gaps and separate forward trajectories can affect common times. Shared RoPE is wrapped repeatedly because global modules share one object; only exclusive events are summed in accounting, including frame uses outside the global boundary. The additional hook dispatch cost means profile wall is diagnostic; uninstrumented medians are authoritative.

## Is the sparse core faster?

Complete-profile Dense global core is the native SDPA call, summed over24 layers. Sparse core is the actual fused AV+Grid/probe kernel, excluding wrapper sorting and Grid reduction. Compare attention_core_s above for the full-input native trajectories. Sparse core savings must pay all additional global work before an end-to-end gain is possible.

To separate trajectory/input differences, a second diagnostic replays the SAME Q/K/V from the first64-group window (2048 queries) of every beta layer, including Specials, original compensation and original probes. Native full-Key Dense SDPA and sparse executor are warmed and timed three times independently. These summed window medians are NOT whole-model timings; L0 is slower sparse.

| dense_core_s | sparse_core_s | sparse_executor_s | dense_over_sparse_core | dense_over_sparse_executor |
| --- | --- | --- | --- | --- |
| 0.158774 | 0.098297 | 0.114703 | 1.615242 | 1.384212 |

## Why beta quality collapses

Same-input shadow analysis uses the actual beta trajectory: first4 query groups/window0, original four probes and all heads. The shadow v8 estimator sees the SAME previous sparse support, raw unweighted logits, current Q/K, clustering and coarse observations. Alternate outputs do not feed the network. Priors are compared with exact current-QK Dense probabilities; L1/KL normalize patch mass only for comparing distributions. Other reported support masses are original full-normalization patch mass.

At L1: beta assigns previous selected Keys mass 0.7843304872512817, exact Dense 0.1964631825685501, and shadow v8 0.22327876091003418. Prior distribution L1 error: beta 1.1956727504730225, v8 0.08968348801136017.

The actual sparse probability already contains population/mass-transfer compensation: P_sparse(i)=exp(raw_i+bias_i)/Z_sparse. It therefore estimates represented mass on emitted Keys, not the unweighted individual-Key conditional cached by v8. Passing that value as an individual-Key prior reuses compensation in the next layer. Global normalization and per-Grid conditional normalization are not interchangeable, even if total current Grid mass is conserved.

The new hard cap can assign all current Grid mass to selected Keys, leaving exactly zero prior for unobserved members. At L1 the sampled beta prior zero fraction is 0.07914869487285614; shadow v8 has zero fraction 0.0. A zero-prior Key can still be emitted by Exact/History. Existing mass transfer computes weight=max(assigned_mass/max(own_prior,1e-30),1). An owner with zero own prior and positive assigned mass produces a huge compensation. The sampled L1 selected zero-prior fraction is 0.015635065734386444, maximum compensation log-bias 61.2362060546875 (weight about exp(61), not a normal cluster-size factor).

Those biased probabilities become highly concentrated BEFORE cache rounding: at L1 0.984364926815033 of valid per-probe cache entries are below half the smallest positive FP16 subnormal; the FP32 zero fraction is 0.0. FP16 storage then makes them exactly zero; shadow v8 Grid-conditional cache has zero fraction 0.0. This feeds the next layer and creates a feedback loop. Merely using FP32 cache would preserve tiny values but would not fix the wrong represented-mass semantics or huge log-bias.

Sparse AV is independently compared with FP32 exact softmax on the SAME emitted support and compensation. Maximum sampled mean relative L2 is 0.006264358875341713. This supports estimator/compensation failure, not an observed gross mask-execution bug; it is a sampled check, not proof that every kernel case is correct. The chain above is directly observed, but relative causal contributions have not been isolated by new full-dataset interventions. No probability floor, retention, confidence or quality gate was tuned against test truth.

## Existing complete geometry results

These are reused complete historical matched-input controls and the already-frozen beta campaign, NOT quality measured by the timing diagnostics. All7 7Scenes scenes128frames and all11 ETH3D scenes14–76frames; equal-scene macro; AUC/F1 percentages, distances unsquared metres, CD=(ACC+Comp)/2. Pose+recon_unposed. Threshold/voxel7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. Full per-scene raw metrics are in the beta campaign remote commit7abc8442.

| dataset | method | scenes | auc03_percent | auc30_percent | acc_m | comp_m | cd_m | f1_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense | 7 | 23.668916 | 84.638873 | 0.153741 | 0.125616 | 0.139678 | 48.012766 |
| 7scenes | v8 | 7 | 21.910152 | 84.202522 | 0.144386 | 0.121930 | 0.133158 | 46.895067 |
| 7scenes | beta_gridmass | 7 | 1.455287 | 40.405828 | 0.669175 | 0.554980 | 0.612078 | 10.958466 |
| eth3d | dense | 11 | 29.183465 | 81.816212 | 0.313193 | 0.848854 | 0.581024 | 62.573136 |
| eth3d | v8 | 11 | 21.938461 | 78.234578 | 0.308629 | 0.872234 | 0.590431 | 59.581300 |
| eth3d | beta_gridmass | 11 | 0.000000 | 1.961011 | 1.859754 | 7.197541 | 4.528647 | 1.967020 |

P0 is a separate execution prototype of original v8, not beta. Initial P0 kernels preserve sampled Key order/owners but are slower than their reference kernels; they are not promoted to the default path. All23 changed layers receive a captured-vs-eager routing check, and complete P0 geometry evaluation is recorded separately. This first P0 prototype reuses geometry across neighboring Query Groups; it does NOT claim to have implemented the document's Cluster–Grid sufficient-statistics decomposition.

Every timing/kernel/estimator diagnostic retains all six quality fields as unavailable with a concrete reason. End-to-end Dense speedup belongs only to the matched full-forward medians above; operator ratios and historical geometry rows must not be used as substitutes.

## Subsequent P0 validation

The corrected P0 passes the first window of23 layers but FAILS the later all-window equivalence check at L1. Neither P0 prototype is accepted as a globally equivalent optimization. Full P0 evaluation is reported separately; the initial comparison numbers above belong to the initial prototype.
