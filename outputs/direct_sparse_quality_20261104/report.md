# Grid-free Direct Sparse Attention, original v8 history

Campaign complete: **True**. Opt-in backend; default unchanged. Configuration and source frozen before quality; no test-truth tuning.

## Matched geometry quality

Only Direct reran. Dense/SAF3R/v8 controls are complete historical local measurements,not new timing/quality reruns. Same checkpoint,exact input frames,shape,preprocessing,pose+recon_unposed,alignment/fusion and reconstruction protocol audited. All7 7Scenes scenes128frames; all11 ETH3D scenes14–76frames; equal-scene macros. AUC/F1 percentages in this table,JSON fractions. Distances metres,unsquared CD=(ACC+Comp)/2. Threshold/voxel7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. SAF3R is local lazy first-correspondence-head Top4/ETH3D profile,not paper DINO Top-K.

| dataset | method | scenes | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense | 7 | True | 23.6689% | 84.6389% | 0.153741 | 0.125616 | 0.139678 | 48.0128% |
| 7scenes | saf3r | 7 | True | 23.9349% | 84.7131% | 0.143263 | 0.127883 | 0.135573 | 47.3154% |
| 7scenes | v8 | 7 | True | 21.9102% | 84.2025% | 0.144386 | 0.121930 | 0.133158 | 46.8951% |
| 7scenes | direct | 7 | True | 22.6091% | 84.4353% | 0.145531 | 0.120729 | 0.133130 | 46.4245% |
| eth3d | dense | 11 | True | 29.1835% | 81.8162% | 0.313193 | 0.848854 | 0.581024 | 62.5731% |
| eth3d | saf3r | 11 | True | 23.3322% | 80.1747% | 0.303106 | 0.857720 | 0.580413 | 62.8304% |
| eth3d | v8 | 11 | True | 21.9385% | 78.2346% | 0.308629 | 0.872234 | 0.590431 | 59.5813% |
| eth3d | direct | 11 | True | 22.4769% | 77.9733% | 0.320937 | 0.859388 | 0.590162 | 60.9808% |

Relative changes against historical matched v8: [{"dataset": "7scenes", "reference": "historical matched v8", "auc03_relative_change_percent": 3.1900101609711795, "auc30_relative_change_percent": 0.27643337628615594, "acc_relative_change_percent": 0.7925549468652093, "comp_relative_change_percent": -0.9847975856681623, "cd_relative_change_percent": -0.021184340036162208, "f1_relative_change_percent": -1.0034749421520495, "auc03_percentage_point_change": 0.6989360704911907, "auc30_percentage_point_change": 0.23276387326585102, "f1_percentage_point_change": -0.47058024296087186}, {"dataset": "eth3d", "reference": "historical matched v8", "auc03_relative_change_percent": 2.454112436164513, "auc30_relative_change_percent": -0.33401194961433234, "acc_relative_change_percent": 3.9881097004426103, "comp_relative_change_percent": -1.4727702840063528, "cd_relative_change_percent": -0.04552198441786226, "f1_relative_change_percent": 2.348936513880573, "auc03_percentage_point_change": 0.5383945007014307, "auc30_percentage_point_change": -0.26131283794230686, "f1_percentage_point_change": 1.399526909660842}]. These are measured changes,not shifted/scaled scores.

## Complete-forward timing

heads128,RTX5090 D GPU7,onewarmup+three uninstrumented complete forwards and a SEPARATE profiled forward. Includes all clustering,routing,history,reordering,observations,cache,initialization inside forward and heads; excludes checkpoint/image loading and GT evaluation. CPU-only locks,one CUDA application per physical GPU. Full process exit releases the GPU,including CPU quality tails.

Warmed median **18.999162088s**,samples [18.61317575117573, 18.999162088148296, 19.34976469632238],max allocated **18.739395GiB**,max reserved **25.595703GiB**.

End-to-end Dense speedup **N/A**: unchanged Dense full-forward was not rerun in this modified-method campaign. Historical Dense12.642997s belongs to dense_p0_v2_comparison_20261104,not a paired denominator. Historical v820.525772s is also a separate batch; no contemporaneous percentage gain against it is claimed. Dense identity is1.00x.

Separate profile: 19.001787392888218s; percentages use THAT profile wall,not the warmed median.

| operation | seconds | percent_profile | scope |
| --- | --- | --- | --- |
| value_update | 1.141653 | 6.008134 | disjoint top-level |
| pooling | 0.082351 | 0.433386 | disjoint top-level |
| m5c_coarse_observation | 0.143309 | 0.754188 | disjoint top-level |
| execution_and_evidence | 2.764419 | 14.548207 | disjoint top-level |
| direct_independent_legacy_observation | 3.610974 | 19.003341 | disjoint top-level |
| m5c_evidence_and_host_copy | 1.359759 | 7.155954 | disjoint top-level |
| history_summary | 0.043676 | 0.229852 | disjoint top-level |
| coarse_observation | 0.027640 | 0.145457 | disjoint top-level |
| residual_fine_probe | 0.351464 | 1.849635 | disjoint top-level |
| reuse_and_budget | 4.006901 | 21.086966 | disjoint top-level |
| outside_named_stages | 5.469642 | 28.784879 | base model,projection,host/allocator gaps and uninstrumented work |
| legacy_full_query_grid_and_probe_observation | 3.564986 | 18.761320 | nested within observation/cache stages;DO NOT add twice |
| legacy_grid_conditional_cache | 1.358384 | 7.148719 | nested within observation/cache stages;DO NOT add twice |

## Exact-input all-window operator comparison

Original v8 heads128 trajectory. ALL1584 actual windows across24 global layers,all16heads,original shared supports and compensation. Every operator/window has three CUDA-event measurements; table sums per-window medians. Native Dense FLASH_ATTENTION processes the SAME query windows and full K/V; it is a windowed operator baseline,not native full-model Dense timing. Pure Direct intentionally omits compensation and is a different operator mode; compensated Direct and old Grid use identical bias. Alternate results never feed the trajectory.

| method | seconds | dense_operator_speedup | old_grid_operator_speedup |
| --- | --- | --- | --- |
| dense_flash | 10.222401 | 1.000000 | 0.729438 |
| grid_compensated | 7.456610 | 1.370918 | 1.000000 |
| direct_pure | 2.806066 | 3.642965 | 2.657318 |
| direct_compensated | 2.864181 | 3.569048 | 2.603400 |
| direct_history_A | 7.863682 | 1.299951 | 0.948234 |
| direct_history_B | 3.752775 | 2.723958 | 1.986959 |

A stores per-tile normalized FP16 probabilities plus tile LSE during main AV,and only finalizes with the FINAL full-support LSE before Head mean. It is not a provisional-denominator final probability. It avoids full FP32 raw logits but still needs a large per-Head FP16 temporary (~221732864bytes/full window). B writes final LSE and uses a small-probe QK kernel,normalizes each Head before reducing;no per-Head edge bank. B is preferable in measured complete operator time. Single/four-probe,padding,Special,bias and tail/capture tests are separate.

Fixed complete-window kernel128Keys/4warps/2stages. First windows of layers0/8/16/23 additionally tested64/4/1,128/4/2,128/8/2,256/4/2,256/8/2;128/4/2 had lowest aggregate sampled time. Tuning uses only fixed-QKV numerical/performance evidence,not GT. per-window,24-layer and tuning CSVs are published.

Launch counts on the real L0 first window;CUDA memory events are separated from compute kernels. These counts are not extrapolated to shape-dependent old Grid sorts in every window.

| method | kernel_launches | memory_events | cuda_events |
| --- | --- | --- | --- |
| dense_flash | 1 | 0 | 1 |
| grid_sparse | 30 | 15 | 45 |
| direct_pure | 1 | 0 | 1 |
| direct_compensated | 1 | 0 | 1 |
| direct_A | 2 | 0 | 2 |
| direct_B | 2 | 0 | 2 |

| kernel | registers_per_thread | spills | shared_bytes | warps | stages |
| --- | --- | --- | --- | --- | --- |
| direct_pure:_attention | 130 | 0 | 45056 | 4 | 2 |
| direct_compensated:_attention | 128 | 0 | 45568 | 4 | 2 |
| direct_A:_attention | 168 | 0 | 53760 | 4 | 2 |
| direct_A:_inline_mean | 56 | 0 | 0 | 4 | N/A |
| direct_B:_attention | 128 | 0 | 45568 | 4 | 2 |
| direct_B:_probe | 118 | 0 | 18432 | 4 | 1 |

Native hardware occupancy is N/A: actual Nsight Compute attempt returned ERR_NVGPUCTRPERM. Compiler metadata is not measured occupancy.

## Scope of the refactor

Direct AV consumes only Q/K/V,compact IDs,Specials,scale,bias and optional probe positions. No Grid argument,sort,count,offset,Grid loop or Grid output. Online softmax performs its normal AV QK once. No full Query×Key mask is materialized. Kernel capture/replay with mutated inputs passes. Triton Flash-style Direct is NOT PyTorch native FLASH_ATTENTION;the separate forced-PyTorch adapter is retained at c965ff76 and config reuse_v8_torch_flash.json. That adapter was compatible but slow43.434s because of K/V packing,and was superseded by this requested path before a quality campaign.

The user explicitly chose original v8 D2+M5c history. Its full-query Grid summaries and per-Grid conditional normalizers still exist ONLY in an independent module. That module repeats auxiliary QK and its FULL cost is included above. Therefore the full model has NOT eliminated all Grid work or all auxiliary QK. Actual Sparse Probability A/B exports are separate API outputs,not substituted for old Grid-conditionals;no reuse of the failed beta actual-P estimator.

Coarse Key pooling remains once/layer. The independent coarse observer computes one FP32 Probe×pooled-Key QK and separately normalizes full-Key(with Specials) for M5c and patch-only for residual demand. Original residual coarse arithmetic used rounded BF16 pooled Keys;sharing FP32 logits is a precision change,not bitwise identical upstream execution. Together with AV/reduction rounding and discrete budget ties,this can change final supports/quality even though formulas,retention,states,losses,budget16rounds and compensation are unchanged. Do not claim complete bitwise trajectory equivalence from same-input kernel checks.

Legacy sums use a conservative Cauchy Q/K norm bound. For bound<=32,the common shift keeps exponents in[-64,0],allowing stable one-pass selected sums;otherwise a stable original per-Grid max/probe fallback is used. This changes neither a probability floor nor the mathematical prior. Four-probe cache production reuses K loads and Tensor Cores. Original per-Head normalization,EPS compatibility,Head-mean FP16-u14 cache and Special definitions remain. Ordinary/high-norm tests and all24 first-window original AV/Grid/cache plus real FP32 query references pass.

Source unchanged: True;scene audits: 18;missing: [];diagnostics: {'global_attention_density': {'7scenes': 0.1998704406654551, 'eth3d': 0.19992786414469732}, 'max_prior_mass_error': 1.5556812286376953e-05, 'raw_probe_scratch_max_bytes': 0}. Frozen full source archive,per-run source hashes,raw metrics,monitors,failed preflights,all-window benchmarks and per-scene six-metric Markdown are retained. All timing/kernel/failed diagnostics have six fields unavailable with concrete reasons.

## Matched coarse precision sensitivity

{"sampling": "heads128 Direct trajectory,first64-group window/L1\u201323;same Q/K/pooled means and previous Grid statistics;alternative coarse outputs not fed into model", "mean_candidate_replaced_fraction": 0.0002962746859892555, "max_candidate_replaced_fraction": 0.0030754804611206055, "mean_patch_probability_l1": 0.0007091038778130694, "max_coarse_logit_abs_difference": 0.11378860473632812, "interpretation": "Shared FP32 versus old BF16 coarse precision already changes a small fraction of Exact eligibility candidates on identical Q/K;AV/order/reduction rounding can further propagate. This does not isolate the cause of the final geometry differences."}

Quality is mixed and the default remains unchanged. In particular ETH3D ACC worsens3.9881% versus historical v8 even though F1 improves;7Scenes F1 declines1.0035% relative. Do not describe this as proven no-degradation or complete trajectory equivalence.
