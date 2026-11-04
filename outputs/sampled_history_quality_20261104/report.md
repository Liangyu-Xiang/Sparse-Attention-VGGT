# One sparse history Probe every32/256 Queries

Campaign complete: **True**;only the two changed methods rerun,no GT tuning or parameter/quality-gate changes.

## Fixed estimator and unsampled-group rule

Both methods export inline debiased log sparse scores with final compensated LSE,one Probe at offset4.32 samples every Query Group;256 samples every8Groups. Only sampled Groups receive probability history. Other Groups use their OWN current coarse-uniform prior;their residual summary uses own coarse patch mass and selected-count/GridSize. No neighbor Key IDs or probabilities are copied. Their own previous support remains eligible for the unchanged state selector.

Sampled Groups:complete missing Keys using pooled-Key logits in the same sparse-LSE scale;store signed FP16 log conditionals and coarse/unobserved statistics;apply M5c beta(1,2) in the next layer. The prior uses coarse Probe0 consistently;four coarse Probes remain for residual candidate exploration. Cache is sampled-only and one-Probe. Retention20%,adaptive-local clusters,Drop/Rep/History/Exact,state loss,16-round budget and mass-transfer compensation stay fixed. The inline log exporter has FP16 auxiliary quantization;this is not an exact old-four-Probe trajectory claim.

## Complete-forward runtime

Both intervals use GPU7 RTX5090 D,the identical heads128 image list/checkpoint/preprocessing,one warmup+three full-forward samples,and a separate profiled forward. Includes initialization inside forward,clustering,routing,history,reordering and prediction heads. CPU-only exclusive leases held through full process exit;one CUDA program per physical GPU.

| interval | median_seconds | peak_allocated_gib | dense_speedup | samples |
| --- | --- | --- | --- | --- |
| 32 | 15.222148 | 18.253244 | N/A | [14.870664987247437, 15.222148088738322, 15.231552585028112] |
| 256 | 14.366136 | 17.992786 | N/A | [14.124551275745034, 14.5040014940314, 14.366136084776372] |

Dense end-to-end speedup N/A:no unchanged Dense rerun. Historical probability4=15.641640s from direct_probability_quality_20261104 is not a paired baseline. Compare32vs256 within this batch;do not label the historical difference a paired speedup.

Disjoint profile stages,percentages use each separate profile wall rather than the warmed median:

| interval | operation | seconds | percent_profile |
| --- | --- | --- | --- |
| 32 | value_update | 1.087766 | 7.247852 |
| 32 | pooling | 0.081852 | 0.545385 |
| 32 | m5c_coarse_observation | 0.132623 | 0.883674 |
| 32 | execution_and_evidence | 3.606277 | 24.028847 |
| 32 | sampled_probe_grid_summary | 0.203928 | 1.358783 |
| 32 | m5c_evidence_and_host_copy | 0.249791 | 1.664371 |
| 32 | history_summary | 0.035814 | 0.238628 |
| 32 | coarse_observation | 0.029666 | 0.197664 |
| 32 | residual_fine_probe | 0.351565 | 2.342499 |
| 32 | reuse_and_budget | 3.900263 | 25.987695 |
| 32 | outside_named_stages | 5.328572 | 35.504604 |
| 256 | value_update | 1.080874 | 7.590262 |
| 256 | pooling | 0.080631 | 0.566220 |
| 256 | m5c_coarse_observation | 0.233461 | 1.639441 |
| 256 | execution_and_evidence | 2.911825 | 20.447823 |
| 256 | sampled_probe_grid_summary | 0.138180 | 0.970348 |
| 256 | m5c_evidence_and_host_copy | 0.144274 | 1.013141 |
| 256 | history_summary | 0.036449 | 0.255958 |
| 256 | coarse_observation | 0.030588 | 0.214798 |
| 256 | residual_fine_probe | 0.353630 | 2.483308 |
| 256 | reuse_and_budget | 3.928120 | 27.584594 |
| 256 | outside_named_stages | 5.302237 | 37.234108 |

Nested AV and probability-finalizer times are already inside execution_and_evidence:

| interval | operation | seconds | scope |
| --- | --- | --- | --- |
| 32 | sampled_direct_av | 3.546845 | nested in execution_and_evidence;do not add twice |
| 32 | sampled_probability_export | 0.050497 | nested in execution_and_evidence;do not add twice |
| 256 | sampled_direct_av | 2.892085 | nested in execution_and_evidence;do not add twice |
| 256 | sampled_probability_export | 0.011656 | nested in execution_and_evidence;do not add twice |

outside_named_stages includes shared original-model work plus uninstrumented host/allocator gaps;it is not all unavoidable Dense cost.

## Six-metric quality

All7 7Scenes scenes128frames and all11 ETH3D scenes14–76frames,pose+recon_unposed,equal-scene macros. Same exact frames/checkpoint/shape/protocol audited against historical probability4. AUC/F1 percentages here,fractions in raw JSON. Distances metres,unsquared CD=(ACC+Comp)/2. Threshold/voxel7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. Dense/SAF3R/v8/Direct-legacy/probability4 controls are historical local results. SAF3R uses local lazy first-correspondence-head Top4/ETH3D profile,not paper DINO Top-K.

| dataset | method | scenes | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense | 7 | True | 23.6689% | 84.6389% | 0.153741 | 0.125616 | 0.139678 | 48.0128% |
| 7scenes | saf3r | 7 | True | 23.9349% | 84.7131% | 0.143263 | 0.127883 | 0.135573 | 47.3154% |
| 7scenes | v8 | 7 | True | 21.9102% | 84.2025% | 0.144386 | 0.121930 | 0.133158 | 46.8951% |
| 7scenes | direct_legacy | 7 | True | 22.6091% | 84.4353% | 0.145531 | 0.120729 | 0.133130 | 46.4245% |
| 7scenes | probability4 | 7 | True | 22.5517% | 84.4984% | 0.142468 | 0.124380 | 0.133424 | 44.9711% |
| 7scenes | sampled32 | 7 | True | 22.7040% | 84.4103% | 0.144742 | 0.123145 | 0.133944 | 46.5072% |
| 7scenes | sampled256 | 7 | True | 22.3806% | 84.4531% | 0.150148 | 0.123331 | 0.136739 | 47.1048% |
| eth3d | dense | 11 | True | 29.1835% | 81.8162% | 0.313193 | 0.848854 | 0.581024 | 62.5731% |
| eth3d | saf3r | 11 | True | 23.3322% | 80.1747% | 0.303106 | 0.857720 | 0.580413 | 62.8304% |
| eth3d | v8 | 11 | True | 21.9385% | 78.2346% | 0.308629 | 0.872234 | 0.590431 | 59.5813% |
| eth3d | direct_legacy | 11 | True | 22.4769% | 77.9733% | 0.320937 | 0.859388 | 0.590162 | 60.9808% |
| eth3d | probability4 | 11 | True | 22.5800% | 79.3415% | 0.342087 | 0.882130 | 0.612108 | 54.7835% |
| eth3d | sampled32 | 11 | True | 21.6994% | 78.2119% | 0.334705 | 0.881240 | 0.607972 | 54.6464% |
| eth3d | sampled256 | 11 | True | 21.7618% | 77.7107% | 0.359736 | 0.930192 | 0.644964 | 52.5911% |

## Per-scene changed methods

| dataset | scene | method | frames | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | heads | sampled32 | 128 | True | 52.4934% | 94.0203% | 0.036200 | 0.032084 | 0.034142 | 83.6571% |
| 7scenes | heads | sampled256 | 128 | True | 49.8606% | 93.8960% | 0.035363 | 0.032194 | 0.033778 | 84.3256% |
| 7scenes | chess | sampled32 | 128 | True | 14.7310% | 86.8086% | 0.141736 | 0.089722 | 0.115729 | 36.1772% |
| 7scenes | chess | sampled256 | 128 | True | 16.2073% | 87.3565% | 0.139249 | 0.087361 | 0.113305 | 37.0764% |
| 7scenes | stairs | sampled32 | 128 | True | 7.9601% | 81.9714% | 0.204990 | 0.466641 | 0.335816 | 16.5314% |
| 7scenes | stairs | sampled256 | 128 | True | 7.4885% | 81.2959% | 0.220548 | 0.474248 | 0.347398 | 15.2717% |
| 7scenes | fire | sampled32 | 128 | True | 54.2118% | 94.4156% | 0.052484 | 0.049650 | 0.051067 | 65.0617% |
| 7scenes | fire | sampled256 | 128 | True | 52.9610% | 94.2064% | 0.051094 | 0.052169 | 0.051631 | 61.7848% |
| 7scenes | office | sampled32 | 128 | True | 12.0366% | 77.3150% | 0.055885 | 0.057152 | 0.056518 | 48.8738% |
| 7scenes | office | sampled256 | 128 | True | 11.6962% | 77.5623% | 0.053796 | 0.056088 | 0.054942 | 54.1992% |
| 7scenes | pumpkin | sampled32 | 128 | True | 9.8343% | 77.1953% | 0.442734 | 0.089508 | 0.266121 | 26.6731% |
| 7scenes | pumpkin | sampled256 | 128 | True | 10.2854% | 77.3618% | 0.472512 | 0.086288 | 0.279400 | 28.3319% |
| 7scenes | redkitchen | sampled32 | 128 | True | 7.6608% | 79.1462% | 0.079166 | 0.077260 | 0.078213 | 48.5765% |
| 7scenes | redkitchen | sampled256 | 128 | True | 8.1652% | 79.4927% | 0.078476 | 0.074965 | 0.076721 | 48.7440% |
| eth3d | courtyard | sampled32 | 38 | True | 24.3243% | 85.9412% | 0.284511 | 0.168725 | 0.226618 | 59.6977% |
| eth3d | courtyard | sampled256 | 38 | True | 21.7639% | 84.1157% | 0.307188 | 0.212152 | 0.259670 | 53.4601% |
| eth3d | delivery_area | sampled32 | 40 | True | 19.7436% | 80.4957% | 0.586009 | 1.424384 | 1.005197 | 20.4942% |
| eth3d | delivery_area | sampled256 | 40 | True | 19.3590% | 83.9615% | 0.585061 | 1.418365 | 1.001713 | 20.1530% |
| eth3d | electro | sampled32 | 39 | True | 30.0045% | 91.6374% | 0.182112 | 5.020795 | 2.601454 | 64.0953% |
| eth3d | electro | sampled256 | 39 | True | 30.8592% | 91.7364% | 0.273644 | 5.180473 | 2.727059 | 56.6193% |
| eth3d | facade | sampled32 | 76 | True | 28.4444% | 89.2246% | 0.328699 | 0.265275 | 0.296987 | 56.8043% |
| eth3d | facade | sampled256 | 76 | True | 34.5965% | 89.5544% | 0.348019 | 0.251237 | 0.299628 | 55.0852% |
| eth3d | kicker | sampled32 | 31 | True | 50.2509% | 94.0430% | 0.058842 | 0.054125 | 0.056483 | 99.3788% |
| eth3d | kicker | sampled256 | 31 | True | 51.4695% | 94.6380% | 0.057305 | 0.054015 | 0.055660 | 99.4159% |
| eth3d | office | sampled32 | 26 | True | 19.4872% | 76.4000% | 0.097288 | 0.095928 | 0.096608 | 98.9954% |
| eth3d | office | sampled256 | 26 | True | 20.6154% | 78.0718% | 0.081313 | 0.075131 | 0.078222 | 99.2781% |
| eth3d | pipes | sampled32 | 14 | True | 24.9084% | 87.4725% | 0.240402 | 0.219258 | 0.229830 | 74.1484% |
| eth3d | pipes | sampled256 | 14 | True | 26.3736% | 88.9744% | 0.200484 | 0.321418 | 0.260951 | 78.0534% |
| eth3d | playground | sampled32 | 32 | True | 10.2151% | 67.4261% | 0.354635 | 0.901867 | 0.628251 | 42.8965% |
| eth3d | playground | sampled256 | 32 | True | 10.7527% | 66.8548% | 0.408047 | 0.880580 | 0.644314 | 41.7816% |
| eth3d | relief | sampled32 | 19 | True | 21.6374% | 80.3704% | 0.681120 | 0.691028 | 0.686074 | 5.6282% |
| eth3d | relief | sampled256 | 19 | True | 13.0604% | 74.4444% | 0.714746 | 0.800713 | 0.757729 | 4.6410% |
| eth3d | relief_2 | sampled32 | 20 | True | 2.6316% | 47.1579% | 0.472623 | 0.610511 | 0.541567 | 16.3563% |
| eth3d | relief_2 | sampled256 | 20 | True | 2.6316% | 37.4561% | 0.641780 | 0.825524 | 0.733652 | 4.2456% |
| eth3d | terrains | sampled32 | 42 | True | 7.0461% | 60.1626% | 0.395513 | 0.241742 | 0.318627 | 62.6152% |
| eth3d | terrains | sampled256 | 42 | True | 7.8978% | 65.0097% | 0.339505 | 0.212499 | 0.276002 | 65.7688% |

Source unchanged=True;scene audits=36;missing=[];max prior patch-mass error=6.61611557006836e-05;actual stored-log conditional normalization error=0.0005905628204345703.

Raw metrics,per-layer rows,source snapshot/hashes and all GPU monitors are preserved. Numerical and two-frame diagnostic runs report all six fields N/A with concrete reasons. The first numerical graph test required bitwise equality of atomic cluster sums;the corrected harness keeps exact prior equality and tolerates FP32 atomic summation rounding(~1e-8),without changing the estimator. Completed quality evaluation does not imply quality equivalence or automatic default promotion.

## Final interpretation

Same-batch complete-forward medians:32=15.222148089s,256=14.366136085s.
256 reduces median wall time by0.856012004s(5.62346%)relative to32.
Separate-profile Attention+inline log probability stages are3.606277s/2.911825s;
these are24-layer totals and include the inline export work. The budget/state
stage remains3.900263s/3.928120s. Do not sum profile numbers into warmed medians.

Compared with32,256 raises7Scenes F1 by0.597578percentage points,but increases
CD by2.087174%. ETH3D F1 falls2.055319points,CD increases6.084394%,and AUC@30
falls0.501285points. Thus256 trades reconstruction quality for a modest speed
benefit;32 is the more conservative choice for ETH3D and reconstruction distance.
Neither configuration is promoted automatically.

The inherited m5c_evidence_and_host_copy stage name now refers to GPU-only
completion/cache/normalization checks;the compact sampled cache does not copy
evidence to the CPU. outside_named_stages mixes shared model work with host,
allocator and uninstrumented gaps;it is not a measurement of unavoidable cost.

All36 six-metric audits,38 exclusive GPU monitors and737 frozen source files
pass independent verification. GPU6 initially waited for an unrelated trace to
finish;it then ran the original job. A prospective duplicate GPU4 lease was
cancelled before any child/CUDA execution,so no duplicate quality run occurred.
See resource_wait_note.json. Only the two changed methods were evaluated.

A complete frozen source archive and every per-run source hash are published;
redundant per-scene uncompressed source tar files remain local. No quality
parameters or retention thresholds were tuned after inspecting test results.
