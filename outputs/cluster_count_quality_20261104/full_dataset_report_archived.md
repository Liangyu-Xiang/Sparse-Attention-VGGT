# Cluster-count sweep:32 Query history,20% Attention budget

Campaign complete: **True**. Only15%/10%/5% cluster variants rerun;54 quality runs. No test-truth tuning or changed quality gates.

## Method and scope

Base c85543d. Same selected-Key inline probability history every32Queries,coarse calibration and missing-token completion. Only cluster_fraction is varied;20% Attention retention budget,states,loss,16-round allocation and mass transfer fixed. The old guard F>=budget is removed. First layer uses one real representative for every cluster,then deterministic uniformly spaced non-representative real members to fill the SAME budget. Extra members have weight1,subtracted from their representative weights;cluster mass is conserved and IDs are unique. This initialization adaptation is necessary because old initialization duplicates representatives for F<budget. Legacy F>=budget path produces identical seed IDs/masses. Initial-layer state1 diagnostics count representative clusters;extra members are explicitly reported separately,not claimed to be allocator state choices.

Larger clusters can increase geometry width and trigger existing width>64 fallback instead of fused state/reused graph kernels. The initial width256 history kernel exceeded shared memory,and one width>256 scene failed its guard. Retry preserves identical launches for width<=128,caps history tiles at8192 elements for wider clusters,and extends exact geometry/drop bins to512;no cluster membership or state math changes. Successful first-attempt runs with width<=128 and clean exclusive monitors are reused;all failed/overlapped runs are retried. Initial failed runs and source snapshot remain in cluster_count_quality_attempt1_20261104. Synthetic widths32–512 preserve exact owners and losses within FP32 reduction tolerance. Actual per-layer densities and geometry widths are in per_layer.csv. Retention is an upper budget,not an exact guarantee for the discrete allocator.

## Warmed full-forward runtime

All3 variants timed on physicalGPU2 RTX5090D with identical heads128 frames/checkpoint/preprocessing,one warmup+three full forwards+separate profile. Includes initialization inside forward,clustering,routing,history,reordering and prediction heads. Initial batch used all idle GPUs0–7;after other sessions occupied0/3/4/7,the retry batch used idle1/2/5/6,one CUDA program per physical GPU,CPU-only exclusive leases and overlap monitoring.

| cluster_percent | median_seconds | peak_allocated_gib | dense_speedup | samples |
| --- | --- | --- | --- | --- |
| 15 | 14.456317 | 18.237918 | N/A | [14.456316964700818, 14.458854969590902, 14.410488390363753] |
| 10 | 14.517267 | 17.518579 | N/A | [14.633027207106352, 14.517267497256398, 14.170704883988947] |
| 5 | 20.388079 | 15.795390 | N/A | [20.406122067943215, 20.38807929493487, 20.272130103316158] |

Unchanged sampled32 baseline approximately20% clusters:15.222148s historical warmed median from sampled_history_quality_20261104;not rerun and not a paired speed comparison. Dense speedup=N/A because unchanged Dense not rerun under current user scope. Historical SAF3R timings are also not paired. Dense identity=1.00x.

Profile percentages use separate profile wall,not warmed median. Outside named stages includes original shared work and unclassified dispatch gaps,not wholly unavoidable Dense cost.

| cluster_percent | operation | seconds | percent_profile |
| --- | --- | --- | --- |
| 15 | value_update | 0.851522 | 5.809673 |
| 15 | pooling | 0.085738 | 0.584962 |
| 15 | cluster_seed_prepare | 0.000336 | 0.002290 |
| 15 | m5c_coarse_observation | 0.132384 | 0.903215 |
| 15 | execution_and_evidence | 3.598936 | 24.554421 |
| 15 | sampled_probe_grid_summary | 0.203264 | 1.386807 |
| 15 | m5c_evidence_and_host_copy | 0.254976 | 1.739623 |
| 15 | history_summary | 0.037472 | 0.255663 |
| 15 | coarse_observation | 0.029236 | 0.199465 |
| 15 | residual_fine_probe | 0.317536 | 2.166448 |
| 15 | reuse_and_budget | 4.163671 | 28.407434 |
| 15 | outside_named_stages | 4.981906 | 33.989999 |
| 10 | value_update | 0.680356 | 4.493976 |
| 10 | pooling | 0.087633 | 0.578844 |
| 10 | cluster_seed_prepare | 0.000443 | 0.002929 |
| 10 | m5c_coarse_observation | 0.229769 | 1.517700 |
| 10 | execution_and_evidence | 3.622009 | 23.924580 |
| 10 | sampled_probe_grid_summary | 0.205673 | 1.358539 |
| 10 | m5c_evidence_and_host_copy | 0.248575 | 1.641918 |
| 10 | history_summary | 0.036182 | 0.238997 |
| 10 | coarse_observation | 0.029945 | 0.197800 |
| 10 | residual_fine_probe | 0.178901 | 1.181704 |
| 10 | reuse_and_budget | 4.798701 | 31.697028 |
| 10 | outside_named_stages | 5.021091 | 33.165984 |
| 5 | value_update | 0.563215 | 2.632821 |
| 5 | pooling | 0.094206 | 0.440380 |
| 5 | cluster_seed_prepare | 0.000320 | 0.001494 |
| 5 | m5c_coarse_observation | 0.131849 | 0.616346 |
| 5 | execution_and_evidence | 3.623117 | 16.936719 |
| 5 | sampled_probe_grid_summary | 0.218541 | 1.021596 |
| 5 | m5c_evidence_and_host_copy | 0.252132 | 1.178623 |
| 5 | history_summary | 0.038124 | 0.178216 |
| 5 | coarse_observation | 0.029831 | 0.139449 |
| 5 | residual_fine_probe | 0.132898 | 0.621249 |
| 5 | reuse_and_budget | 11.093396 | 51.857479 |
| 5 | outside_named_stages | 5.214455 | 24.375630 |

Nested AV/probability export already included in execution_and_evidence:

| cluster_percent | operation | seconds | scope |
| --- | --- | --- | --- |
| 15 | sampled_direct_av | 3.538814 | nested in execution_and_evidence;do not add twice |
| 15 | sampled_probability_export | 0.051224 | nested in execution_and_evidence;do not add twice |
| 10 | sampled_direct_av | 3.561309 | nested in execution_and_evidence;do not add twice |
| 10 | sampled_probability_export | 0.052198 | nested in execution_and_evidence;do not add twice |
| 5 | sampled_direct_av | 3.561166 | nested in execution_and_evidence;do not add twice |
| 5 | sampled_probability_export | 0.053077 | nested in execution_and_evidence;do not add twice |

## Six-metric quality

All7 7Scenes scenes128frames and11 ETH3D scenes14–76frames,pose+recon_unposed,equal-scene macro averages. Exact image paths/indices,checkpoint,shape,protocol audited against reused sampled32. AUC/F1 percentages in tables,fractions in JSON;distances metres,unsquared CD=(ACC+Comp)/2. Threshold/downsample7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. Dense/SAF3R/sampled32 controls are historical local measurements. SAF3R local lazy first-correspondence-head Top4/ETH3D profile is not paper DINO feature Top-K or Top128. No published-reference scores are mixed into local tables.

| dataset | method | scenes | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | dense | 7 | True | 23.6689% | 84.6389% | 0.153741 | 0.125616 | 0.139678 | 48.0128% |
| 7scenes | saf3r | 7 | True | 23.9349% | 84.7131% | 0.143263 | 0.127883 | 0.135573 | 47.3154% |
| 7scenes | sampled32_baseline | 7 | True | 22.7040% | 84.4103% | 0.144742 | 0.123145 | 0.133944 | 46.5072% |
| 7scenes | clusters15 | 7 | True | 22.1345% | 84.3702% | 0.136966 | 0.118777 | 0.127872 | 48.8924% |
| 7scenes | clusters10 | 7 | True | 22.2441% | 84.1773% | 0.149419 | 0.120583 | 0.135001 | 48.2532% |
| 7scenes | clusters5 | 7 | True | 22.3056% | 84.1978% | 0.146492 | 0.119510 | 0.133001 | 45.7845% |
| eth3d | dense | 11 | True | 29.1835% | 81.8162% | 0.313193 | 0.848854 | 0.581024 | 62.5731% |
| eth3d | saf3r | 11 | True | 23.3322% | 80.1747% | 0.303106 | 0.857720 | 0.580413 | 62.8304% |
| eth3d | sampled32_baseline | 11 | True | 21.6994% | 78.2119% | 0.334705 | 0.881240 | 0.607972 | 54.6464% |
| eth3d | clusters15 | 11 | True | 21.4417% | 78.1789% | 0.323627 | 0.888496 | 0.606061 | 56.1086% |
| eth3d | clusters10 | 11 | True | 22.3049% | 77.9751% | 0.359489 | 0.901419 | 0.630454 | 55.0861% |
| eth3d | clusters5 | 11 | True | 20.3036% | 77.5982% | 0.349969 | 0.921173 | 0.635571 | 52.5231% |

## Per-scene metrics

| dataset | scene | method | frames | complete | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | heads | dense | 128 | True | 51.5789% | 93.9944% | 0.034306 | 0.030839 | 0.032573 | 84.9976% |
| 7scenes | heads | saf3r | 128 | True | 50.9596% | 93.8611% | 0.032851 | 0.030137 | 0.031494 | 85.1315% |
| 7scenes | heads | sampled32_baseline | 128 | True | 52.4934% | 94.0203% | 0.036200 | 0.032084 | 0.034142 | 83.6571% |
| 7scenes | heads | clusters15 | 128 | True | 48.1258% | 93.5507% | 0.034819 | 0.032112 | 0.033466 | 83.9809% |
| 7scenes | heads | clusters10 | 128 | True | 48.3801% | 93.4945% | 0.035253 | 0.031032 | 0.033143 | 84.2716% |
| 7scenes | heads | clusters5 | 128 | True | 49.6842% | 93.6786% | 0.035559 | 0.032113 | 0.033836 | 84.1457% |
| 7scenes | chess | dense | 128 | True | 20.5135% | 88.4658% | 0.139921 | 0.087736 | 0.113828 | 35.9524% |
| 7scenes | chess | saf3r | 128 | True | 20.1198% | 88.3756% | 0.142773 | 0.079959 | 0.111366 | 38.5585% |
| 7scenes | chess | sampled32_baseline | 128 | True | 14.7310% | 86.8086% | 0.141736 | 0.089722 | 0.115729 | 36.1772% |
| 7scenes | chess | clusters15 | 128 | True | 17.9790% | 87.9097% | 0.138541 | 0.088569 | 0.113555 | 36.1513% |
| 7scenes | chess | clusters10 | 128 | True | 18.9181% | 88.0495% | 0.135078 | 0.084520 | 0.109799 | 38.2648% |
| 7scenes | chess | clusters5 | 128 | True | 20.4191% | 88.4231% | 0.137208 | 0.084269 | 0.110738 | 38.2905% |
| 7scenes | stairs | dense | 128 | True | 4.9172% | 80.9535% | 0.220771 | 0.479673 | 0.350222 | 15.3143% |
| 7scenes | stairs | saf3r | 128 | True | 8.2800% | 83.4117% | 0.199268 | 0.486030 | 0.342649 | 16.3935% |
| 7scenes | stairs | sampled32_baseline | 128 | True | 7.9601% | 81.9714% | 0.204990 | 0.466641 | 0.335816 | 16.5314% |
| 7scenes | stairs | clusters15 | 128 | True | 6.1721% | 80.9719% | 0.205345 | 0.453328 | 0.329337 | 15.2749% |
| 7scenes | stairs | clusters10 | 128 | True | 4.9869% | 79.9352% | 0.224531 | 0.458213 | 0.341372 | 14.7718% |
| 7scenes | stairs | clusters5 | 128 | True | 5.4831% | 80.0201% | 0.217229 | 0.431484 | 0.324356 | 15.8822% |
| 7scenes | fire | dense | 128 | True | 55.3150% | 94.5579% | 0.048892 | 0.051295 | 0.050093 | 66.7010% |
| 7scenes | fire | saf3r | 128 | True | 56.2787% | 94.5632% | 0.048822 | 0.054649 | 0.051735 | 67.5308% |
| 7scenes | fire | sampled32_baseline | 128 | True | 54.2118% | 94.4156% | 0.052484 | 0.049650 | 0.051067 | 65.0617% |
| 7scenes | fire | clusters15 | 128 | True | 52.6001% | 94.1207% | 0.052016 | 0.051596 | 0.051806 | 61.6552% |
| 7scenes | fire | clusters10 | 128 | True | 53.4203% | 94.3381% | 0.054795 | 0.051416 | 0.053106 | 64.8962% |
| 7scenes | fire | clusters5 | 128 | True | 51.1729% | 93.9321% | 0.054684 | 0.053040 | 0.053862 | 60.5354% |
| 7scenes | office | dense | 128 | True | 14.2758% | 77.1457% | 0.047099 | 0.052099 | 0.049599 | 61.0877% |
| 7scenes | office | saf3r | 128 | True | 13.4186% | 75.8743% | 0.048098 | 0.056819 | 0.052458 | 58.5554% |
| 7scenes | office | sampled32_baseline | 128 | True | 12.0366% | 77.3150% | 0.055885 | 0.057152 | 0.056518 | 48.8738% |
| 7scenes | office | clusters15 | 128 | True | 12.7092% | 77.2777% | 0.045501 | 0.050866 | 0.048184 | 64.2430% |
| 7scenes | office | clusters10 | 128 | True | 12.4344% | 76.7462% | 0.048687 | 0.053014 | 0.050851 | 59.8028% |
| 7scenes | office | clusters5 | 128 | True | 11.9587% | 77.2732% | 0.056364 | 0.061302 | 0.058833 | 49.3701% |
| 7scenes | pumpkin | dense | 128 | True | 10.6176% | 77.5980% | 0.505694 | 0.096496 | 0.301095 | 25.0739% |
| 7scenes | pumpkin | saf3r | 128 | True | 10.0025% | 76.8221% | 0.450419 | 0.101176 | 0.275798 | 22.8550% |
| 7scenes | pumpkin | sampled32_baseline | 128 | True | 9.8343% | 77.1953% | 0.442734 | 0.089508 | 0.266121 | 26.6731% |
| 7scenes | pumpkin | clusters15 | 128 | True | 10.0066% | 77.2371% | 0.410971 | 0.091217 | 0.251094 | 26.4557% |
| 7scenes | pumpkin | clusters10 | 128 | True | 10.5848% | 77.2982% | 0.470975 | 0.099438 | 0.285207 | 23.2269% |
| 7scenes | pumpkin | clusters5 | 128 | True | 10.2485% | 77.1174% | 0.452134 | 0.105417 | 0.278776 | 19.6505% |
| 7scenes | redkitchen | dense | 128 | True | 8.4646% | 79.7568% | 0.079504 | 0.081175 | 0.080339 | 46.9624% |
| 7scenes | redkitchen | saf3r | 128 | True | 8.4851% | 80.0837% | 0.080608 | 0.086410 | 0.083509 | 42.1834% |
| 7scenes | redkitchen | sampled32_baseline | 128 | True | 7.6608% | 79.1462% | 0.079166 | 0.077260 | 0.078213 | 48.5765% |
| 7scenes | redkitchen | clusters15 | 128 | True | 7.3491% | 79.5235% | 0.071571 | 0.063753 | 0.067662 | 54.4861% |
| 7scenes | redkitchen | clusters10 | 128 | True | 6.9841% | 79.3791% | 0.076614 | 0.066448 | 0.071531 | 52.5380% |
| 7scenes | redkitchen | clusters5 | 128 | True | 7.1727% | 78.9403% | 0.072265 | 0.068947 | 0.070606 | 52.6172% |
| eth3d | courtyard | dense | 38 | True | 32.4798% | 88.7008% | 0.270738 | 0.185862 | 0.228300 | 58.1241% |
| eth3d | courtyard | saf3r | 38 | True | 19.2034% | 84.6657% | 0.190286 | 0.146474 | 0.168380 | 75.1637% |
| eth3d | courtyard | sampled32_baseline | 38 | True | 24.3243% | 85.9412% | 0.284511 | 0.168725 | 0.226618 | 59.6977% |
| eth3d | courtyard | clusters15 | 38 | True | 25.8890% | 86.7141% | 0.174870 | 0.112385 | 0.143627 | 83.5659% |
| eth3d | courtyard | clusters10 | 38 | True | 28.2598% | 86.9654% | 0.300503 | 0.180648 | 0.240575 | 56.9183% |
| eth3d | courtyard | clusters5 | 38 | True | 26.2684% | 87.8331% | 0.295319 | 0.180045 | 0.237682 | 56.0115% |
| eth3d | delivery_area | dense | 40 | True | 21.0684% | 85.1838% | 0.730870 | 1.512615 | 1.121743 | 11.1014% |
| eth3d | delivery_area | saf3r | 40 | True | 15.0000% | 80.3291% | 0.511141 | 1.382719 | 0.946930 | 36.4061% |
| eth3d | delivery_area | sampled32_baseline | 40 | True | 19.7436% | 80.4957% | 0.586009 | 1.424384 | 1.005197 | 20.4942% |
| eth3d | delivery_area | clusters15 | 40 | True | 17.4786% | 80.5641% | 0.481112 | 1.329648 | 0.905380 | 26.5894% |
| eth3d | delivery_area | clusters10 | 40 | True | 18.6752% | 83.3675% | 0.591858 | 1.437848 | 1.014853 | 16.2073% |
| eth3d | delivery_area | clusters5 | 40 | True | 18.9316% | 84.2650% | 0.646495 | 1.489687 | 1.068091 | 11.4469% |
| eth3d | electro | dense | 39 | True | 38.4166% | 93.0139% | 0.161289 | 5.012226 | 2.586758 | 66.0763% |
| eth3d | electro | saf3r | 39 | True | 25.1462% | 90.6658% | 0.359525 | 5.258731 | 2.809128 | 39.0095% |
| eth3d | electro | sampled32_baseline | 39 | True | 30.0045% | 91.6374% | 0.182112 | 5.020795 | 2.601454 | 64.0953% |
| eth3d | electro | clusters15 | 39 | True | 29.6446% | 91.9388% | 0.235721 | 5.080505 | 2.658113 | 54.1612% |
| eth3d | electro | clusters10 | 39 | True | 34.0081% | 92.4112% | 0.201067 | 5.070110 | 2.635589 | 63.9492% |
| eth3d | electro | clusters5 | 39 | True | 32.6136% | 91.9028% | 0.177593 | 5.067146 | 2.622370 | 66.5687% |
| eth3d | facade | dense | 76 | True | 36.2573% | 90.5357% | 0.263764 | 0.229859 | 0.246811 | 61.8670% |
| eth3d | facade | saf3r | 76 | True | 32.5497% | 89.3942% | 0.299112 | 0.278049 | 0.288580 | 59.0180% |
| eth3d | facade | sampled32_baseline | 76 | True | 28.4444% | 89.2246% | 0.328699 | 0.265275 | 0.296987 | 56.8043% |
| eth3d | facade | clusters15 | 76 | True | 25.7193% | 88.3825% | 0.329541 | 0.249441 | 0.289491 | 57.0360% |
| eth3d | facade | clusters10 | 76 | True | 27.8012% | 89.2409% | 0.427215 | 0.295461 | 0.361338 | 43.7509% |
| eth3d | facade | clusters5 | 76 | True | 28.5965% | 88.2327% | 0.271958 | 0.225915 | 0.248936 | 64.2017% |
| eth3d | kicker | dense | 31 | True | 57.9928% | 95.5412% | 0.050711 | 0.046187 | 0.048449 | 99.6216% |
| eth3d | kicker | saf3r | 31 | True | 51.3262% | 94.8029% | 0.061966 | 0.056630 | 0.059298 | 99.4875% |
| eth3d | kicker | sampled32_baseline | 31 | True | 50.2509% | 94.0430% | 0.058842 | 0.054125 | 0.056483 | 99.3788% |
| eth3d | kicker | clusters15 | 31 | True | 43.7993% | 93.3835% | 0.057644 | 0.047842 | 0.052743 | 99.2897% |
| eth3d | kicker | clusters10 | 31 | True | 48.6738% | 93.7563% | 0.056819 | 0.049756 | 0.053288 | 99.3642% |
| eth3d | kicker | clusters5 | 31 | True | 53.1183% | 94.4014% | 0.059422 | 0.049663 | 0.054542 | 99.4408% |
| eth3d | office | dense | 26 | True | 25.4359% | 79.7436% | 0.080806 | 0.083616 | 0.082211 | 99.0686% |
| eth3d | office | saf3r | 26 | True | 25.9487% | 76.9231% | 0.097604 | 0.100112 | 0.098858 | 98.8164% |
| eth3d | office | sampled32_baseline | 26 | True | 19.4872% | 76.4000% | 0.097288 | 0.095928 | 0.096608 | 98.9954% |
| eth3d | office | clusters15 | 26 | True | 20.8205% | 75.1282% | 0.057552 | 0.052335 | 0.054943 | 99.4080% |
| eth3d | office | clusters10 | 26 | True | 21.2308% | 77.0256% | 0.084137 | 0.081675 | 0.082906 | 99.3432% |
| eth3d | office | clusters5 | 26 | True | 19.3846% | 76.4821% | 0.052019 | 0.047827 | 0.049923 | 99.6672% |
| eth3d | pipes | dense | 14 | True | 42.4908% | 93.0403% | 0.187647 | 0.183130 | 0.185389 | 80.3675% |
| eth3d | pipes | saf3r | 14 | True | 36.6300% | 92.8205% | 0.224999 | 0.211783 | 0.218391 | 81.6348% |
| eth3d | pipes | sampled32_baseline | 14 | True | 24.9084% | 87.4725% | 0.240402 | 0.219258 | 0.229830 | 74.1484% |
| eth3d | pipes | clusters15 | 14 | True | 32.2344% | 89.5604% | 0.263111 | 0.291483 | 0.277297 | 76.2040% |
| eth3d | pipes | clusters10 | 14 | True | 30.0366% | 86.4835% | 0.274395 | 0.246352 | 0.260373 | 67.9478% |
| eth3d | pipes | clusters5 | 14 | True | 15.7509% | 83.0403% | 0.268906 | 0.244485 | 0.256695 | 75.5500% |
| eth3d | playground | dense | 32 | True | 18.6828% | 77.2177% | 0.340276 | 0.646137 | 0.493207 | 73.5691% |
| eth3d | playground | saf3r | 32 | True | 15.2554% | 72.9772% | 0.287455 | 0.693197 | 0.490326 | 70.1166% |
| eth3d | playground | sampled32_baseline | 32 | True | 10.2151% | 67.4261% | 0.354635 | 0.901867 | 0.628251 | 42.8965% |
| eth3d | playground | clusters15 | 32 | True | 9.6774% | 63.8306% | 0.424171 | 0.941463 | 0.682817 | 33.7593% |
| eth3d | playground | clusters10 | 32 | True | 8.9382% | 66.8145% | 0.363511 | 0.900418 | 0.631965 | 41.1110% |
| eth3d | playground | clusters5 | 32 | True | 9.4086% | 69.9664% | 0.395712 | 0.884130 | 0.639921 | 38.9978% |
| eth3d | relief | dense | 19 | True | 29.2398% | 85.1462% | 0.473799 | 0.459467 | 0.466633 | 42.1449% |
| eth3d | relief | saf3r | 19 | True | 21.8324% | 83.3528% | 0.530953 | 0.535222 | 0.533087 | 28.8515% |
| eth3d | relief | sampled32_baseline | 19 | True | 21.6374% | 80.3704% | 0.681120 | 0.691028 | 0.686074 | 5.6282% |
| eth3d | relief | clusters15 | 19 | True | 18.7135% | 80.2924% | 0.635796 | 0.674913 | 0.655355 | 4.0128% |
| eth3d | relief | clusters10 | 19 | True | 15.3996% | 77.9727% | 0.680185 | 0.699826 | 0.690006 | 11.0954% |
| eth3d | relief | clusters5 | 19 | True | 9.3567% | 76.9981% | 0.790017 | 0.917048 | 0.853533 | 1.5339% |
| eth3d | relief_2 | dense | 20 | True | 8.7719% | 47.5965% | 0.536483 | 0.696539 | 0.616511 | 30.0321% |
| eth3d | relief_2 | saf3r | 20 | True | 5.4386% | 50.5439% | 0.389656 | 0.534630 | 0.462143 | 37.3555% |
| eth3d | relief_2 | sampled32_baseline | 20 | True | 2.6316% | 47.1579% | 0.472623 | 0.610511 | 0.541567 | 16.3563% |
| eth3d | relief_2 | clusters15 | 20 | True | 2.2807% | 48.5439% | 0.604041 | 0.801495 | 0.702768 | 11.6664% |
| eth3d | relief_2 | clusters10 | 20 | True | 2.8070% | 48.5088% | 0.664784 | 0.717006 | 0.690895 | 37.6811% |
| eth3d | relief_2 | clusters5 | 20 | True | 2.6316% | 45.7895% | 0.548657 | 0.698870 | 0.623764 | 6.4575% |
| eth3d | terrains | dense | 42 | True | 10.1820% | 64.2586% | 0.348743 | 0.281758 | 0.315250 | 66.3317% |
| eth3d | terrains | saf3r | 42 | True | 8.3237% | 65.4472% | 0.381467 | 0.237370 | 0.309419 | 65.2747% |
| eth3d | terrains | sampled32_baseline | 42 | True | 7.0461% | 60.1626% | 0.395513 | 0.241742 | 0.318627 | 62.6152% |
| eth3d | terrains | clusters15 | 42 | True | 9.6012% | 61.6299% | 0.296333 | 0.191949 | 0.244141 | 71.5014% |
| eth3d | terrains | clusters10 | 42 | True | 9.5238% | 55.1800% | 0.309900 | 0.236513 | 0.273206 | 68.5787% |
| eth3d | terrains | clusters5 | 42 | True | 7.2784% | 54.6690% | 0.343566 | 0.328087 | 0.335827 | 57.8779% |

Source unchanged=True;audits=54;missing=[]. Raw results,layer records,source snapshot/hashes and GPU monitors preserved. Numerical/smoke/timing/failed runs carry all six fields as unavailable with concrete reasons;they are not complete quality evaluations.
