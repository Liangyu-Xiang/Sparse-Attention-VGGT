# Physical History–Rep Budget Ablation Analysis



## 1. Evaluated revision

Upstream commit:1c8019197547c74c2e357748198f5d9246536150; execution commit:39e9c178af02906a756b5be85f96121fa8d9e55f.

Stage0:three requested modules py_compile passed; original suite11passed, expanded diagnostic suite14passed. Existing diagnostic compatibility plugin supplies FP32 sparse.mm and the same four-effective-probe padded kernel. Capture-only merged-History instrumentation was corrected and tested; production capture=False math is upstream unchanged.

Config:physical_overlap_accounting=true; control=false; all other settings frozen. Hardware/UUID assignments:accounting_complete.json,pipeline_complete.json,quality_complete.json.

Seven-scene128 matched original checkpoint/frame lists, uniform seed0;1warmup/3complete forward measurements; capture=False. AUC/F1 percent, distances metres; CD=(ACC+Comp)/2 unsquared; recon_unposed threshold.05m/downsample.0078125m. No baseline reruns, no truth tuning. Diagnostic six quality metrics and paired speedup N/A.



## 2. Matched two-frame accounting sanity

| stage | mode | optimizer_cost_mean | counts_mean | physical_budget_utilization_mean | optimizer_physical_gap_mean | unused_budget_mean | history_rep_candidate_overlap_mean | history_rep_merges_mean | history_exact_overlap_selected_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0-L8 | control | 405.96297 | 351.16331 | 0.86493 | 54.79966 | 54.83670 | 172.24411 | 54.77778 | 0.00000 |
| L9-L16 | control | 405.91478 | 335.18562 | 0.82558 | 70.72917 | 70.81440 | 197.26326 | 69.82576 | 0.00000 |
| L17-L23 | control | 405.89395 | 336.51733 | 0.82886 | 69.37662 | 69.48269 | 193.78355 | 69.10606 | 0.00000 |
| L0-L8 | physical | 405.98654 | 405.98654 | 0.99997 | 0.00000 | 0.01347 | 211.90741 | 205.18687 | 0.00000 |
| L9-L16 | physical | 405.95455 | 405.95455 | 0.99989 | 0.00000 | 0.04545 | 243.97160 | 230.00000 | 0.00000 |
| L17-L23 | physical | 405.97836 | 405.97836 | 0.99995 | 0.00000 | 0.02165 | 239.78789 | 234.51299 | 0.00000 |

Stage1 input frames are the same uniform Chess endpoint frames in both arms; Stage2 uses frames0/20. Physical-support uniqueness/budget audits and explicit acceptance are archived. Read acceptance.json for the actual measured decision; no utilization threshold was fitted to quality.



## 3. History filtering after accounting correction

| layer | candidate_density | selected_density | recall | precision | enrichment | mass |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.19594 | 0.17198 | 0.25669 | 0.29848 | 1.49385 | 0.24336 |
| 10 | 0.19594 | 0.16123 | 0.21289 | 0.26376 | 1.32008 | 0.21671 |
| 11 | 0.19591 | 0.16580 | 0.18851 | 0.22719 | 1.13703 | 0.21315 |
| 12 | 0.19593 | 0.15977 | 0.18138 | 0.22678 | 1.13498 | 0.18929 |
| 13 | 0.19593 | 0.16654 | 0.19017 | 0.22802 | 1.14118 | 0.19834 |
| 14 | 0.19592 | 0.15560 | 0.20357 | 0.26087 | 1.30559 | 0.21874 |
| 15 | 0.19591 | 0.14128 | 0.19645 | 0.27675 | 1.38509 | 0.21792 |
| 16 | 0.19590 | 0.14049 | 0.16661 | 0.23717 | 1.18697 | 0.16823 |

All values except enrichment are fractions0–1. Precision/enrichment use current Top20 on the same sparse trajectory. New minus old endpoints alone cannot establish monotonic sharpening.

| layer | old_precision | new_precision | old_enrichment | new_enrichment |
| --- | --- | --- | --- | --- |
| 9 | 0.44604 | 0.29848 | 2.23238 | 1.49385 |
| 10 | 0.38239 | 0.26376 | 1.91378 | 1.32008 |
| 11 | 0.29754 | 0.22719 | 1.48912 | 1.13703 |
| 12 | 0.30101 | 0.22678 | 1.50652 | 1.13498 |
| 13 | 0.29588 | 0.22802 | 1.48084 | 1.14118 |
| 14 | 0.32420 | 0.26087 | 1.62255 | 1.30559 |
| 15 | 0.33947 | 0.27675 | 1.69900 | 1.38509 |
| 16 | 0.27298 | 0.23717 | 1.36624 | 1.18697 |

Strictly monotonic L10–L15 enrichment: False. This is descriptive, not a tuned success criterion.



## 4. Seven-scene macro quality

| method | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Dense historical | 23.04204 | 84.29269 | 0.14804 | 0.12436 | 0.13620 | 48.05374 |
| SAF3R historical | 23.60271 | 84.43107 | 0.12976 | 0.12427 | 0.12701 | 49.16638 |
| Old reuse+residual historical | 22.20601 | 83.82997 | 0.12183 | 0.11292 | 0.11737 | 50.44266 |
| 57bdb78 | 22.30034 | 84.30616 | 0.13782 | 0.12741 | 0.13262 | 46.80339 |
| All-edge/no-coverage | 22.65127 | 84.43552 | 0.13144 | 0.12824 | 0.12984 | 46.31437 |
| Physical H/Rep | 22.41400 | 84.31466 | 0.14087 | 0.13278 | 0.13682 | 46.20702 |

New minus all-edge/no-coverage (AUC/F1 percentage points; distances metres): {"auc03": -0.23727502812148082, "auc30": -0.12086379827519478, "acc": 0.009421912372508445, "comp": 0.004538851903589658, "cd": 0.006980382138049079, "f1": -0.10734889066512099}

Per-scene six metrics:

| scene | method | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heads | Dense historical | 51.62812 | 94.00714 | 0.03407 | 0.03081 | 0.03244 | 85.19571 |
| heads | SAF3R historical | 50.97195 | 93.85007 | 0.03250 | 0.03005 | 0.03127 | 85.09014 |
| heads | Old reuse+residual historical | 42.84367 | 92.59555 | 0.03657 | 0.03314 | 0.03486 | 82.06291 |
| heads | 57bdb78 | 46.89140 | 93.45554 | 0.03472 | 0.03301 | 0.03387 | 83.50833 |
| heads | All-edge/no-coverage | 45.44783 | 93.16232 | 0.03420 | 0.03381 | 0.03401 | 82.18456 |
| heads | Physical H/Rep | 45.21818 | 93.11885 | 0.03412 | 0.03223 | 0.03318 | 83.22430 |
| chess | Dense historical | 20.42733 | 88.46621 | 0.14207 | 0.08831 | 0.11519 | 35.29243 |
| chess | SAF3R historical | 20.16486 | 88.37927 | 0.14415 | 0.08074 | 0.11245 | 37.75651 |
| chess | Old reuse+residual historical | 22.52297 | 89.13181 | 0.13315 | 0.08637 | 0.10976 | 37.09496 |
| chess | 57bdb78 | 21.12451 | 88.55807 | 0.12476 | 0.08920 | 0.10698 | 36.13336 |
| chess | All-edge/no-coverage | 21.57972 | 88.66839 | 0.11661 | 0.08552 | 0.10106 | 44.67872 |
| chess | Physical H/Rep | 20.52986 | 88.43914 | 0.12019 | 0.07864 | 0.09941 | 45.41633 |
| stairs | Dense historical | 4.81873 | 80.94160 | 0.22003 | 0.48109 | 0.35056 | 15.15768 |
| stairs | SAF3R historical | 8.52608 | 83.55192 | 0.19683 | 0.48621 | 0.34152 | 16.45867 |
| stairs | Old reuse+residual historical | 5.57333 | 76.93734 | 0.22320 | 0.43405 | 0.32862 | 15.07784 |
| stairs | 57bdb78 | 7.43930 | 82.06611 | 0.23032 | 0.47873 | 0.35452 | 14.14482 |
| stairs | All-edge/no-coverage | 7.30807 | 81.38492 | 0.22951 | 0.49227 | 0.36089 | 13.92106 |
| stairs | Physical H/Rep | 8.31693 | 81.97917 | 0.23213 | 0.49510 | 0.36362 | 13.82124 |
| fire | Dense historical | 55.11811 | 94.52305 | 0.04902 | 0.05125 | 0.05013 | 66.59086 |
| fire | SAF3R historical | 56.49606 | 94.58374 | 0.04923 | 0.05482 | 0.05202 | 67.02756 |
| fire | Old reuse+residual historical | 52.98146 | 93.66060 | 0.04095 | 0.04871 | 0.04483 | 72.98486 |
| fire | 57bdb78 | 52.59596 | 94.08136 | 0.05136 | 0.05243 | 0.05189 | 65.24269 |
| fire | All-edge/no-coverage | 53.82628 | 93.95956 | 0.05005 | 0.05368 | 0.05186 | 65.17505 |
| fire | Physical H/Rep | 53.85089 | 94.25320 | 0.04798 | 0.05266 | 0.05032 | 66.56591 |
| office | Dense historical | 14.15272 | 77.13419 | 0.04718 | 0.05242 | 0.04980 | 60.97041 |
| office | SAF3R historical | 13.43914 | 75.83251 | 0.04782 | 0.05687 | 0.05235 | 58.48455 |
| office | Old reuse+residual historical | 14.53412 | 78.07866 | 0.05439 | 0.05558 | 0.05498 | 54.09283 |
| office | 57bdb78 | 13.24639 | 76.91929 | 0.05179 | 0.05594 | 0.05387 | 55.11043 |
| office | All-edge/no-coverage | 13.63599 | 77.44300 | 0.05636 | 0.05926 | 0.05781 | 49.51491 |
| office | Physical H/Rep | 13.76312 | 76.91765 | 0.05580 | 0.05822 | 0.05701 | 50.86173 |
| pumpkin | Dense historical | 6.70522 | 75.20833 | 0.46392 | 0.08467 | 0.27430 | 26.39064 |
| pumpkin | SAF3R historical | 7.11942 | 74.73466 | 0.35765 | 0.07522 | 0.21643 | 37.03154 |
| pumpkin | Old reuse+residual historical | 5.86860 | 74.50582 | 0.29429 | 0.07621 | 0.18525 | 31.28626 |
| pumpkin | 57bdb78 | 6.42224 | 75.30266 | 0.39347 | 0.09910 | 0.24628 | 27.20287 |
| pumpkin | All-edge/no-coverage | 6.29921 | 75.35925 | 0.35545 | 0.08777 | 0.22161 | 24.38739 |
| pumpkin | Physical H/Rep | 6.14337 | 75.22474 | 0.41332 | 0.11559 | 0.26445 | 21.69191 |
| redkitchen | Dense historical | 8.44406 | 79.76829 | 0.07995 | 0.08195 | 0.08095 | 46.77843 |
| redkitchen | SAF3R historical | 8.50148 | 80.08530 | 0.08014 | 0.08598 | 0.08306 | 42.31569 |
| redkitchen | Old reuse+residual historical | 11.11795 | 81.90002 | 0.07026 | 0.05635 | 0.06330 | 60.49893 |
| redkitchen | 57bdb78 | 8.38255 | 79.76009 | 0.07833 | 0.08346 | 0.08089 | 46.28119 |
| redkitchen | All-edge/no-coverage | 10.46178 | 81.07119 | 0.07794 | 0.08535 | 0.08165 | 44.33890 |
| redkitchen | Physical H/Rep | 9.07562 | 80.26985 | 0.08252 | 0.09700 | 0.08976 | 41.86773 |



## 5. Seven-scene layer behavior

| stage | utilization | optimizer_gap | unused | history_candidates | selected_history | history_fraction | merged_h_rep | population_rep_clusters | exact | history_exact_residual | density | additive_rep_keys | stored_history_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0-L8 | 0.99994 | 0.00000 | 1.62243 | 23117.74532 | 18463.57409 | 0.70994 | 13558.71853 | 21102.52130 | 0.00000 | 0.00000 | 0.19998 | 7543.80270 | 619.70602 |
| L9-L16 | 0.99457 | 95.75121 | 141.18389 | 25911.76702 | 20940.16347 | 0.80931 | 15777.29930 | 20604.20850 | 100.74278 | 95.85673 | 0.19894 | 4826.90913 | 618.96016 |
| L17-L23 | 0.99978 | 0.00027 | 5.76442 | 25952.87580 | 20863.24781 | 0.80233 | 15412.39481 | 20551.81003 | 0.57205 | 0.00027 | 0.19995 | 5139.41505 | 619.70602 |

Fractions are0–1. history_fraction uses physical selected History / physical PATCH support as requested. Additive_rep_keys excludes merged H+Rep; population_rep_clusters includes them. Complete L0–L23:

| layer | utilization | optimizer_gap | unused | history_candidates | selected_history | history_fraction | merged_h_rep | population_rep_clusters | exact | history_exact_residual | density | additive_rep_keys | stored_history_mib |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 26008.99805 | 0.00000 | 0.00000 | 0.20000 | 26008.99805 | 619.70602 |
| 1 | 0.99995 | 0.00000 | 1.23593 | 26008.99805 | 19054.18862 | 0.73263 | 14345.51842 | 21299.09347 | 0.00000 | 0.00000 | 0.19999 | 6953.57499 | 619.70602 |
| 2 | 0.99994 | 0.00000 | 1.66138 | 26007.76395 | 19422.31083 | 0.74680 | 13038.07631 | 19623.10352 | 0.00000 | 0.00000 | 0.19998 | 6585.02672 | 619.70602 |
| 3 | 0.99993 | 0.00000 | 1.90782 | 26007.33789 | 19851.30664 | 0.76330 | 13984.36928 | 20140.15569 | 0.00000 | 0.00000 | 0.19998 | 6155.78557 | 619.70602 |
| 4 | 0.99992 | 0.00000 | 2.13123 | 26007.09068 | 21022.03767 | 0.80833 | 15736.86426 | 20721.69559 | 0.00000 | 0.00000 | 0.19998 | 4984.83098 | 619.70602 |
| 5 | 0.99993 | 0.00000 | 1.89567 | 26006.86802 | 21177.41350 | 0.81429 | 15769.89272 | 20599.58259 | 0.00000 | 0.00000 | 0.19998 | 4829.68987 | 619.70602 |
| 6 | 0.99994 | 0.00000 | 1.59387 | 26007.10352 | 21973.74721 | 0.84490 | 16480.47377 | 20514.13198 | 0.00000 | 0.00000 | 0.19998 | 4033.65845 | 619.70602 |
| 7 | 0.99993 | 0.00000 | 1.85924 | 26007.40513 | 22114.54436 | 0.85033 | 16550.94727 | 20443.54185 | 0.00000 | 0.00000 | 0.19998 | 3892.59556 | 619.70602 |
| 8 | 0.99991 | 0.00000 | 2.31676 | 26007.14062 | 21556.61802 | 0.82889 | 16122.32478 | 20572.38895 | 0.00000 | 0.00000 | 0.19998 | 4450.06410 | 619.70602 |
| 9 | 0.99997 | 0.00000 | 0.86394 | 26006.68304 | 22479.10184 | 0.86431 | 16673.52706 | 20202.56027 | 0.00000 | 0.00000 | 0.19999 | 3529.03285 | 619.70602 |
| 10 | 0.99985 | 0.00000 | 3.83326 | 26008.13477 | 21712.81055 | 0.83494 | 16272.88198 | 20565.17188 | 0.06687 | 0.00000 | 0.19997 | 4292.28885 | 619.70602 |
| 11 | 0.99980 | 0.00765 | 5.17813 | 26005.16574 | 21618.97991 | 0.83138 | 16434.45201 | 20819.04129 | 0.25220 | 0.00765 | 0.19996 | 4384.58880 | 619.70602 |
| 12 | 0.99875 | 3.22732 | 32.63274 | 26003.82059 | 21059.73800 | 0.81073 | 16389.27776 | 21284.21177 | 21.69514 | 3.22732 | 0.19975 | 4894.93380 | 619.70602 |
| 13 | 0.99874 | 10.15922 | 32.73161 | 25976.36719 | 21518.43415 | 0.82838 | 16739.85686 | 21170.22684 | 27.46326 | 10.17277 | 0.19975 | 4430.37019 | 619.70602 |
| 14 | 0.98960 | 89.69425 | 270.53598 | 25976.26842 | 19905.56669 | 0.77336 | 14923.71875 | 20441.03013 | 315.58470 | 89.74386 | 0.19797 | 5517.31194 | 619.29756 |
| 15 | 0.98348 | 343.52490 | 429.76601 | 25738.46317 | 20064.60017 | 0.78443 | 14884.28739 | 20052.81250 | 346.10775 | 343.86132 | 0.19677 | 5168.52567 | 617.10892 |
| 16 | 0.98639 | 319.39635 | 353.92948 | 25579.23326 | 19162.07645 | 0.74692 | 13900.39258 | 20298.61328 | 94.77237 | 319.84094 | 0.19734 | 6398.22091 | 616.74471 |
| 17 | 0.99981 | 0.00000 | 4.87924 | 25655.07003 | 19613.30664 | 0.75424 | 13963.68917 | 20353.39537 | 1.10707 | 0.00000 | 0.19996 | 6389.70619 | 619.70602 |
| 18 | 0.99988 | 0.00000 | 3.07201 | 26004.12026 | 19436.68359 | 0.74739 | 13761.03390 | 20330.27762 | 0.00000 | 0.00000 | 0.19997 | 6569.24344 | 619.70602 |
| 19 | 0.99919 | 0.00189 | 21.13953 | 26005.92718 | 21049.80329 | 0.80999 | 15922.60798 | 20858.95843 | 1.70615 | 0.00189 | 0.19984 | 4936.35045 | 619.70602 |
| 20 | 0.99996 | 0.00000 | 1.15672 | 25987.85910 | 21673.28962 | 0.83334 | 15936.93164 | 20271.48493 | 0.00000 | 0.00000 | 0.19999 | 4334.55315 | 619.70602 |
| 21 | 0.99973 | 0.00000 | 7.03925 | 26007.84208 | 21286.78237 | 0.81866 | 15820.18722 | 20534.17411 | 1.19109 | 0.00000 | 0.19994 | 4713.98626 | 619.70602 |
| 22 | 0.99994 | 0.00000 | 1.64852 | 26001.96094 | 20955.69782 | 0.80576 | 15983.68555 | 21035.33873 | 0.00000 | 0.00000 | 0.19998 | 5051.65304 | 619.70602 |
| 23 | 0.99995 | 0.00000 | 1.41567 | 26007.35100 | 22027.17132 | 0.84695 | 16498.62821 | 20479.04102 | 0.00000 | 0.00000 | 0.19998 | 3980.41281 | 619.70602 |



## 6. Runtime and memory

| method | seconds | peak_allocated_gib | peak_reserved_gib | density | utilization | history_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| Dense historical | 17.11768 | 14.35857 | 17.55469 | 1.00000 | nan | nan |
| SAF3R historical | 8.76281 | 14.61248 | 18.02539 | nan | nan | nan |
| Old reuse+residual historical | 116.74196 | 15.49932 | 20.28125 | 0.19978 | 0.99890 | 0.08541 |
| 57bdb78 | 72.62345 | 15.69668 | 19.68750 | 0.19905 | 0.99517 | 0.00823 |
| All-edge/no-coverage | 131.83629 | 17.07543 | 22.30469 | 0.16765 | 0.83431 | 0.50653 |
| Physical H/Rep | 159.18326 | 17.40879 | 23.04297 | 0.19963 | 0.99810 | 0.77001 |

Times are different historical phases, not contemporaneous paired speedups. New end-to-end Dense speedup N/A; no paired Dense rerun was authorized. Timing includes all initialization, clustering, routing, history and reordering. Exploration still does not shrink as History grows. This is not the final efficient implementation.



## 7. Causal conclusions

### Budget accounting

L9–L16 full128 utilization 0.820215→0.994572; unused budget 4676.035→141.184; actual density 0.164903→0.198936. New optimizer/physical gap 95.751. Frozen old128 records do NOT contain optimizer cost: old128 gap is N/A, not nominal budget minus executed keys. The matched same-code two-frame control above provides the direct old/new gap measurement.

### Quality effect of restoring physical budget

Quality changes above are attributable to the tested physical overlap accounting under frozen scoring/configuration, but not proof that any one scene must improve. Inspect Chess/Fire/Pumpkin/RedKitchen rows and all six metrics; do not equate lower CD with higher F1.

### Remaining accounting issue

L9–L16 selected History/Exact residual 95.857 keys/group; measured optimizer gap 95.751. Reported separately from integer solver slack. Neither Exact overlap nor budget backfilling was fixed in this experiment.

### History filtering

Precision/enrichment trend is tabulated against the prior experiment, not inferred from a higher History fraction. Targets evolve with each model trajectory; these are not controlled identical-QK masks.



## 8. Exactly one next recommendation

Run one separate History/Exact physical-overlap accounting ablation: residual overlap explains a material remaining gap. Freeze all scoring, clustering, Rep/Exact criteria, grouping, population and retention.

This recommendation has NOT been implemented.
