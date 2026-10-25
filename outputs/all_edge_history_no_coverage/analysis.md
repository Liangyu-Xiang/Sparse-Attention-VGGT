# All-executed-edge History + no-coverage causal analysis



Evaluated upstream1a7bb41d955da284eded195a2e0b11faf49c2ec3 with disclosed evaluation-only compatibility. Stage0 logs, Stage1 sanity, seven complete quality runs and all raw outputs are archived.

Only upstream all-edge eligibility + removal of coverage are under test. These two coupled upstream edits are not separated factorially; their individual effects cannot be identified here.



## A. Did bootstrap deadlock disappear?

Stage1 verifies candidate density, key count, recall and mass match previous_full at every L9–L16 on all three clips. Stage0 full-layer tests verify History is written from all executed patch support afterL0, available startingL1. This establishes semantic eligibility, not the usefulness of every candidate.



### chess: Stage1 filtering (fractions0–1; enrichment is a multiplier)

| layer | previous_density | candidate_density | selected_density | recall | precision | enrichment | mass | filter_keep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.16905 | 0.16905 | 0.09460 | 0.22374 | 0.47261 | 2.36532 | 0.20395 | 0.55959 |
| 10 | 0.16550 | 0.16550 | 0.07855 | 0.14911 | 0.37928 | 1.89822 | 0.15928 | 0.47466 |
| 11 | 0.16856 | 0.16856 | 0.08465 | 0.13201 | 0.31172 | 1.56011 | 0.19659 | 0.50220 |
| 12 | 0.16149 | 0.16149 | 0.07308 | 0.11395 | 0.31152 | 1.55913 | 0.14561 | 0.45254 |
| 13 | 0.16374 | 0.16374 | 0.07693 | 0.11024 | 0.28639 | 1.43333 | 0.13294 | 0.46987 |
| 14 | 0.16003 | 0.16003 | 0.07903 | 0.12544 | 0.31501 | 1.57657 | 0.15924 | 0.49386 |
| 15 | 0.16137 | 0.16137 | 0.06844 | 0.10932 | 0.31858 | 1.59443 | 0.13524 | 0.42412 |
| 16 | 0.17036 | 0.17036 | 0.07915 | 0.10123 | 0.25598 | 1.28112 | 0.10447 | 0.46460 |



### fire: Stage1 filtering (fractions0–1; enrichment is a multiplier)

| layer | previous_density | candidate_density | selected_density | recall | precision | enrichment | mass | filter_keep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.16214 | 0.16214 | 0.09960 | 0.21637 | 0.43423 | 2.17323 | 0.19118 | 0.61427 |
| 10 | 0.15927 | 0.15927 | 0.08282 | 0.15965 | 0.38490 | 1.92635 | 0.15318 | 0.52004 |
| 11 | 0.16569 | 0.16569 | 0.08616 | 0.12504 | 0.29037 | 1.45325 | 0.14048 | 0.51999 |
| 12 | 0.15875 | 0.15875 | 0.08048 | 0.11542 | 0.28702 | 1.43651 | 0.13482 | 0.50693 |
| 13 | 0.15899 | 0.15899 | 0.08310 | 0.12354 | 0.29748 | 1.48884 | 0.13850 | 0.52265 |
| 14 | 0.15502 | 0.15502 | 0.08667 | 0.14067 | 0.32301 | 1.61659 | 0.16753 | 0.55909 |
| 15 | 0.15399 | 0.15399 | 0.07362 | 0.12846 | 0.34760 | 1.73967 | 0.15629 | 0.47806 |
| 16 | 0.16893 | 0.16893 | 0.07949 | 0.10795 | 0.27094 | 1.35601 | 0.10896 | 0.47051 |



### redkitchen: Stage1 filtering (fractions0–1; enrichment is a multiplier)

| layer | previous_density | candidate_density | selected_density | recall | precision | enrichment | mass | filter_keep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.16600 | 0.16600 | 0.09704 | 0.20942 | 0.43130 | 2.15858 | 0.18983 | 0.58458 |
| 10 | 0.15770 | 0.15770 | 0.08204 | 0.15730 | 0.38299 | 1.91678 | 0.16307 | 0.52023 |
| 11 | 0.16084 | 0.16084 | 0.07859 | 0.11410 | 0.29052 | 1.45399 | 0.14309 | 0.48861 |
| 12 | 0.16063 | 0.16063 | 0.06875 | 0.10550 | 0.30449 | 1.52394 | 0.12388 | 0.42800 |
| 13 | 0.16272 | 0.16272 | 0.07666 | 0.11663 | 0.30378 | 1.52035 | 0.13789 | 0.47109 |
| 14 | 0.15888 | 0.15888 | 0.08435 | 0.14219 | 0.33458 | 1.67450 | 0.17621 | 0.53092 |
| 15 | 0.15484 | 0.15484 | 0.07205 | 0.12824 | 0.35224 | 1.76290 | 0.16295 | 0.46531 |
| 16 | 0.16861 | 0.16861 | 0.07822 | 0.11430 | 0.29203 | 1.46158 | 0.12066 | 0.46392 |



### macro: Stage1 filtering (fractions0–1; enrichment is a multiplier)

| layer | previous_density | candidate_density | selected_density | recall | precision | enrichment | mass | filter_keep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.16573 | 0.16573 | 0.09708 | 0.21651 | 0.44604 | 2.23238 | 0.19499 | 0.58577 |
| 10 | 0.16082 | 0.16082 | 0.08114 | 0.15535 | 0.38239 | 1.91378 | 0.15851 | 0.50453 |
| 11 | 0.16503 | 0.16503 | 0.08313 | 0.12372 | 0.29754 | 1.48912 | 0.16005 | 0.50374 |
| 12 | 0.16029 | 0.16029 | 0.07410 | 0.11162 | 0.30101 | 1.50652 | 0.13477 | 0.46230 |
| 13 | 0.16181 | 0.16181 | 0.07890 | 0.11680 | 0.29588 | 1.48084 | 0.13644 | 0.48757 |
| 14 | 0.15798 | 0.15798 | 0.08335 | 0.13610 | 0.32420 | 1.62255 | 0.16766 | 0.52762 |
| 15 | 0.15674 | 0.15674 | 0.07137 | 0.12201 | 0.33947 | 1.69900 | 0.15150 | 0.45535 |
| 16 | 0.16930 | 0.16930 | 0.07895 | 0.10782 | 0.27298 | 1.36624 | 0.11137 | 0.46634 |



## B. Does implicit filtering work?

L10→L15 selected precision 0.38239→0.33947; enrichment 1.91378→1.69900; mass 0.15851→0.15150. This reports endpoints, not an assumption of monotonic progression. Inspect all layers/scenes above.

Precision is selected support membership in current patch Top20; enrichment divides by actual integer-rounded target density. Recall alone cannot establish a good filter. These two-frame targets are on the sparse trajectory, not identical to a Dense trajectory.



## C. How much History is actually selected?

| layer | history_candidates | selected_history | history_fraction_patch | history_fraction_all | filter_keep | rep_overlap | unused_budget | rep | exact | stored_history_mib | actual_unique_density |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 22001.63030 | 12645.76297 | 0.59585 | 0.57839 | 0.57482 | 13551.53237 | 4785.08119 | 8578.15499 | 0.00000 | 511.24053 | 0.16408 |
| 10 | 21223.91825 | 10626.57729 | 0.48501 | 0.47116 | 0.50069 | 12642.01981 | 4093.44664 | 11288.91295 | 0.06254 | 530.68984 | 0.16927 |
| 11 | 21915.55246 | 10660.51102 | 0.50015 | 0.48524 | 0.48654 | 13621.38937 | 4673.32003 | 10675.10896 | 0.05928 | 529.84230 | 0.16492 |
| 12 | 21335.67913 | 9461.15681 | 0.44234 | 0.42888 | 0.44390 | 13463.48884 | 4585.23410 | 11956.17871 | 6.42981 | 537.23535 | 0.16558 |
| 13 | 21423.76479 | 10075.60170 | 0.48440 | 0.46913 | 0.47096 | 14221.68066 | 5165.64830 | 10760.96680 | 6.78239 | 532.89550 | 0.16123 |
| 14 | 20843.35045 | 9769.93959 | 0.47297 | 0.45816 | 0.46928 | 12985.83552 | 5324.08085 | 10726.46456 | 188.51382 | 532.17050 | 0.16004 |
| 15 | 20684.91908 | 9846.27651 | 0.46769 | 0.45300 | 0.47629 | 12653.05539 | 4904.51765 | 11036.07254 | 222.13280 | 539.63162 | 0.16319 |
| 16 | 21104.48158 | 9693.67955 | 0.43826 | 0.42576 | 0.45973 | 12296.20299 | 3876.94821 | 12410.73689 | 27.63449 | 541.66369 | 0.17090 |

All-layer and stage-average tables are in comparison.md and layerwise.csv. Candidate bank size is not selected History fraction. Inclusion/exclusion of mandatory special keys is labeled explicitly.



## D. Does quality improve?

| method | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Dense (historical) | 23.04204 | 84.29269 | 0.14804 | 0.12436 | 0.13620 | 48.05374 |
| SAF3R lazy Top4 (historical) | 23.60271 | 84.43107 | 0.12976 | 0.12427 | 0.12701 | 49.16638 |
| Old reuse+residual (historical) | 22.20601 | 83.82997 | 0.12183 | 0.11292 | 0.11737 | 50.44266 |
| 57bdb78 (previous) | 22.30034 | 84.30616 | 0.13782 | 0.12741 | 0.13262 | 46.80339 |
| All-edge/no-coverage (new) | 22.65127 | 84.43552 | 0.13144 | 0.12824 | 0.12984 | 46.31437 |

New minus57bdb78: {"auc03": 0.3509326959130088, "auc30": 0.12935883014620897, "acc": -0.006377031824606388, "comp": 0.000830474832311634, "cd": -0.002773278496147419, "f1": -0.48901529478881933}. AUC/F1 differences are percentage points; distance differences metres.

Per-scene differences for the requested reconstruction-sensitive scenes:

| scene | auc03 | auc30 | acc | comp | cd | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| redkitchen | 2.07923 | 1.31111 | -0.00039 | 0.00190 | 0.00076 | -1.94229 |
| fire | 1.23031 | -0.12180 | -0.00131 | 0.00125 | -0.00003 | -0.06764 |
| pumpkin | -0.12303 | 0.05659 | -0.03803 | -0.01132 | -0.02467 | -2.81548 |



## E. What is the efficiency cost?

| method | mean_scene_median_s | mean_cold_s | peak_allocated_gib | peak_reserved_gib | speedup |
| --- | --- | --- | --- | --- | --- |
| Dense (historical) | 17.11768 | 17.72123 | 14.35857 | 17.55469 | historical only |
| SAF3R lazy Top4 (historical) | 8.76281 | 13.30167 | 14.61248 | 18.02539 | historical only |
| Old reuse+residual (historical) | 116.74196 | 121.26202 | 15.49932 | 20.28125 | historical only |
| 57bdb78 (previous) | 72.62345 | 77.69574 | 15.69668 | 19.68750 | N/A: no fresh pair |
| All-edge/no-coverage (new) | 131.83629 | 136.40484 | 17.07543 | 22.30469 | N/A: no fresh pair |

| stage | history_candidates | selected_history | history_fraction_patch | history_fraction_all | filter_keep | rep_overlap | unused_budget | rep | exact | stored_history_mib | actual_unique_density |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0-L8 | 19932.15885 | 10288.20241 | 0.46886 | 0.45553 | 0.45853 | 11355.69789 | 3632.21468 | 12088.58198 | 0.00000 | 539.08739 | 0.17274 |
| L9-L16 | 21316.66200 | 10347.43818 | 0.48583 | 0.47122 | 0.48528 | 13179.40062 | 4676.03462 | 10929.07455 | 56.45189 | 531.92117 | 0.16490 |
| L17-L23 | 21493.01240 | 12247.58994 | 0.57922 | 0.56213 | 0.57164 | 12399.45757 | 4761.54001 | 8999.66837 | 0.20089 | 513.90181 | 0.16426 |

All-edge History banks retain a full previous20% patch support per query group; stored_history_mib is recorded explicitly. The unchanged selector still evaluates every Fine exploration item. Higher reuse is not claimed to reduce exploration work.

History/Rep overlap and unused budgets are confounds of independent cost accounting; this experiment deliberately does NOT fix them. Timings across historical phases do not establish a fresh paired speedup.



## Evidence-based next action (not implemented)

Do not increase History blindly. Its bank is healthy but quality/filtering does not establish success; next isolate filtering strength or insufficient coarse evidence with one controlled ablation, keeping Rep/Exact fixed.

No additional production change was implemented. All quality metrics use the same frozen checkpoint/inputs; no thresholds were tuned using7Scenes truth.
