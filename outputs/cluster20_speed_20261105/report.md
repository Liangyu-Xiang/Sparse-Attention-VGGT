# Cluster20 equivalent engineering: completed evaluation

All43 new runs complete; 54 method/scene records including reused references. Quality macro gate: False; every-scene paired speed gate: True; every-scene point-cloud change<2%: False.

VGGT; same .20 patch-cluster fraction,10%/22%/10% attention retention,256-query mask sharing,one current-Value local update,cluster_population weights,16 budget steps. No probability/state/objective changes or test-truth tuning. Engineering configuration is frozen in each plan; first-candidate and conservative variants are reported separately. Precision/self-repeat diagnostics are retained in the first campaign.1 warmup+3 measured complete forwards; median per scene, equal-scene macros. RTX5090D,GPU4/5/6,strict same-GPU scene pairs,all initialization/clustering/history/routing/checks/reordering and prediction heads included. Model loading/image decode/geometry evaluation excluded. Any enabled cuDNN first-call search remains in recorded cold forward; warm and cold memory both shown.

End-to-end implementation comparison: common allocator/RoPE/residual improvements are enabled only for the changed method. This is not evidence that the sparse selector itself is faster than SAF3R, or that every optimized SAF3R implementation is slower. SAF3R timing rerun explicitly authorized; quality reused. SAF3R is lazy first-correspondence Top4 per frame, not paper DINO TopK. Dense not rerun: formal original-Dense speedup N/A.7Scenes128-frame .20-cluster reference quality was newly completed in the first campaign and reused in the conservative campaign. ETH3D reference reuses the earlier complete20% batch. Reused reference timings are historical for the conservative candidate, not paired.

| Dataset | Method | Scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Forward s | Warm allocated GiB | All-forward peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | reference | 7 | 22.220074 | 84.322214 | 0.141468 | 0.122515 | 0.131992 | 47.606175 | 7.388654 | 15.498117 | 15.498117 |
| 7scenes | saf3r | 7 | 23.934899 | 84.713102 | 0.143263 | 0.127883 | 0.135573 | 47.315421 | 6.779607 | 14.615167 | 14.615167 |
| 7scenes | optimized | 7 | 23.145154 | 84.450459 | 0.139658 | 0.124505 | 0.132081 | 48.711787 | 6.161170 | 14.978006 | 20.328788 |
| eth3d | reference | 11 | 16.627806 | 74.557086 | 0.368234 | 0.951485 | 0.659859 | 51.417147 | 1.723118 | 11.239786 | 11.239786 |
| eth3d | saf3r | 11 | 23.332202 | 80.174741 | 0.303106 | 0.857720 | 0.580413 | 62.830390 | 1.579174 | 10.686710 | 10.686710 |
| eth3d | optimized | 11 | 16.622792 | 74.561327 | 0.367687 | 0.951490 | 0.659588 | 51.394765 | 1.272822 | 11.071013 | 26.253484 |

AUC/F1 displayed as percentages; JSON/CSV fractions. Distances unsquared metres; CD=(ACC+Comp)/2.7Scenes includes seven existing scene/sequence entries at128frames each, not the full official test split.7Scenes F1 threshold/downsample .05/.0078125m; ETH3D .25/.0390625m; unchanged native unposed alignment/fusion and exact input lists/checkpoint hash audited.

| Dataset | ΔACC % | ΔComp % | ΔCD % | ΔF1 % | Quality <2% | vs SAF3R speedup | All scenes faster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | -1.279881 | +1.624546 | +0.068068 | +2.322412 | False | 1.1004x | True |
| eth3d | -0.148522 | +0.000476 | -0.041098 | -0.043531 | True | 1.2407x | True |

Relative quality changes are absolute-tested (<2%, not2 percentage points); improvements exceeding2% are also flagged under the requested change criterion. Per-scene deviations retained in quality_changes.csv; macro gates do not establish every-scene equivalence. No threshold relaxation or post-quality parameter changes.

## Per-scene quality and timing

| Dataset | Scene | Method | AUC3 % | AUC30 % | ACC m | Comp m | CD m | F1 % | Forward s | SAF3R speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | heads | reference | 47.42864 | 93.26649 | 0.03426 | 0.03174 | 0.03300 | 84.58878 | 7.244225 | 0.9181x |
| 7scenes | heads | saf3r | 50.95965 | 93.86114 | 0.03285 | 0.03014 | 0.03149 | 85.13147 | 6.650894 | 1.0000x |
| 7scenes | heads | optimized | 50.39780 | 93.78814 | 0.03380 | 0.03130 | 0.03255 | 84.59849 | 6.152163 | 1.0811x |
| 7scenes | chess | reference | 20.32480 | 88.42233 | 0.12824 | 0.08702 | 0.10763 | 37.56947 | 7.325407 | 0.9334x |
| 7scenes | chess | saf3r | 20.11975 | 88.37557 | 0.14277 | 0.07996 | 0.11137 | 38.55846 | 6.837556 | 1.0000x |
| 7scenes | chess | optimized | 20.08694 | 88.29601 | 0.11656 | 0.07662 | 0.09659 | 47.43124 | 6.177847 | 1.1068x |
| 7scenes | stairs | reference | 6.76673 | 80.71973 | 0.22543 | 0.45446 | 0.33995 | 14.72257 | 7.223891 | 0.9391x |
| 7scenes | stairs | saf3r | 8.28002 | 83.41166 | 0.19927 | 0.48603 | 0.34265 | 16.39346 | 6.783718 | 1.0000x |
| 7scenes | stairs | optimized | 8.83776 | 81.44972 | 0.22333 | 0.47251 | 0.34792 | 14.83136 | 6.171662 | 1.0992x |
| 7scenes | fire | reference | 50.87762 | 93.75533 | 0.04982 | 0.05332 | 0.05157 | 66.74227 | 7.710550 | 0.8724x |
| 7scenes | fire | saf3r | 56.27871 | 94.56324 | 0.04882 | 0.05465 | 0.05174 | 67.53076 | 6.726567 | 1.0000x |
| 7scenes | fire | optimized | 52.50164 | 93.91527 | 0.05208 | 0.05614 | 0.05411 | 65.92454 | 6.176132 | 1.0891x |
| 7scenes | office | reference | 13.13156 | 77.32858 | 0.04676 | 0.05315 | 0.04995 | 59.99717 | 7.378536 | 0.9233x |
| 7scenes | office | saf3r | 13.41864 | 75.87434 | 0.04810 | 0.05682 | 0.05246 | 58.55540 | 6.812635 | 1.0000x |
| 7scenes | office | optimized | 13.08645 | 76.87705 | 0.04666 | 0.05246 | 0.04956 | 60.79606 | 6.164907 | 1.1051x |
| 7scenes | pumpkin | reference | 8.91568 | 77.24532 | 0.42531 | 0.09597 | 0.26064 | 21.11615 | 7.362162 | 0.9358x |
| 7scenes | pumpkin | saf3r | 10.00246 | 76.82210 | 0.45042 | 0.10118 | 0.27580 | 22.85502 | 6.889601 | 1.0000x |
| 7scenes | pumpkin | optimized | 9.27657 | 77.46596 | 0.42650 | 0.09635 | 0.26142 | 19.93309 | 6.131515 | 1.1236x |
| 7scenes | redkitchen | reference | 8.09547 | 79.51772 | 0.08045 | 0.08194 | 0.08120 | 48.50681 | 7.475810 | 0.9038x |
| 7scenes | redkitchen | saf3r | 8.48507 | 80.08366 | 0.08061 | 0.08641 | 0.08351 | 42.18338 | 6.756278 | 1.0000x |
| 7scenes | redkitchen | optimized | 7.82890 | 79.36106 | 0.07868 | 0.08616 | 0.08242 | 47.46772 | 6.153960 | 1.0979x |
| eth3d | courtyard | reference | 11.52205 | 79.93362 | 0.47845 | 0.26823 | 0.37334 | 39.04933 | 1.836724 | N/A historical |
| eth3d | courtyard | saf3r | 19.20341 | 84.66572 | 0.19029 | 0.14647 | 0.16838 | 75.16366 | 1.750047 | 1.0000x |
| eth3d | courtyard | optimized | 11.56946 | 79.99052 | 0.47467 | 0.26938 | 0.37203 | 38.73689 | 1.393504 | 1.2559x |
| eth3d | delivery_area | reference | 11.15385 | 78.74786 | 0.70318 | 1.50668 | 1.10493 | 12.43781 | 2.030451 | N/A historical |
| eth3d | delivery_area | saf3r | 15.00000 | 80.32906 | 0.51114 | 1.38272 | 0.94693 | 36.40611 | 1.951507 | 1.0000x |
| eth3d | delivery_area | optimized | 11.15385 | 78.74786 | 0.70419 | 1.50715 | 1.10567 | 12.39152 | 1.445873 | 1.3497x |
| eth3d | electro | reference | 33.69321 | 92.04678 | 0.22437 | 5.13307 | 2.67872 | 61.21236 | 1.975892 | N/A historical |
| eth3d | electro | saf3r | 25.14620 | 90.66577 | 0.35952 | 5.25873 | 2.80913 | 39.00947 | 1.841901 | 1.0000x |
| eth3d | electro | optimized | 33.69321 | 92.04678 | 0.22539 | 5.13325 | 2.67932 | 61.16392 | 1.431129 | 1.2870x |
| eth3d | facade | reference | 22.99415 | 87.19532 | 0.28288 | 0.27918 | 0.28103 | 58.54067 | 3.666490 | N/A historical |
| eth3d | facade | saf3r | 32.54971 | 89.39415 | 0.29911 | 0.27805 | 0.28858 | 59.01804 | 3.289160 | 1.0000x |
| eth3d | facade | optimized | 22.99415 | 87.19532 | 0.28271 | 0.27880 | 0.28075 | 58.53209 | 2.855583 | 1.1518x |
| eth3d | kicker | reference | 38.13620 | 92.69534 | 0.07123 | 0.06012 | 0.06567 | 99.09432 | 1.487486 | N/A historical |
| eth3d | kicker | saf3r | 51.32616 | 94.80287 | 0.06197 | 0.05663 | 0.05930 | 99.48749 | 1.488080 | 1.0000x |
| eth3d | kicker | optimized | 38.13620 | 92.69534 | 0.07122 | 0.06012 | 0.06567 | 99.09548 | 1.128679 | 1.3184x |
| eth3d | office | reference | 22.66667 | 74.68718 | 0.07485 | 0.07319 | 0.07402 | 98.85790 | 1.330293 | N/A historical |
| eth3d | office | saf3r | 25.94872 | 76.92308 | 0.09760 | 0.10011 | 0.09886 | 98.81638 | 1.254930 | 1.0000x |
| eth3d | office | optimized | 22.56410 | 74.67692 | 0.07473 | 0.07184 | 0.07329 | 98.84013 | 0.965459 | 1.2998x |
| eth3d | pipes | reference | 18.31502 | 83.22344 | 0.21389 | 0.24350 | 0.22870 | 77.21791 | 0.805345 | N/A historical |
| eth3d | pipes | saf3r | 36.63004 | 92.82051 | 0.22500 | 0.21178 | 0.21839 | 81.63477 | 0.647956 | 1.0000x |
| eth3d | pipes | optimized | 18.31502 | 83.22344 | 0.21366 | 0.24350 | 0.22858 | 77.22576 | 0.557832 | 1.1616x |
| eth3d | playground | reference | 7.32527 | 65.18145 | 0.41842 | 0.98027 | 0.69934 | 32.57769 | 1.607205 | N/A historical |
| eth3d | playground | saf3r | 15.25538 | 72.97715 | 0.28745 | 0.69320 | 0.49033 | 70.11664 | 1.548318 | 1.0000x |
| eth3d | playground | optimized | 7.32527 | 65.18145 | 0.41847 | 0.98026 | 0.69936 | 32.58757 | 1.167611 | 1.3261x |
| eth3d | relief | reference | 8.77193 | 72.12476 | 0.72627 | 0.83986 | 0.78307 | 2.18094 | 1.240120 | N/A historical |
| eth3d | relief | saf3r | 21.83236 | 83.35283 | 0.53095 | 0.53522 | 0.53309 | 28.85149 | 0.809389 | 1.0000x |
| eth3d | relief | optimized | 8.77193 | 72.12476 | 0.72622 | 0.83987 | 0.78304 | 2.18217 | 0.751885 | 1.0765x |
| eth3d | relief_2 | reference | 3.33333 | 42.59649 | 0.47596 | 0.68359 | 0.57977 | 21.99526 | 1.045511 | N/A historical |
| eth3d | relief_2 | saf3r | 5.43860 | 50.54386 | 0.38966 | 0.53463 | 0.46214 | 37.35554 | 0.993635 | 1.0000x |
| eth3d | relief_2 | optimized | 3.33333 | 42.59649 | 0.47221 | 0.68360 | 0.57791 | 22.15601 | 0.787580 | 1.2616x |
| eth3d | terrains | reference | 4.99419 | 51.69570 | 0.38107 | 0.39865 | 0.38986 | 62.42444 | 1.928776 | N/A historical |
| eth3d | terrains | saf3r | 8.32365 | 65.44715 | 0.38147 | 0.23737 | 0.30942 | 65.27471 | 1.795987 | 1.0000x |
| eth3d | terrains | optimized | 4.99419 | 51.69570 | 0.38109 | 0.39862 | 0.38985 | 62.43088 | 1.515909 | 1.1848x |
