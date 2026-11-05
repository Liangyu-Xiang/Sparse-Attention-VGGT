# Cluster20 equivalent engineering: completed evaluation

All36 new runs complete; 54 method/scene records including reused references. Quality macro gate: False; every-scene paired speed gate: True; every-scene point-cloud change<2%: False.

VGGT; same .20 patch-cluster fraction,10%/22%/10% attention retention,256-query mask sharing,one current-Value local update,cluster_population weights,16 budget steps. No probability/state/objective changes or test-truth tuning. Engineering configuration is frozen in each plan; first-candidate and conservative variants are reported separately. Precision/self-repeat diagnostics are retained in the first campaign.1 warmup+3 measured complete forwards; median per scene, equal-scene macros. RTX5090D,GPU4/5/6,strict same-GPU scene pairs,all initialization/clustering/history/routing/checks/reordering and prediction heads included. Model loading/image decode/geometry evaluation excluded. Any enabled cuDNN first-call search remains in recorded cold forward; warm and cold memory both shown.

End-to-end implementation comparison: common allocator/RoPE/residual improvements are enabled only for the changed method. This is not evidence that the sparse selector itself is faster than SAF3R, or that every optimized SAF3R implementation is slower. SAF3R timing rerun explicitly authorized; quality reused. SAF3R is lazy first-correspondence Top4 per frame, not paper DINO TopK. Dense not rerun: formal original-Dense speedup N/A.7Scenes128-frame .20-cluster reference quality was newly completed in the first campaign and reused in the conservative campaign. ETH3D reference reuses the earlier complete20% batch. Reused reference timings are historical for the conservative candidate, not paired.

| Dataset | Method | Scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Forward s | Warm allocated GiB | All-forward peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | reference | 7 | 22.220074 | 84.322214 | 0.141468 | 0.122515 | 0.131992 | 47.606175 | 7.388654 | 15.498117 | 15.498117 |
| 7scenes | saf3r | 7 | 23.934899 | 84.713102 | 0.143263 | 0.127883 | 0.135573 | 47.315421 | 6.668902 | 14.615167 | 14.615167 |
| 7scenes | optimized | 7 | 22.098214 | 84.216934 | 0.140613 | 0.125862 | 0.133238 | 47.842894 | 6.427234 | 14.979228 | 14.979228 |
| eth3d | reference | 11 | 16.627806 | 74.557086 | 0.368234 | 0.951485 | 0.659859 | 51.417147 | 1.723118 | 11.239786 | 11.239786 |
| eth3d | saf3r | 11 | 23.332202 | 80.174741 | 0.303106 | 0.857720 | 0.580413 | 62.830390 | 1.601425 | 10.686710 | 10.686710 |
| eth3d | optimized | 11 | 16.156547 | 74.397395 | 0.373163 | 0.954378 | 0.663771 | 50.671868 | 1.323959 | 10.958631 | 10.958631 |

AUC/F1 displayed as percentages; JSON/CSV fractions. Distances unsquared metres; CD=(ACC+Comp)/2.7Scenes includes seven existing scene/sequence entries at128frames each, not the full official test split.7Scenes F1 threshold/downsample .05/.0078125m; ETH3D .25/.0390625m; unchanged native unposed alignment/fusion and exact input lists/checkpoint hash audited.

| Dataset | ΔACC % | ΔComp % | ΔCD % | ΔF1 % | Quality <2% | vs SAF3R speedup | All scenes faster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | -0.604255 | +2.732106 | +0.944155 | +0.497244 | False | 1.0376x | True |
| eth3d | +1.338693 | +0.304042 | +0.592735 | -1.449475 | True | 1.2096x | True |

Relative quality changes are absolute-tested (<2%, not2 percentage points); improvements exceeding2% are also flagged under the requested change criterion. Per-scene deviations retained in quality_changes.csv; macro gates do not establish every-scene equivalence. No threshold relaxation or post-quality parameter changes.

## Per-scene quality and timing

| Dataset | Scene | Method | AUC3 % | AUC30 % | ACC m | Comp m | CD m | F1 % | Forward s | SAF3R speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7scenes | heads | reference | 47.42864 | 93.26649 | 0.03426 | 0.03174 | 0.03300 | 84.58878 | 7.244225 | N/A historical |
| 7scenes | heads | saf3r | 50.95965 | 93.86114 | 0.03285 | 0.03014 | 0.03149 | 85.13147 | 6.518585 | 1.0000x |
| 7scenes | heads | optimized | 47.09646 | 93.42356 | 0.03431 | 0.03228 | 0.03330 | 84.14244 | 6.402030 | 1.0182x |
| 7scenes | chess | reference | 20.32480 | 88.42233 | 0.12824 | 0.08702 | 0.10763 | 37.56947 | 7.325407 | N/A historical |
| 7scenes | chess | saf3r | 20.11975 | 88.37557 | 0.14277 | 0.07996 | 0.11137 | 38.55846 | 6.598462 | 1.0000x |
| 7scenes | chess | optimized | 20.48064 | 88.44529 | 0.12405 | 0.08109 | 0.10257 | 42.44671 | 6.470275 | 1.0198x |
| 7scenes | stairs | reference | 6.76673 | 80.71973 | 0.22543 | 0.45446 | 0.33995 | 14.72257 | 7.223891 | N/A historical |
| 7scenes | stairs | saf3r | 8.28002 | 83.41166 | 0.19927 | 0.48603 | 0.34265 | 16.39346 | 6.517752 | 1.0000x |
| 7scenes | stairs | optimized | 5.90551 | 80.04880 | 0.22625 | 0.48039 | 0.35332 | 13.93535 | 6.402370 | 1.0180x |
| 7scenes | fire | reference | 50.87762 | 93.75533 | 0.04982 | 0.05332 | 0.05157 | 66.74227 | 7.710550 | N/A historical |
| 7scenes | fire | saf3r | 56.27871 | 94.56324 | 0.04882 | 0.05465 | 0.05174 | 67.53076 | 6.679310 | 1.0000x |
| 7scenes | fire | optimized | 50.96785 | 93.76517 | 0.04976 | 0.05333 | 0.05155 | 66.75383 | 6.405714 | 1.0427x |
| 7scenes | office | reference | 13.13156 | 77.32858 | 0.04676 | 0.05315 | 0.04995 | 59.99717 | 7.378536 | N/A historical |
| 7scenes | office | saf3r | 13.41864 | 75.87434 | 0.04810 | 0.05682 | 0.05246 | 58.55540 | 6.725352 | 1.0000x |
| 7scenes | office | optimized | 13.07825 | 76.92339 | 0.04690 | 0.05192 | 0.04941 | 61.44773 | 6.414594 | 1.0484x |
| 7scenes | pumpkin | reference | 8.91568 | 77.24532 | 0.42531 | 0.09597 | 0.26064 | 21.11615 | 7.362162 | N/A historical |
| 7scenes | pumpkin | saf3r | 10.00246 | 76.82210 | 0.45042 | 0.10118 | 0.27580 | 22.85502 | 6.821169 | 1.0000x |
| 7scenes | pumpkin | optimized | 9.14124 | 77.39214 | 0.42414 | 0.09891 | 0.26152 | 18.07450 | 6.492439 | 1.0506x |
| 7scenes | redkitchen | reference | 8.09547 | 79.51772 | 0.08045 | 0.08194 | 0.08120 | 48.50681 | 7.475810 | N/A historical |
| 7scenes | redkitchen | saf3r | 8.48507 | 80.08366 | 0.08061 | 0.08641 | 0.08351 | 42.18338 | 6.821685 | 1.0000x |
| 7scenes | redkitchen | optimized | 8.01755 | 79.52018 | 0.07888 | 0.08312 | 0.08100 | 48.09970 | 6.403219 | 1.0654x |
| eth3d | courtyard | reference | 11.52205 | 79.93362 | 0.47845 | 0.26823 | 0.37334 | 39.04933 | 1.836724 | N/A historical |
| eth3d | courtyard | saf3r | 19.20341 | 84.66572 | 0.19029 | 0.14647 | 0.16838 | 75.16366 | 1.663936 | 1.0000x |
| eth3d | courtyard | optimized | 11.52205 | 79.93362 | 0.47845 | 0.26823 | 0.37334 | 39.04933 | 1.447620 | 1.1494x |
| eth3d | delivery_area | reference | 11.15385 | 78.74786 | 0.70318 | 1.50668 | 1.10493 | 12.43781 | 2.030451 | N/A historical |
| eth3d | delivery_area | saf3r | 15.00000 | 80.32906 | 0.51114 | 1.38272 | 0.94693 | 36.40611 | 1.864382 | 1.0000x |
| eth3d | delivery_area | optimized | 11.15385 | 78.74786 | 0.70445 | 1.50952 | 1.10699 | 12.39813 | 1.509069 | 1.2355x |
| eth3d | electro | reference | 33.69321 | 92.04678 | 0.22437 | 5.13307 | 2.67872 | 61.21236 | 1.975892 | N/A historical |
| eth3d | electro | saf3r | 25.14620 | 90.66577 | 0.35952 | 5.25873 | 2.80913 | 39.00947 | 1.805762 | 1.0000x |
| eth3d | electro | optimized | 30.13945 | 92.09177 | 0.24169 | 5.13490 | 2.68830 | 57.69583 | 1.473459 | 1.2255x |
| eth3d | facade | reference | 22.99415 | 87.19532 | 0.28288 | 0.27918 | 0.28103 | 58.54067 | 3.666490 | N/A historical |
| eth3d | facade | saf3r | 32.54971 | 89.39415 | 0.29911 | 0.27805 | 0.28858 | 59.01804 | 3.479583 | 1.0000x |
| eth3d | facade | optimized | 22.99415 | 87.19532 | 0.28288 | 0.27918 | 0.28103 | 58.54067 | 2.977992 | 1.1684x |
| eth3d | kicker | reference | 38.13620 | 92.69534 | 0.07123 | 0.06012 | 0.06567 | 99.09432 | 1.487486 | N/A historical |
| eth3d | kicker | saf3r | 51.32616 | 94.80287 | 0.06197 | 0.05663 | 0.05930 | 99.48749 | 1.526612 | 1.0000x |
| eth3d | kicker | optimized | 38.13620 | 92.69534 | 0.07123 | 0.06012 | 0.06567 | 99.09432 | 1.182545 | 1.2910x |
| eth3d | office | reference | 22.66667 | 74.68718 | 0.07485 | 0.07319 | 0.07402 | 98.85790 | 1.330293 | N/A historical |
| eth3d | office | saf3r | 25.94872 | 76.92308 | 0.09760 | 0.10011 | 0.09886 | 98.81638 | 1.279485 | 1.0000x |
| eth3d | office | optimized | 22.66667 | 74.68718 | 0.07485 | 0.07319 | 0.07402 | 98.85790 | 1.010062 | 1.2667x |
| eth3d | pipes | reference | 18.31502 | 83.22344 | 0.21389 | 0.24350 | 0.22870 | 77.21791 | 0.805345 | N/A historical |
| eth3d | pipes | saf3r | 36.63004 | 92.82051 | 0.22500 | 0.21178 | 0.21839 | 81.63477 | 0.944012 | 1.0000x |
| eth3d | pipes | optimized | 18.31502 | 83.22344 | 0.21389 | 0.24350 | 0.22870 | 77.21791 | 0.577462 | 1.6348x |
| eth3d | playground | reference | 7.32527 | 65.18145 | 0.41842 | 0.98027 | 0.69934 | 32.57769 | 1.607205 | N/A historical |
| eth3d | playground | saf3r | 15.25538 | 72.97715 | 0.28745 | 0.69320 | 0.49033 | 70.11664 | 1.484975 | 1.0000x |
| eth3d | playground | optimized | 7.32527 | 65.18145 | 0.41842 | 0.98027 | 0.69934 | 32.57769 | 1.197500 | 1.2401x |
| eth3d | relief | reference | 8.77193 | 72.12476 | 0.72627 | 0.83986 | 0.78307 | 2.18094 | 1.240120 | N/A historical |
| eth3d | relief | saf3r | 21.83236 | 83.35283 | 0.53095 | 0.53522 | 0.53309 | 28.85149 | 0.919042 | 1.0000x |
| eth3d | relief | optimized | 8.18713 | 71.92982 | 0.72799 | 0.83286 | 0.78043 | 1.97475 | 0.784907 | 1.1709x |
| eth3d | relief_2 | reference | 3.33333 | 42.59649 | 0.47596 | 0.68359 | 0.57977 | 21.99526 | 1.045511 | N/A historical |
| eth3d | relief_2 | saf3r | 5.43860 | 50.54386 | 0.38966 | 0.53463 | 0.46214 | 37.35554 | 0.939159 | 1.0000x |
| eth3d | relief_2 | optimized | 3.33333 | 42.59649 | 0.47596 | 0.68359 | 0.57977 | 21.99526 | 0.818792 | 1.1470x |
| eth3d | terrains | reference | 4.99419 | 51.69570 | 0.38107 | 0.39865 | 0.38986 | 62.42444 | 1.928776 | N/A historical |
| eth3d | terrains | saf3r | 8.32365 | 65.44715 | 0.38147 | 0.23737 | 0.30942 | 65.27471 | 1.708728 | 1.0000x |
| eth3d | terrains | optimized | 3.94890 | 50.08904 | 0.41498 | 0.43279 | 0.42389 | 57.98877 | 1.584146 | 1.0786x |
