# Compact acceleration of the complete combined revision: 7Scenes 128-frame evaluation

Same checkpoint, exact per-scene uniform128-frame list, preprocessing, pose and recon_unposed protocol.
AUC/F1 in percent, ACC/Comp/CD in metres; CD=(ACC+Comp)/2, unsquared.
Reconstruction threshold0.05m, downsampling0.0078125m. No parameters tuned to7Scenes truth.
Only the compact implementation is newly evaluated; every prior method is reused. No staged algorithm ablations. Original and previous cosine quality results are reused from the earlier completed cohort; their timings are historical. Dense/SAF3R and the unoptimized combined control are reused from the completed cohort. Timing ratios against those references use stored medians, NOT fresh contemporaneous control timings. See reference_provenance.json.
Combined changes: Drop=pi and Rep-relative gains; all executed patch edges provide history; fused compact raw-logit residual history calibrated within current canonical grids; estimated mass transfer retained as previously requested. See docs/experiments/reuse_rep_revision.md. Second-QK mass summaries are retained and counted; the combined revision adds no extra QK pass and stores no full logit tensor.
Budget20% is an upper bound; indivisible state choices may leave unused budget, recorded in layers.json.

| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm median s | Peak allocated GiB | Ratio vs reused Dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | dense | 51.5789 | 93.9944 | 0.0343 | 0.0308 | 0.0326 | 84.9976 | 12.216 | 14.360 | 1.000x |
| heads | saf3r_dino64 | 51.7470 | 93.9477 | 0.0331 | 0.0300 | 0.0316 | 85.3997 | 6.998 | 18.912 | 1.746x |
| heads | original | 44.6399 | 92.9536 | 0.0361 | 0.0329 | 0.0345 | 82.4135 | 81.806 | 15.496 | 0.149x |
| heads | cosine_distortion | 11.8274 | 81.8090 | 0.1282 | 0.0439 | 0.0861 | 41.6218 | 81.003 | 15.502 | 0.150x |
| heads | rep_combined | 28.2480 | 88.1447 | 0.0408 | 0.0382 | 0.0395 | 78.2750 | 105.681 | 15.903 | 0.116x |
| heads | rep_compact | 30.0935 | 88.8579 | 0.0418 | 0.0380 | 0.0399 | 76.3746 | 35.156 | 15.905 | 0.347x |
| chess | dense | 20.5135 | 88.4658 | 0.1399 | 0.0877 | 0.1138 | 35.9524 | 12.252 | 14.360 | 1.000x |
| chess | saf3r_dino64 | 20.0213 | 88.3095 | 0.1519 | 0.0845 | 0.1182 | 36.2923 | 7.013 | 18.912 | 1.747x |
| chess | original | 22.1826 | 88.9399 | 0.1368 | 0.0849 | 0.1109 | 37.3951 | 81.186 | 15.497 | 0.151x |
| chess | cosine_distortion | 10.1173 | 77.3897 | 0.2418 | 0.1209 | 0.1814 | 12.8323 | 79.166 | 15.499 | 0.154x |
| chess | rep_combined | 12.6640 | 84.0600 | 0.1341 | 0.0862 | 0.1102 | 31.9654 | 109.112 | 15.903 | 0.112x |
| chess | rep_compact | 12.3114 | 84.0137 | 0.1288 | 0.0839 | 0.1063 | 35.2589 | 35.102 | 15.903 | 0.349x |
| stairs | dense | 4.9172 | 80.9535 | 0.2208 | 0.4797 | 0.3502 | 15.3143 | 12.172 | 14.360 | 1.000x |
| stairs | saf3r_dino64 | 9.0223 | 83.3629 | 0.2002 | 0.4658 | 0.3330 | 15.5669 | 6.989 | 18.912 | 1.742x |
| stairs | original | 5.8809 | 77.2351 | 0.2251 | 0.4373 | 0.3312 | 14.3102 | 80.928 | 15.497 | 0.151x |
| stairs | cosine_distortion | 4.1462 | 70.7304 | 0.8837 | 0.2895 | 0.5866 | 7.1804 | 79.668 | 15.494 | 0.153x |
| stairs | rep_combined | 6.5453 | 78.6622 | 0.2110 | 0.4881 | 0.3496 | 15.6457 | 105.957 | 15.901 | 0.115x |
| stairs | rep_compact | 6.1926 | 78.5109 | 0.2178 | 0.4877 | 0.3527 | 15.0082 | 34.962 | 15.904 | 0.348x |
| fire | dense | 55.3150 | 94.5579 | 0.0489 | 0.0513 | 0.0501 | 66.7010 | 12.370 | 14.360 | 1.000x |
| fire | saf3r_dino64 | 55.7497 | 94.5128 | 0.0512 | 0.0566 | 0.0539 | 65.2743 | 7.099 | 18.912 | 1.742x |
| fire | original | 54.3225 | 93.7344 | 0.0402 | 0.0469 | 0.0436 | 75.8949 | 82.374 | 15.497 | 0.150x |
| fire | cosine_distortion | 15.0016 | 83.9042 | 0.1674 | 0.0844 | 0.1259 | 29.5292 | 81.359 | 15.501 | 0.152x |
| fire | rep_combined | 39.7884 | 91.8676 | 0.0411 | 0.0452 | 0.0432 | 70.8812 | 118.891 | 15.902 | 0.104x |
| fire | rep_compact | 44.6973 | 92.5697 | 0.0385 | 0.0424 | 0.0404 | 73.8699 | 35.680 | 15.904 | 0.347x |
| office | dense | 14.2758 | 77.1457 | 0.0471 | 0.0521 | 0.0496 | 61.0877 | 12.274 | 14.360 | 1.000x |
| office | saf3r_dino64 | 13.0823 | 75.4655 | 0.0475 | 0.0567 | 0.0521 | 59.4168 | 7.005 | 18.912 | 1.752x |
| office | original | 14.6120 | 77.9417 | 0.0535 | 0.0540 | 0.0538 | 55.1339 | 81.224 | 15.499 | 0.151x |
| office | cosine_distortion | 2.8543 | 70.9961 | 0.2325 | 0.0958 | 0.1642 | 18.5815 | 80.934 | 15.501 | 0.151x |
| office | rep_combined | 13.3694 | 74.3635 | 0.0469 | 0.0499 | 0.0484 | 63.5229 | 117.312 | 15.902 | 0.105x |
| office | rep_compact | 12.0899 | 74.9069 | 0.0492 | 0.0492 | 0.0492 | 60.3457 | 35.283 | 15.903 | 0.348x |
| pumpkin | dense | 10.6176 | 77.5980 | 0.5057 | 0.0965 | 0.3011 | 25.0739 | 12.435 | 14.360 | 1.000x |
| pumpkin | saf3r_dino64 | 10.1665 | 77.0107 | 0.4357 | 0.1039 | 0.2698 | 21.7118 | 7.114 | 18.912 | 1.748x |
| pumpkin | original | 10.1378 | 77.1805 | 0.3256 | 0.0877 | 0.2067 | 28.7542 | 82.450 | 15.498 | 0.151x |
| pumpkin | cosine_distortion | 4.2405 | 68.1976 | 0.4250 | 0.1241 | 0.2746 | 6.8789 | 81.189 | 15.497 | 0.153x |
| pumpkin | rep_combined | 6.4592 | 75.1267 | 0.3819 | 0.0872 | 0.2346 | 23.6341 | 109.645 | 15.901 | 0.113x |
| pumpkin | rep_compact | 6.0655 | 74.5796 | 0.4327 | 0.0911 | 0.2619 | 21.9025 | 35.764 | 15.903 | 0.348x |
| redkitchen | dense | 8.4646 | 79.7568 | 0.0795 | 0.0812 | 0.0803 | 46.9624 | 12.306 | 14.360 | 1.000x |
| redkitchen | saf3r_dino64 | 8.6450 | 80.1817 | 0.0788 | 0.0886 | 0.0837 | 42.4679 | 7.076 | 18.912 | 1.739x |
| redkitchen | original | 10.9621 | 81.9468 | 0.0704 | 0.0618 | 0.0661 | 57.9058 | 81.614 | 15.498 | 0.151x |
| redkitchen | cosine_distortion | 7.9929 | 75.8600 | 0.2461 | 0.1131 | 0.1796 | 16.3523 | 79.115 | 15.498 | 0.155x |
| redkitchen | rep_combined | 3.9780 | 73.6225 | 0.0864 | 0.0787 | 0.0825 | 42.6319 | 106.426 | 15.902 | 0.116x |
| redkitchen | rep_compact | 4.7449 | 74.7449 | 0.0837 | 0.0723 | 0.0780 | 50.9192 | 35.396 | 15.902 | 0.348x |

## Equal-scene macro averages (only when all seven quality evaluations finish)

| Method | Complete scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 23.6689 | 84.6389 | 0.1537 | 0.1256 | 0.1397 | 48.0128 | 1.000x |
| saf3r_dino64 | 7/7 | 24.0620 | 84.6844 | 0.1426 | 0.1266 | 0.1346 | 46.5900 | 1.745x |
| original | 7/7 | 23.2483 | 84.2760 | 0.1268 | 0.1151 | 0.1209 | 50.2582 | N/A (historical) |
| cosine_distortion | 7/7 | 8.0258 | 75.5553 | 0.3321 | 0.1245 | 0.2283 | 18.9966 | N/A (historical) |
| rep_combined | 7/7 | 15.8646 | 80.8353 | 0.1346 | 0.1248 | 0.1297 | 46.6509 | N/A (historical) |
| rep_compact | 7/7 | 16.5993 | 81.1691 | 0.1418 | 0.1235 | 0.1326 | 47.6684 | 0.348x |

¹ Ratio of equal-scene mean warmed median full-forward times against stored Dense medians; reused controls are not contemporaneous remeasurements. Below1x is slower.
Quality, speed and memory gates are separate. Depth AbsRel is not evaluated here, so complete historical acceptance cannot be claimed.
Missing/OOM/interrupted runs retain six N/A fields with a concrete reason; no silent frame/budget reductions.
Per-method input/config/source snapshots and raw metrics are stored under scene/method/.

## Quality differences against reused original results

AUC/F1 differences in percentage points; distance differences in metres. Runtime ratio = original / variant; >1 means faster.

| Variant | dAUC@3 pp | dAUC@30 pp | dACC m | dComp m | dCD m | dF1 pp | Original/variant time |
|---|---:|---:|---:|---:|---:|---:|---:|
| saf3r_dino64 | +0.813765 | +0.408406 | +0.015809 | +0.011533 | +0.013671 | -3.668269 | N/A (historical reference) |
| cosine_distortion | -15.222511 | -8.720707 | +0.205293 | +0.009464 | +0.107379 | -31.261593 | N/A (historical reference) |
| rep_combined | -7.383647 | -3.440664 | +0.007793 | +0.009724 | +0.008758 | -3.607343 | N/A (historical reference) |
| rep_compact | -6.648974 | -3.106897 | +0.014943 | +0.008448 | +0.011695 | -2.589802 | N/A (historical reference) |
