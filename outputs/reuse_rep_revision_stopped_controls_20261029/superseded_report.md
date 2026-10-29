# Rep-baseline revision: matched 7Scenes 128-frame staged ablations

Same checkpoint, exact per-scene uniform128-frame list, preprocessing, pose and recon_unposed protocol.
AUC/F1 in percent, ACC/Comp/CD in metres; CD=(ACC+Comp)/2, unsquared.
Reconstruction threshold0.05m, downsampling0.0078125m. No parameters tuned to7Scenes truth.
Fresh paired Dense, SAF3R DINO per-frame Top64 (ETH3D calibrated profile), original output proxy, and new history-conditioned tokenwise cosine distortion with corresponding estimated-mass transfer.
Stages: A Drop=pi and Rep-relative gains; B all executed patch edges provide history; C fused compact raw-logit residual history calibrated within current canonical grids; D original population execution instead of estimated mass transfer. See docs/experiments/reuse_rep_revision.md. Second-QK mass summaries are retained and counted in every stage; C adds no extra QK pass and stores no full logit tensor.
Budget20% is an upper bound; indivisible state choices may leave unused budget, recorded in layers.json.

| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm median s | Peak allocated GiB | Speedup vs Dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | dense | 51.5789 | 93.9944 | 0.0343 | 0.0308 | 0.0326 | 84.9976 | 12.216 | 14.360 | 1.000x |
| heads | saf3r_dino64 | 51.7470 | 93.9477 | 0.0331 | 0.0300 | 0.0316 | 85.3997 | 6.998 | 18.912 | 1.746x |
| heads | original | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| heads | cosine_distortion | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| heads | rep_a | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| heads | rep_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| heads | rep_c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| heads | rep_d | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chess | dense | 20.5135 | 88.4658 | 0.1399 | 0.0877 | 0.1138 | 35.9524 | 12.252 | 14.360 | 1.000x |
| chess | saf3r_dino64 | 20.0213 | 88.3095 | 0.1519 | 0.0845 | 0.1182 | 36.2923 | 7.013 | 18.912 | 1.747x |
| chess | original | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chess | cosine_distortion | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chess | rep_a | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chess | rep_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chess | rep_c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| chess | rep_d | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| stairs | dense | 4.9172 | 80.9535 | 0.2208 | 0.4797 | 0.3502 | 15.3143 | 12.172 | 14.360 | 1.000x |
| stairs | saf3r_dino64 | 9.0223 | 83.3629 | 0.2002 | 0.4658 | 0.3330 | 15.5669 | 6.989 | 18.912 | 1.742x |
| stairs | original | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| stairs | cosine_distortion | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| stairs | rep_a | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| stairs | rep_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| stairs | rep_c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| stairs | rep_d | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| fire | dense | 55.3150 | 94.5579 | 0.0489 | 0.0513 | 0.0501 | 66.7010 | 12.370 | 14.360 | 1.000x |
| fire | saf3r_dino64 | 55.7497 | 94.5128 | 0.0512 | 0.0566 | 0.0539 | 65.2743 | 7.099 | 18.912 | 1.742x |
| fire | original | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| fire | cosine_distortion | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| fire | rep_a | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| fire | rep_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| fire | rep_c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| fire | rep_d | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| office | dense | 14.2758 | 77.1457 | 0.0471 | 0.0521 | 0.0496 | 61.0877 | 12.274 | 14.360 | 1.000x |
| office | saf3r_dino64 | 13.0823 | 75.4655 | 0.0475 | 0.0567 | 0.0521 | 59.4168 | 7.005 | 18.912 | 1.752x |
| office | original | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| office | cosine_distortion | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| office | rep_a | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| office | rep_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| office | rep_c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| office | rep_d | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| pumpkin | dense | 10.6176 | 77.5980 | 0.5057 | 0.0965 | 0.3011 | 25.0739 | 12.435 | 14.360 | 1.000x |
| pumpkin | saf3r_dino64 | 10.1665 | 77.0107 | 0.4357 | 0.1039 | 0.2698 | 21.7118 | 7.114 | 18.912 | 1.748x |
| pumpkin | original | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| pumpkin | cosine_distortion | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| pumpkin | rep_a | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| pumpkin | rep_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| pumpkin | rep_c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| pumpkin | rep_d | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| redkitchen | dense | 8.4646 | 79.7568 | 0.0795 | 0.0812 | 0.0803 | 46.9624 | 12.306 | 14.360 | 1.000x |
| redkitchen | saf3r_dino64 | 8.6450 | 80.1817 | 0.0788 | 0.0886 | 0.0837 | 42.4679 | 7.076 | 18.912 | 1.739x |
| redkitchen | original | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| redkitchen | cosine_distortion | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| redkitchen | rep_a | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| redkitchen | rep_b | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| redkitchen | rep_c | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| redkitchen | rep_d | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Equal-scene macro averages (only when all seven quality evaluations finish)

| Method | Complete scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 23.6689 | 84.6389 | 0.1537 | 0.1256 | 0.1397 | 48.0128 | 1.000x |
| saf3r_dino64 | 7/7 | 24.0620 | 84.6844 | 0.1426 | 0.1266 | 0.1346 | 46.5900 | 1.745x |
| original | 0/7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| cosine_distortion | 0/7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| rep_a | 0/7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| rep_b | 0/7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| rep_c | 0/7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| rep_d | 0/7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

¹ Ratio of equal-scene mean warmed median full-forward times. Below1x is slower.
Quality, speed and memory gates are separate. Depth AbsRel is not evaluated here, so complete historical acceptance cannot be claimed.
Missing/OOM/interrupted runs retain six N/A fields with a concrete reason; no silent frame/budget reductions.
Per-method input/config/source snapshots and raw metrics are stored under scene/method/.

## Differences against original (staged changes are cumulative)

AUC/F1 differences in percentage points; distance differences in metres. Runtime ratio = original / variant; >1 means faster.

| Variant | dAUC@3 pp | dAUC@30 pp | dACC m | dComp m | dCD m | dF1 pp | Original/variant time |
|---|---:|---:|---:|---:|---:|---:|---:|
