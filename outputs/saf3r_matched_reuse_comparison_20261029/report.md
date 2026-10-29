# SAF3R addition to reuse comparison

Local 7Scenes all seven scenes, uniform128 identical frames/checkpoint/preprocessing and pose/recon_unposed protocol. Fresh paired Dense for this cohort. Warmup1 +3 complete forwards; per-forward routing reset and DINO Top-K construction included. Distances metres, F1 threshold0.05m, downsample0.0078125m, CD=(ACC+Comp)/2 unsquared. DINO64 is a declared paper-mechanism configuration, not a recovered unpublished scene-to-K mapping. ETH3D calibrated head profile. lazy4 is separately labelled. No published reference rows.

|Scene|Method|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Median s|Peak GiB|Speedup|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | dense | 51.5789 | 93.9944 | 0.0343 | 0.0308 | 0.0326 | 84.9976 | 12.2092 | 14.3596 | 1.0000x |
| heads | saf3r_dino64 | 51.7470 | 93.9477 | 0.0331 | 0.0300 | 0.0316 | 85.3997 | 7.0180 | 18.9125 | 1.7397x |
| heads | saf3r_lazy4 | 50.9596 | 93.8611 | 0.0329 | 0.0301 | 0.0315 | 85.1315 | 6.5051 | 14.6152 | 1.8769x |
| chess | dense | 20.5135 | 88.4658 | 0.1399 | 0.0877 | 0.1138 | 35.9524 | 12.2401 | 14.3596 | 1.0000x |
| chess | saf3r_dino64 | 20.0213 | 88.3095 | 0.1519 | 0.0845 | 0.1182 | 36.2923 | 7.0231 | 18.9125 | 1.7428x |
| chess | saf3r_lazy4 | 20.1198 | 88.3756 | 0.1428 | 0.0800 | 0.1114 | 38.5585 | 6.4095 | 14.6152 | 1.9097x |
| stairs | dense | 4.9172 | 80.9535 | 0.2208 | 0.4797 | 0.3502 | 15.3143 | 12.2095 | 14.3596 | 1.0000x |
| stairs | saf3r_dino64 | 9.0223 | 83.3629 | 0.2002 | 0.4658 | 0.3330 | 15.5669 | 6.9900 | 18.9125 | 1.7467x |
| stairs | saf3r_lazy4 | 8.2800 | 83.4117 | 0.1993 | 0.4860 | 0.3426 | 16.3935 | 6.3819 | 14.6152 | 1.9131x |
| fire | dense | 55.3150 | 94.5579 | 0.0489 | 0.0513 | 0.0501 | 66.7010 | 12.3799 | 14.3596 | 1.0000x |
| fire | saf3r_dino64 | 55.7497 | 94.5128 | 0.0512 | 0.0566 | 0.0539 | 65.2743 | 7.0943 | 18.9125 | 1.7450x |
| fire | saf3r_lazy4 | 56.2787 | 94.5632 | 0.0488 | 0.0546 | 0.0517 | 67.5308 | 6.5384 | 14.6152 | 1.8934x |
| office | dense | 14.2758 | 77.1457 | 0.0471 | 0.0521 | 0.0496 | 61.0877 | 12.2663 | 14.3596 | 1.0000x |
| office | saf3r_dino64 | 13.0823 | 75.4655 | 0.0475 | 0.0567 | 0.0521 | 59.4168 | 7.0165 | 18.9125 | 1.7482x |
| office | saf3r_lazy4 | 13.4186 | 75.8743 | 0.0481 | 0.0568 | 0.0525 | 58.5554 | 6.3871 | 14.6152 | 1.9205x |
| pumpkin | dense | 10.6176 | 77.5980 | 0.5057 | 0.0965 | 0.3011 | 25.0739 | 12.4442 | 14.3596 | 1.0000x |
| pumpkin | saf3r_dino64 | 10.1665 | 77.0107 | 0.4357 | 0.1039 | 0.2698 | 21.7118 | 7.0947 | 18.9125 | 1.7540x |
| pumpkin | saf3r_lazy4 | 10.0025 | 76.8221 | 0.4504 | 0.1012 | 0.2758 | 22.8550 | 6.4340 | 14.6152 | 1.9341x |
| redkitchen | dense | 8.4646 | 79.7568 | 0.0795 | 0.0812 | 0.0803 | 46.9624 | 12.3119 | 14.3596 | 1.0000x |
| redkitchen | saf3r_dino64 | 8.6450 | 80.1817 | 0.0788 | 0.0886 | 0.0837 | 42.4679 | 7.0635 | 18.9125 | 1.7430x |
| redkitchen | saf3r_lazy4 | 8.4851 | 80.0837 | 0.0806 | 0.0864 | 0.0835 | 42.1834 | 6.4268 | 14.6152 | 1.9157x |

|Method macro|Scenes|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Median s|Peak GiB|Speedup|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 23.6689 | 84.6389 | 0.1537 | 0.1256 | 0.1397 | 48.0128 | 12.2944 | 14.3595 | 1.0000x |
| saf3r_dino64 | 7/7 | 24.0620 | 84.6844 | 0.1426 | 0.1266 | 0.1346 | 46.5900 | 7.0429 | 18.9122 | 1.7457x |
| saf3r_lazy4 | 7/7 | 23.9349 | 84.7131 | 0.1433 | 0.1279 | 0.1356 | 47.3154 | 6.4404 | 14.6152 | 1.9090x |
