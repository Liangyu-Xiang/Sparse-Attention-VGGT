# Reuse + residual exploration: matched7Scenes128-frame evaluation

Same checkpoint, exact per-scene uniform128-frame list, preprocessing, pose and recon_unposed protocol.
AUC/F1 in percent, ACC/Comp/CD in metres; CD=(ACC+Comp)/2, unsquared.
Reconstruction threshold0.05m, downsampling0.0078125m. No parameters tuned to7Scenes truth.
Fresh paired Dense, SAF3R DINO per-frame Top64 (ETH3D calibrated profile), original output proxy, and new history-conditioned tokenwise cosine distortion with corresponding estimated-mass transfer.
New loss uses per-token cosine distortion, Drop=2*pi, original state costs/eligibility and <=20% budget. Transfer multiplier=assigned prior mass / own prior mass; genuine history saves unit-Key probability, excluding transferred mass. All prior/assignment/transfer/evidence costs count. Second-QK evidence is retained.
Budget20% is an upper bound; indivisible state choices may leave unused budget, recorded in layers.json.

| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm median s | Peak allocated GiB | Speedup vs Dense |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heads | dense | 51.5789 | 93.9944 | 0.0343 | 0.0308 | 0.0326 | 84.9976 | 12.189 | 14.360 | 1.000x |
| heads | saf3r_dino64 | 51.7470 | 93.9477 | 0.0331 | 0.0300 | 0.0316 | 85.3997 | 7.021 | 18.912 | 1.736x |
| heads | original | 44.6399 | 92.9536 | 0.0361 | 0.0329 | 0.0345 | 82.4135 | 81.806 | 15.496 | 0.149x |
| heads | cosine_distortion | 11.8274 | 81.8090 | 0.1282 | 0.0439 | 0.0861 | 41.6218 | 81.003 | 15.502 | 0.150x |
| chess | dense | 20.5135 | 88.4658 | 0.1399 | 0.0877 | 0.1138 | 35.9524 | 12.231 | 14.360 | 1.000x |
| chess | saf3r_dino64 | 20.0213 | 88.3095 | 0.1519 | 0.0845 | 0.1182 | 36.2923 | 7.024 | 18.912 | 1.741x |
| chess | original | 22.1826 | 88.9399 | 0.1368 | 0.0849 | 0.1109 | 37.3951 | 81.186 | 15.497 | 0.151x |
| chess | cosine_distortion | 10.1173 | 77.3897 | 0.2418 | 0.1209 | 0.1814 | 12.8323 | 79.166 | 15.499 | 0.154x |
| stairs | dense | 4.9172 | 80.9535 | 0.2208 | 0.4797 | 0.3502 | 15.3143 | 12.184 | 14.360 | 1.000x |
| stairs | saf3r_dino64 | 9.0223 | 83.3629 | 0.2002 | 0.4658 | 0.3330 | 15.5669 | 6.997 | 18.912 | 1.741x |
| stairs | original | 5.8809 | 77.2351 | 0.2251 | 0.4373 | 0.3312 | 14.3102 | 80.928 | 15.497 | 0.151x |
| stairs | cosine_distortion | 4.1462 | 70.7304 | 0.8837 | 0.2895 | 0.5866 | 7.1804 | 79.668 | 15.494 | 0.153x |
| fire | dense | 55.3150 | 94.5579 | 0.0489 | 0.0513 | 0.0501 | 66.7010 | 12.375 | 14.360 | 1.000x |
| fire | saf3r_dino64 | 55.7497 | 94.5128 | 0.0512 | 0.0566 | 0.0539 | 65.2743 | 7.104 | 18.912 | 1.742x |
| fire | original | 54.3225 | 93.7344 | 0.0402 | 0.0469 | 0.0436 | 75.8949 | 82.374 | 15.497 | 0.150x |
| fire | cosine_distortion | 15.0016 | 83.9042 | 0.1674 | 0.0844 | 0.1259 | 29.5292 | 81.359 | 15.501 | 0.152x |
| office | dense | 14.2758 | 77.1457 | 0.0471 | 0.0521 | 0.0496 | 61.0877 | 12.248 | 14.360 | 1.000x |
| office | saf3r_dino64 | 13.0823 | 75.4655 | 0.0475 | 0.0567 | 0.0521 | 59.4168 | 7.009 | 18.912 | 1.747x |
| office | original | 14.6120 | 77.9417 | 0.0535 | 0.0540 | 0.0538 | 55.1339 | 81.224 | 15.499 | 0.151x |
| office | cosine_distortion | 2.8543 | 70.9961 | 0.2325 | 0.0958 | 0.1642 | 18.5815 | 80.934 | 15.501 | 0.151x |
| pumpkin | dense | 10.6176 | 77.5980 | 0.5057 | 0.0965 | 0.3011 | 25.0739 | 12.439 | 14.360 | 1.000x |
| pumpkin | saf3r_dino64 | 10.1665 | 77.0107 | 0.4357 | 0.1039 | 0.2698 | 21.7118 | 7.125 | 18.912 | 1.746x |
| pumpkin | original | 10.1378 | 77.1805 | 0.3256 | 0.0877 | 0.2067 | 28.7542 | 82.450 | 15.498 | 0.151x |
| pumpkin | cosine_distortion | 4.2405 | 68.1976 | 0.4250 | 0.1241 | 0.2746 | 6.8789 | 81.189 | 15.497 | 0.153x |
| redkitchen | dense | 8.4646 | 79.7568 | 0.0795 | 0.0812 | 0.0803 | 46.9624 | 12.301 | 14.360 | 1.000x |
| redkitchen | saf3r_dino64 | 8.6450 | 80.1817 | 0.0788 | 0.0886 | 0.0837 | 42.4679 | 7.067 | 18.912 | 1.741x |
| redkitchen | original | 10.9621 | 81.9468 | 0.0704 | 0.0618 | 0.0661 | 57.9058 | 81.614 | 15.498 | 0.151x |
| redkitchen | cosine_distortion | 7.9929 | 75.8600 | 0.2461 | 0.1131 | 0.1796 | 16.3523 | 79.115 | 15.498 | 0.155x |

## Equal-scene macro averages (only when all seven quality evaluations finish)

| Method | Complete scenes | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Speedup¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 7/7 | 23.6689 | 84.6389 | 0.1537 | 0.1256 | 0.1397 | 48.0128 | 1.000x |
| saf3r_dino64 | 7/7 | 24.0620 | 84.6844 | 0.1426 | 0.1266 | 0.1346 | 46.5900 | 1.742x |
| original | 7/7 | 23.2483 | 84.2760 | 0.1268 | 0.1151 | 0.1209 | 50.2582 | 0.150x |
| cosine_distortion | 7/7 | 8.0258 | 75.5553 | 0.3321 | 0.1245 | 0.2283 | 18.9966 | 0.153x |

¹ Ratio of equal-scene mean warmed median full-forward times. Below1x is slower.
Quality, speed and memory gates are separate. Depth AbsRel is not evaluated here, so complete historical acceptance cannot be claimed.
Missing/OOM/interrupted runs retain six N/A fields with a concrete reason; no silent frame/budget reductions.
Per-method input/config/source snapshots and raw metrics are stored under scene/method/.

## Single-change differences against original

AUC/F1 differences in percentage points; distance differences in metres. Runtime ratio = original / variant; >1 means faster.

| Variant | dAUC@3 pp | dAUC@30 pp | dACC m | dComp m | dCD m | dF1 pp | Original/variant time |
|---|---:|---:|---:|---:|---:|---:|---:|
| saf3r_dino64 | +0.813765 | +0.408406 | +0.015809 | +0.011533 | +0.013671 | -3.668269 | 11.5827x |
| cosine_distortion | -15.222511 | -8.720707 | +0.205293 | +0.009464 | +0.107379 | -31.261593 | 1.0163x |

## New cosine objective and mass-transfer implementation

Loss=sum_j a_hat_j*(1-cos(V_j,V_representative(j))) using original head-mean Value features. Drop=2*pi; Exact=0. History-only assigns members to nearest selected History; History+Rep keeps historical members and represents others by Rep. Execution multiplier=assigned estimated mass/own estimated mass; genuine evidence excludes transferred mass. Original2x2grid,32-query groups,20%cap and candidate/Exact eligibility retained. Fixed numerical prior full-support mix1e-6, not tuned to test truth. Final true current-QK probabilities can differ from the estimated prior.

Layout bug in initial GPU helper was repaired before the valid quality rerun. All invalid interrupted attempts and early diagnostics are excluded;29 CPU/CUDA tests (including padded/noncontiguous history rows) and8 full128-frame numerical checks passed. All28 valid final runs passed finite metric, CD, input/hardware/protocol and timing checks. Control records retained through the layout repair never executed the new kernel.

| Method | Mean selected History / cluster / query group | Rep fraction of selected Keys | Exact fraction of selected Keys | Dropped cluster fraction |
|---|---:|---:|---:|---:|
| original | 0.085481349 | 70.0113% | 21.4116% | 22.2338% |
| cosine_distortion | 0.001743127 | 99.2541% | 0.5714% | 0.6192% |

The new combined loss/execution variant substantially degrades measured quality and nearly collapses to representative states. These observations do not isolate the contribution of loss versus transfer. Original controls were remeasured in this cohort and show small numerical/discrete-selection reproduction variation versus the previous cohort; comparisons here use the fresh controls only. No parameters were changed after inspecting quality.

Reference definition and implementation details: docs/experiments/reuse_cosine_distortion_reference.md and docs/experiments/reuse_cosine_distortion.md.
State histograms/count summaries are in per-run layers.json; state_behavior.json contains equal-scene/layer summaries.
