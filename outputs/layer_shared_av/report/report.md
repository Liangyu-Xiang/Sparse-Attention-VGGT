# 7Scenes128 revised A/V experiment

Zero-based layers. Fixed final density including mandatory special keys; local/frame attention unchanged.
One warmup + three measured forwards per scene. Macro means of per-scene medians.

| Method | AUC30 (%) | AUC5 (%) | Whole (s) | Global (s) | A 3–9 (%) | A 10–16 (%) | A 17–23 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| reference | 33.322 | 2.227 | 8.347 | 2.709 | 0.00 | 0.00 | 0.00 |
| screened_tau08_v1 | 46.403 | 9.799 | 108.018 | 102.468 | 9.45 | 50.19 | 1.07 |
| tau09_noncanonical_v1 | 46.882 | 11.515 | 76.457 | 70.931 | 13.28 | 50.37 | 2.79 |
| first_corrected_tau09_v3 | 52.098 | 10.906 | 85.188 | 79.649 | 15.83 | 50.60 | 2.96 |
| selected_joint_v3 | 60.291 | 15.520 | 99.075 | 93.500 | 14.83 | 50.24 | 4.93 |

Reference: multiplier1, tau.9, eta1%. Screened v1: multiplier32, tau.8, eta.25%. Selected v3: multiplier32, tau.95, eta.25%, query chunk1024, canonical coarse-QK tiles128 and token-level history capacity.
Initial screening used Chess offline sensing without pose labels. Final selection used the correct-policy joint Chess scale/threshold sweep; other six scenes were not used to select those parameters.
Offline Dense-trajectory calibration ratios are NOT actual sparse-execution ratios; the table uses actual retained A slots.

| Layer | Reference raw alpha (%) | Final raw alpha (%) | Final A slots (%) | Value-only scenes / 7 |
|---|---:|---:|---:|---:|
| 0 | 0.010 | 7.97 | 7.97 | 0 |
| 1 | 0.006 | 7.81 | 7.81 | 0 |
| 2 | 0.025 | 7.91 | 7.07 | 3 |
| 3 | 0.066 | 13.84 | 13.06 | 2 |
| 4 | 0.071 | 17.57 | 17.57 | 0 |
| 5 | 0.105 | 22.92 | 22.92 | 0 |
| 6 | 0.057 | 14.44 | 12.97 | 3 |
| 7 | 0.084 | 11.14 | 10.44 | 2 |
| 8 | 0.021 | 9.64 | 8.33 | 2 |
| 9 | 0.086 | 18.56 | 18.56 | 0 |
| 10 | 0.119 | 24.12 | 23.90 | 1 |
| 11 | 0.211 | 59.77 | 59.76 | 0 |
| 12 | 0.298 | 58.07 | 58.07 | 0 |
| 13 | 0.168 | 59.56 | 59.56 | 0 |
| 14 | 0.878 | 74.43 | 74.43 | 0 |
| 15 | 0.120 | 54.75 | 54.75 | 0 |
| 16 | 0.036 | 21.26 | 21.21 | 1 |
| 17 | 0.007 | 9.24 | 9.24 | 0 |
| 18 | 0.001 | 2.31 | 0.00 | 7 |
| 19 | 0.023 | 4.54 | 3.71 | 4 |
| 20 | 0.002 | 3.12 | 0.00 | 7 |
| 21 | 0.088 | 17.06 | 16.86 | 2 |
| 22 | 0.001 | 1.14 | 0.00 | 7 |
| 23 | 0.004 | 5.88 | 4.70 | 2 |

Chess was used for calibration, so the seven-scene macro mean includes one calibration scene. Independent held-out six-scene results:

| Method | Held-out AUC30 (%) | Held-out AUC5 (%) |
|---|---:|---:|
| reference | 34.766 | 2.520 |
| screened_tau08_v1 | 53.533 | 11.427 |
| tau09_noncanonical_v1 | 53.788 | 13.428 |
| first_corrected_tau09_v3 | 58.511 | 12.713 |
| selected_joint_v3 | 56.344 | 13.257 |

Archived accuracy references (same input paths/shape/checkpoint checked, NOT paired timings): {"dense": {"auc30": 84.2926860704912, "heldout_six_auc30": 83.59709919072615}, "saf3r": {"auc30": 84.57630296212974, "heldout_six_auc30": 83.92224409448819}}

Earlier fixed hybrid AUC30 83.817% is also an archived accuracy-only reference.
Earlier fixed hybrid kept global layers 0–2 Dense and used full fine Dense QK TopK on layers10–16; it is NOT the present all24 fixed-10% method.
Do not infer speedup or safety from a good branch profile alone. See aggregate.json for stage times and pure-mode fractions.
Stage profiles are from each scene's last measured forward, not medians of per-stage repeats; do not sum them and compare against median whole/global latency.


## Fresh Chess128 full-cost comparison

Same inputs/checkpoint and 1 warmup + 3 repeats. A/V is a separate run, not randomized interleaving. SAF3R rebuilds lazy Top-K inside every timed forward; original profile/topk128-per-frame is not a uniform10% budget.

| Method | AUC30 (%) | Whole (s) | Global attention (s) |
|---|---:|---:|---:|
| dense | 88.466 | 16.726 | 11.170 |
| saf3r | 88.501 | 10.750 | 4.837 |
| selected_joint_v3 | 83.972 | 95.650 | 89.942 |

## Interpretation

Regional A/V contrast was obtained, but the middle is only 50.24% Attention on the seven-scene mean, not100% and not Attention-majority in every middle layer or held-out scene.
Selected seven-scene AUC30 60.291%, independent six-scene 56.344%, Fire 1.743%. This is an unacceptable accuracy loss, despite successful Chess calibration.
No layer entered Attention-only. Value-only accounts for 25.60% of scene-layers; the rest are mixed and incur full per-query routing/indexed-attention/validation costs.
Current implementation is an opt-in experiment, NOT a usable acceleration replacement. Ratio fitting is not evidence of retained model quality or runtime speedup. All rejected arms are retained separately.
