# Controlled coarse sensing and operator re-evaluation

Scenes: chess.
Input frames: 2; query selection: all.
All 24 global layers are probed on the SAME unmodified dense-prefix QKV.
The dense model forward and predictions are not changed by these probes.
No new sparse end-to-end AUC or production speedup is claimed.

Value fine partitions remain shared at 15% for sensing. For each layer,
the 2x2 sensing comparisons share an identical fine/coarse partition.
Coarse-size sweeps regroup the same fixed fine partition, without changing
the source features or feeding the new grouping into subsequent layers.
Layer-0 sensing is a counterfactual diagnostic; its actual Value-only bootstrap
is not changed. Uniform initialization and all mandatory special keys remain.

Reliability is measured on relevance probabilities WITHOUT a population prior
in the corrected default helper. Biased probabilities are explicit ablations.
CS is recorded as an uncalibrated statistic, not silently rescaled into a new
budget rule. Q_h and K_h are paired before any optional averaging.

## Matched 2x2 ablation (coarse size 128)

| Sensing | Size bias | C | S | CS | Top20% true patch mass | Top20% patch Top-K recall | Token coverage | Mass lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_mean_qk | True | 0.040687 | 0.001086 | 0.000057 | 40.464% | 41.098% | 38.701% | 1.060 |
| legacy_mean_qk | False | 0.002093 | 0.001121 | 0.000021 | 33.635% | 39.384% | 22.452% | 1.520 |
| per_head | True | 0.152034 | 0.080635 | 0.040369 | 53.643% | 63.048% | 32.151% | 1.791 |
| per_head | False | 0.131642 | 0.081751 | 0.039546 | 45.744% | 64.882% | 22.122% | 2.117 |

## Coarse-size sweep (matched-head relevance, no size prior)

| Coarse size | C | S | Top10% true patch mass | Top20% true patch mass | Top20% Top-K recall | Token coverage |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 0.137743 | 0.076153 | 34.082% | 48.127% | 73.754% | 19.050% |
| 64 | 0.136861 | 0.078252 | 33.041% | 46.560% | 69.242% | 20.171% |
| 128 | 0.131642 | 0.081751 | 28.664% | 45.744% | 64.882% | 22.122% |

## Isolated operator budget sweep

These selectors are DIAGNOSTIC CONTROLS, not a proposed deployable final method.
A common shared 20%-fine partition is used across 10/15/20% budgets so that
all budgets can select enough UNIQUE representatives. This differs from the
15%-fine sensing partition and must not be conflated with the old cumulative
15%-fine result. Same current input, FP32 mathematical logits/output, total
budget includes mandatory specials. The dense-topk control knows exact dense
scores and is not deployable cheaply; it is not an oracle for MINIMUM output NMSE.

| Total retention | Selector | Mean NMSE | Median NMSE | P90 head/layer/scene NMSE | Retained dense mass |
|---|---|---:|---:|---:|---:|
| 10% | shared_value | 1.415093 | 0.209059 | 2.274057 | 12.625% |
| 10% | uniform | 1.371878 | 0.240011 | 2.329889 | 12.760% |
| 10% | dense_topk_oracle | 4.755657 | 0.313159 | 8.366120 | 45.779% |
| 15% | shared_value | 0.722402 | 0.106462 | 0.989114 | 17.482% |
| 15% | uniform | 0.730387 | 0.121922 | 1.052991 | 17.155% |
| 15% | dense_topk_oracle | 3.775648 | 0.193281 | 7.020157 | 53.580% |
| 20% | shared_value | 0.431522 | 0.062004 | 0.511903 | 22.288% |
| 20% | uniform | 0.457535 | 0.106115 | 0.591868 | 22.056% |
| 20% | dense_topk_oracle | 3.038553 | 0.124025 | 5.883448 | 59.652% |

## Per-layer shared-Value sensitivity

| Layer (0-based) | 10% mean NMSE | 15% mean NMSE | 20% mean NMSE | 10% dense-TopK NMSE |
|---|---:|---:|---:|---:|
| 0 | 0.018719 | 0.019102 | 0.008219 | 0.264942 |
| 1 | 0.112671 | 0.070326 | 0.018137 | 10.275553 |
| 2 | 0.431681 | 0.245007 | 0.157879 | 1.191294 |
| 3 | 0.262067 | 0.151143 | 0.117856 | 0.260797 |
| 4 | 0.239592 | 0.105989 | 0.069317 | 0.249442 |
| 5 | 0.175637 | 0.094299 | 0.053019 | 0.255129 |
| 6 | 0.238154 | 0.131407 | 0.080015 | 0.306443 |
| 7 | 0.699108 | 0.468057 | 0.290266 | 0.336851 |
| 8 | 0.346233 | 0.156273 | 0.095253 | 1.898617 |
| 9 | 0.574201 | 0.336614 | 0.223574 | 0.440702 |
| 10 | 0.633390 | 0.319521 | 0.202274 | 0.199283 |
| 11 | 0.265516 | 0.173173 | 0.127670 | 0.061957 |
| 12 | 0.459893 | 0.290393 | 0.207627 | 0.043929 |
| 13 | 0.830956 | 0.514752 | 0.321059 | 0.305009 |
| 14 | 0.116065 | 0.068724 | 0.048782 | 0.047104 |
| 15 | 0.356484 | 0.185960 | 0.114080 | 0.110214 |
| 16 | 1.294555 | 0.659747 | 0.367310 | 0.610901 |
| 17 | 0.639353 | 0.280412 | 0.158950 | 1.062405 |
| 18 | 6.759072 | 4.064556 | 2.608130 | 25.608002 |
| 19 | 3.247450 | 1.374364 | 0.711549 | 1.896240 |
| 20 | 3.177281 | 1.602675 | 0.943960 | 6.089991 |
| 21 | 3.006898 | 1.610753 | 1.055159 | 7.816829 |
| 22 | 5.990978 | 2.818449 | 1.538176 | 35.737176 |
| 23 | 4.086279 | 1.595956 | 0.838274 | 19.066954 |

## Metric corrections and limits

- True patch mass: dense probability in selected groups / total dense patch probability.
- Dense-total mass retains the special-token mass denominator and is separately saved.
- Group Top-K recall is count overlap, NOT attention mass (the old review mixed these).
- Token coverage and mass lift are saved: large groups can improve raw mass simply
  by covering more keys. Removing size bias need not increase raw retained mass.
- Head/query errors are not interchangeable; per-scene/layer/head CSVs are retained.
- Coarse size 32 entails roughly 4x sensing QK work versus 128 at equal heads/queries.
- Sampling 64 queries on 128-frame inputs is a probe, NOT a full-query certification.
- No NMSE safety threshold, alpha calibration, adaptive total budget or new shared/head
  mask policy has been silently introduced. The slow indexed production path is not fixed
  by these sensing-only changes and requires separate kernel/selection redesign.
- A different future mask changes later features; new final AUC requires a new model run.
