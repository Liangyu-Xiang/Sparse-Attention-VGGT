# Controlled coarse sensing and operator re-evaluation

Scenes: chess, fire, heads, office, pumpkin, redkitchen, stairs.
Input frames: 128; query selection: 64.
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
| legacy_mean_qk | True | 0.019859 | 0.000586 | 0.000014 | 41.024% | 41.045% | 35.260% | 1.170 |
| legacy_mean_qk | False | 0.001575 | 0.000588 | 0.000005 | 38.524% | 46.616% | 19.063% | 2.021 |
| per_head | True | 0.126212 | 0.049305 | 0.019369 | 59.366% | 73.030% | 27.952% | 2.264 |
| per_head | False | 0.112607 | 0.049469 | 0.019002 | 53.419% | 79.544% | 19.453% | 2.755 |

## Coarse-size sweep (matched-head relevance, no size prior)

| Coarse size | C | S | Top10% true patch mass | Top20% true patch mass | Top20% Top-K recall | Token coverage |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 0.114252 | 0.045394 | 41.376% | 55.116% | 84.712% | 19.247% |
| 64 | 0.113062 | 0.047287 | 40.540% | 54.336% | 82.242% | 19.361% |
| 128 | 0.112607 | 0.049469 | 39.430% | 53.419% | 79.544% | 19.453% |

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
| 10% | shared_value | 0.887728 | 0.127961 | 1.657202 | 12.017% |
| 10% | uniform | 0.849826 | 0.111785 | 1.542238 | 12.349% |
| 10% | dense_topk_oracle | 1.999959 | 0.274598 | 4.298212 | 49.338% |
| 15% | shared_value | 0.462086 | 0.056143 | 0.844636 | 16.896% |
| 15% | uniform | 0.448510 | 0.053729 | 0.803833 | 17.227% |
| 15% | dense_topk_oracle | 1.560866 | 0.177773 | 3.376729 | 56.390% |
| 20% | shared_value | 0.277611 | 0.030875 | 0.486354 | 21.904% |
| 20% | uniform | 0.270257 | 0.028691 | 0.472895 | 22.069% |
| 20% | dense_topk_oracle | 1.252841 | 0.124771 | 2.818612 | 61.907% |

## Per-layer shared-Value sensitivity

| Layer (0-based) | 10% mean NMSE | 15% mean NMSE | 20% mean NMSE | 10% dense-TopK NMSE |
|---|---:|---:|---:|---:|
| 0 | 0.015718 | 0.009799 | 0.007603 | 0.272266 |
| 1 | 0.120958 | 0.045558 | 0.031044 | 8.695701 |
| 2 | 0.483438 | 0.248130 | 0.146250 | 1.019693 |
| 3 | 0.155963 | 0.080043 | 0.048464 | 0.239121 |
| 4 | 0.163307 | 0.075242 | 0.041969 | 0.177782 |
| 5 | 0.088477 | 0.041315 | 0.023622 | 0.237095 |
| 6 | 0.096782 | 0.043561 | 0.025648 | 0.280613 |
| 7 | 0.339841 | 0.168167 | 0.095712 | 0.290617 |
| 8 | 0.357908 | 0.162746 | 0.087342 | 1.659360 |
| 9 | 0.307424 | 0.147812 | 0.083913 | 0.374407 |
| 10 | 0.816306 | 0.460405 | 0.276080 | 0.234122 |
| 11 | 0.330261 | 0.202268 | 0.130844 | 0.061652 |
| 12 | 0.699720 | 0.447147 | 0.298748 | 0.060074 |
| 13 | 0.855499 | 0.501589 | 0.310666 | 0.375500 |
| 14 | 0.162821 | 0.073198 | 0.038987 | 0.031109 |
| 15 | 0.550119 | 0.275523 | 0.143551 | 0.031084 |
| 16 | 1.081875 | 0.565838 | 0.325605 | 0.335104 |
| 17 | 0.545121 | 0.271240 | 0.144186 | 2.008751 |
| 18 | 5.312807 | 2.742867 | 1.672154 | 9.823383 |
| 19 | 3.759630 | 2.047911 | 1.304077 | 1.246635 |
| 20 | 0.928636 | 0.435031 | 0.240671 | 2.624232 |
| 21 | 1.864119 | 1.052774 | 0.664565 | 3.858417 |
| 22 | 1.468839 | 0.647428 | 0.343138 | 10.027101 |
| 23 | 0.799915 | 0.344471 | 0.177834 | 4.035191 |

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
