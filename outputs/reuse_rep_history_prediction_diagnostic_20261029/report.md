# History probability and fixed-support execution diagnostics

Heads, uniform 128 frames. Only the first query group (32 queries), all heads, layers 2/12/14/23. This is not a dataset quality evaluation. All six pose/reconstruction fields remain unavailable in quality.json.
Current Dense patch probabilities are calculated for diagnosis only and never enter selection, history estimation, tuning or subsequent layers.

| Layer | Relative-logit L1 | Probability-based L1 | Coarse uniform L1 | Relative-logit KL | Probability-based KL | Coarse uniform KL |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.405664 | 0.407346 | 0.404301 | 0.134749 | 0.136227 | 0.133754 |
| 12 | 0.887416 | 0.983804 | 0.955233 | 0.751625 | 2.552934 | 0.835123 |
| 14 | 0.938703 | 1.098093 | 1.053979 | 0.949453 | 2.779082 | 1.116065 |
| 23 | 0.085089 | 0.111081 | 0.087353 | 0.006468 | 0.010916 | 0.006836 |

Probability-based estimates here reuse the SAME C-run observations/QKV as the relative-logit estimator, not a separately trained/network B result. L1 is summed absolute probability error (range 0 to 2); KL is target Dense to estimate.
Relative evidence helps at sampled middle layers but is not universally better than coarse uniform; all are estimates.

| Layer | Transfer output relative L2 to current Dense | Original population output relative L2 to current Dense | Transfer vs population relative L2 |
|---|---:|---:|---:|
| 2 | 0.078160 | 0.071220 | 0.015499 |
| 12 | 0.190097 | 0.124517 | 0.112527 |
| 14 | 0.081417 | 0.065781 | 0.040663 |
| 23 | 0.020293 | 0.019970 | 0.007256 |

The second table fixes the exact selected support and current Q/K/V. Its alternate outputs do not feed later layers. Current-input Dense attention is an attention-output diagnostic, not an independently propagated Dense network or six-metric evaluation.
Warmed full-forward speedup is N/A because this instrumented diagnostic includes synchronized CPU/GPU reference checks and has no matched warmed Dense forward. Raw diagnostic runtime/memory are retained separately in forward.json.
