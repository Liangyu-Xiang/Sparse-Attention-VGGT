# Independent validation and diagnostic stage breakdown

These extra forwards are NOT formal speed measurements. Extra held-out output checks and diagnostic events are enabled.
Their pose predictions must agree with the formal run (max absolute pose-encoding difference <=1e-4).
Validation queries use offset 3/8 of each query block, distinct from routing offsets 1/4,3/4 and calibration offset 1/2.
The reference is dense attention on the SAME current sparse-prefix QKV, not a Dense-prefix reference.
Reference output uses BF16 Flash SDPA; reported kernel errors therefore include its small numerical differences.
No validation output feeds back into masks or budget decisions. Sampled checks still do not certify all queries.

| Method | Mean calibration NMSE | Mean independent NMSE | P90 head/layer/scene NMSE | Independent pass % |
|---|---:|---:|---:|---:|
| adaptive | 0.010442 | 0.010569 | 0.018138 | 94.90 |
| fixed10 | 0.319123 | 0.318068 | 0.615726 | 14.62 |

## 24-layer diagnostic stage totals

Per-scene sums, then scene macro mean (ms). Event intervals include intervening host dispatch/synchronization gaps.
The separate independent validation checks are outside these stage intervals and outside formal timing.
QKV/projection and bookkeeping outside marked intervals mean stage sums are not identical to attention-module totals.

| Stage | adaptive | fixed10 |
|---|---:|---:|
| coarse_sensing | 171.023 | 171.156 |
| dense_budget_probes | 249.495 | 245.887 |
| dense_fallback | 302.263 | 0.375 |
| history_matching_budget_selection | 592.048 | 168.299 |
| mask_assembly | 239.757 | 289.323 |
| normalization_layout | 725.520 | 761.359 |
| qkv_layout | 145.150 | 144.864 |
| same_pass_history_update | 47.184 | 24.596 |
| sparse_attention | 5266.652 | 1474.211 |
| value_hierarchy | 1086.983 | 1165.102 |
