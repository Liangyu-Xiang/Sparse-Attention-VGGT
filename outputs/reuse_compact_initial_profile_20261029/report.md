# heads 128-frame compact combined reuse: detailed stage profiling

Profile wall time 35.728s; CUDA forward interval 35.728s. One warmup, one profiled forward on GPU7. Peak allocated 15.903 GiB.

| Stage | Seconds | Share of CUDA forward interval |
|---|---:|---:|
| allocation_setup_and_gaps | 17.108451 | 47.885% |
| other_model_operations | 5.468383 | 15.305% |
| attention_online | 5.234982 | 14.652% |
| budget | 2.357439 | 6.598% |
| losses | 1.553655 | 4.349% |
| evidence_second_qk | 1.445768 | 4.047% |
| residual_fine_probe | 0.701875 | 1.964% |
| top_history | 0.538670 | 1.508% |
| coarse_observation | 0.517319 | 1.448% |
| pack | 0.292744 | 0.819% |
| history_summary | 0.163165 | 0.457% |
| transfer | 0.129261 | 0.362% |
| execution_setup_and_gaps | 0.121052 | 0.339% |
| pooling | 0.055705 | 0.156% |
| value_update | 0.028657 | 0.080% |
| geometry_prepare | 0.011283 | 0.032% |

Historical uninstrumented combined reference on GPU0: warmed median 105.681s (1 warmup, 3 measurements).

CUDA-event elapsed intervals include launch/queue gaps; not hardware active-kernel-only durations.
Original second-QK mass recovery is timed separately from attention using launch events; no per-layer synchronization.
All event instrumentation overhead remains in diagnostic total; this diagnostic is not a speedup comparison.
Other model operations is the measured total CUDA-event interval minus disjoint profiled stages: includes DINO, local attention, FFN, projections, heads, routing outside tagged intervals and launch gaps.
Reference wall timing is an earlier uninstrumented combined-method run on GPU0, NOT a contemporaneous speed comparison. Do not apply compact profile percentages to it as exact stage durations.
