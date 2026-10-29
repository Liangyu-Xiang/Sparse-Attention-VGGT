# heads 128-frame original reuse_explore: detailed stage profiling

Profile wall time 87.143s; CUDA forward interval 87.143s. One warmup, one profiled forward on GPU7. Peak allocated 15.494 GiB.

| Stage | Seconds | Share of CUDA forward interval |
|---|---:|---:|
| operating_points | 32.881241 | 37.733% |
| residual_fine_probe | 9.108385 | 10.452% |
| top_history | 8.811893 | 10.112% |
| attention_online | 8.347923 | 9.580% |
| budget_states | 7.391378 | 8.482% |
| other_model_operations | 6.907861 | 7.927% |
| pack_support | 6.806485 | 7.811% |
| coarse_observation | 3.942793 | 4.525% |
| evidence_second_qk | 1.486238 | 1.706% |
| allocation_setup_and_gaps | 0.748789 | 0.859% |
| history_summary | 0.569652 | 0.654% |
| execution_setup_and_gaps | 0.058101 | 0.067% |
| pooling | 0.053565 | 0.061% |
| value_update | 0.028486 | 0.033% |

Separate uninstrumented original reference on GPU0: warmed median 84.826s (1 warmup, 3 measurements).

CUDA-event elapsed intervals include launch/queue gaps; not hardware active-kernel-only durations.
Original second-QK mass recovery is timed separately from attention using launch events; no per-layer synchronization.
All event instrumentation overhead remains in diagnostic total; this diagnostic is not a speedup comparison.
Other model operations is the measured total CUDA-event interval minus disjoint profiled stages: includes DINO, local attention, FFN, projections, heads, routing outside tagged intervals and launch gaps.
Reference wall timing is a separate uninstrumented run on GPU0; do not apply profile percentages to it as exact stage durations.
