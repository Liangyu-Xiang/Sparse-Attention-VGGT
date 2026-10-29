# heads 128-frame original reuse_explore: stage profiling

Profile wall time 83.800s; CUDA forward interval 83.800s. One warmup, one profiled forward on GPU7. Peak allocated 15.498 GiB.

| Stage | Seconds | Share of CUDA forward interval |
|---|---:|---:|
| reuse_and_budget | 53.031944 | 63.284% |
| residual_fine_probe | 9.070722 | 10.824% |
| attention_online | 8.639372 | 10.310% |
| other_model_operations | 6.981769 | 8.331% |
| coarse_observation | 3.898691 | 4.652% |
| evidence_second_qk | 1.478486 | 1.764% |
| history_summary | 0.565068 | 0.674% |
| pooling | 0.052932 | 0.063% |
| execution_setup_and_gaps | 0.052348 | 0.062% |
| value_update | 0.028441 | 0.034% |

Separate uninstrumented original reference on GPU0: warmed median 84.826s (1 warmup, 3 measurements).

CUDA-event elapsed intervals include launch/queue gaps; not hardware active-kernel-only durations.
Original second-QK mass recovery is timed separately from attention using launch events; no per-layer synchronization.
All event instrumentation overhead remains in diagnostic total; this diagnostic is not a speedup comparison.
Other model operations is the measured total CUDA-event interval minus disjoint profiled stages: includes DINO, local attention, FFN, projections, heads, routing outside tagged intervals and launch gaps.
Reference wall timing is a separate uninstrumented run on GPU0; do not apply profile percentages to it as exact stage durations.
