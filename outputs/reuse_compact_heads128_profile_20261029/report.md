# heads 128-frame compact combined reuse: detailed stage profiling

Profile wall time 20.010s; CUDA forward interval 20.009s. One warmup, one profiled forward on GPU7. Peak allocated 15.902 GiB.

| Stage | Seconds | Share of CUDA forward interval |
|---|---:|---:|
| other_model_operations | 5.545059 | 27.712% |
| attention_online | 5.221799 | 26.097% |
| budget | 2.369439 | 11.842% |
| losses | 1.548061 | 7.737% |
| evidence_second_qk | 1.442294 | 7.208% |
| allocation_setup_and_gaps | 0.798776 | 3.992% |
| residual_fine_probe | 0.695511 | 3.476% |
| top_history | 0.541733 | 2.707% |
| coarse_observation | 0.514162 | 2.570% |
| finalize | 0.348450 | 1.741% |
| pack | 0.292001 | 1.459% |
| diagnostics | 0.291579 | 1.457% |
| history_summary | 0.163039 | 0.815% |
| transfer | 0.126867 | 0.634% |
| pooling | 0.054732 | 0.274% |
| value_update | 0.028612 | 0.143% |
| execution_setup_and_gaps | 0.016161 | 0.081% |
| geometry_prepare | 0.011164 | 0.056% |

Historical uninstrumented combined reference on GPU0: warmed median 105.681s (1 warmup, 3 measurements).

CUDA-event elapsed intervals include launch/queue gaps; not hardware active-kernel-only durations.
Original second-QK mass recovery is timed separately from attention using launch events; no per-layer synchronization.
All event instrumentation overhead remains in diagnostic total; this diagnostic is not a speedup comparison.
Other model operations is the measured total CUDA-event interval minus disjoint profiled stages: includes DINO, local attention, FFN, projections, heads, routing outside tagged intervals and launch gaps.
Reference wall timing is an earlier uninstrumented combined-method run on GPU0, NOT a contemporaneous speed comparison. Do not apply compact profile percentages to it as exact stage durations.
