# Whole Value-region Attention experiment

Implementation: [scheme](../../docs/experiments/value_region_av.md).

Current result: [Chess128 paired report](chess128_boundary_regions/report.md).

All24 global layers, same manual A/V ratios and20% target retention as the old fused scheme;
no dense prefix/fallback. Whole Value-based regions replace spatial cells and terminal real-Key selection.

Dense / old fused / region warmed median complete forwards:16.4117 /14.2988 /12.5754 seconds.
AUC30:88.4662 /87.3659 /87.0571 percent. Allocated peaks:14.3586 /14.9271 /14.9269 GiB.
New selection2.5244s vs old4.0142s; new terminal local-Key QK count is zero.
New vs old latency reduction12.05%, but AUC30 worsens0.3088pp; final speed/memory/accuracy gates NOT met.

Actual unique density19.99955%–20.00141%; at most20 extra edges per Query group.
Every layer satisfies overshoot < largest current fine cluster population.
Input lists, per-sample/cold timings, configuration, source snapshots and independent profiles are saved.
52 tests pass. This is one sequence's pose-only experiment, not a seven-scene three-task assessment.

`region_interrupted_compile_v1` and `region_before_empty_region_guard` are preserved local development
artifacts, not final results; do not mix their timings with the current `region/chess/result.json`.
