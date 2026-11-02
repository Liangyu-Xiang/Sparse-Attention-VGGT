# Repaired RMS Drop + historical states: dead-work cleanup

This branch contains the exact attention implementation used by the 2026-11-02 cleanup timing diagnostic. The original working tree contains other experiment variants; this branch was assembled from the frozen tested snapshot rather than those variants.

## Method

Keep 20% total support budget, query groups of 32, processing windows of 64, current-layer adaptive Value clustering, RMS Drop, squared-L2 representation losses, 16 budget iterations, historical probability estimation and probability-mass transfer. A historical Key that is also Rep uses the historical estimate. Keep every qualifying historical Key in a single History state; independent Rep is removed for historical Rep. Exact remains eligible under the original rules.

The cleanup removes obsolete Top-4/prefix/mixed losses, unread intermediates and overwritten probabilities. The solver uses the four valid states in the original tie order [Drop, Rep, History, Exact] (legacy IDs [0,1,5,10]). No new approximation or cross-layer frozen state was introduced. Optional modules imported by the frozen implementation are included for a complete checkout.

## Configuration and validation

Use `configs/evaluation/reuse_zdrop_history_states.json` with `scripts/evaluate_zdrop_history_states.py`. Dataset and checkpoint paths must be supplied/configured for the target machine; no datasets or weights are included. `baseline_source_hashes.json` records inherited reference files, not hashes of the cleaned implementation.

Run `python scripts/validate_zdrop_history_cleanup.py` in the project environment with CUDA. It verifies retained losses, costs, ownership, state selection, packed Key IDs, transfer weights, and tie ordering. `ZDROP_SOURCE_ROOT` optionally selects an extracted snapshot.

## Recorded diagnostic

See `outputs/zdrop_cleanup_heads128_20261102/report.md`. On one RTX5090 D, heads/128 frames, one warmup and three full forwards: median 59.88834 -> 53.83086 seconds (1.1125x relative to the pre-cleanup method), peak allocated 20.17847 -> 19.62244 GiB. This is a single-sequence timing diagnostic, not seven-scene quality validation. All six quality metrics and paired Dense speed are N/A; unchanged controls were not rerun. Mathematical equivalence tests passed, but full-model runs show numerical variation; no claim of verified quality nondegradation is made.

Exact before/after source trees and source hashes are archived in `before_source.tar`, `after_source.tar`, and `manifest.json` in that output directory. Extract into an empty directory with `tar -xf <archive> -C <empty-directory>`. Profiling/setup scripts preserve the original local `/tmp` paths and output locations; review those paths before rerunning them.
