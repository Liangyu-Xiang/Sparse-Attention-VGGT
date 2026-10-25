# Sequential evaluation of sharpening-aware History coverage

Pinned remote commit: `57bdb78c72e15f90113e0a58c24ad88214bded00`.
Branch: `refactor/unified-reuse-exploration`.
Isolated checkout: `/data_NVME1/mmc_lyxiang/3D/SAF3R-unified-eval-57bdb78`.
Predecessor: `../unified_refactor_46bd65b/` (must finish all seven quality runs first).

This revision scales History reuse gains by previous explicit-relation coverage
within a fixed canonical region, in addition to current coarse importance,
persistence, and conditional previous salience. Upstream hyperparameters remain
unchanged, including .20 total retention and all24 sparse global layers.

After predecessor completion, run upstream CPU/GPU correctness tests and a
two-frame full-model diagnostic, then run all seven 7Scenes scenes with the SAME
128-frame input lists, original checkpoint, preprocessing and quality protocol.
Only GPUs4/5; idle-card dispatch, one process/card; no unrelated processes killed.
The dependency queue pauses on failed predecessor or failed validation rather
than silently changing the algorithm or dispatching an invalid evaluation.

Compatibility overlay matches the two previous revision evaluations: FP32
autocast-disabled sparse mapping; Tensor Core probe-row padding with exactly four
effective probes; evaluator original checkpoint and strict expected-input checks.
No original result is overwritten; no remote writes are performed.

Required quality: AUC@3, AUC@30, ACC, Comp, CD, F1; unavailable with explicit
reason until measured. AUC/F1 percentages, distances metres; unsquared
CD=(ACC+Comp)/2; reconstruction threshold .05m, downsample .0078125m.
Equal-scene macro only after seven complete scenes. One warmup/three measured
complete forwards includes all clustering, selection, history and initialization.
Record cold/warm times, peak allocated/reserved memory and actual edge density.
Historical Dense/SAF3R/History80/reuse+residual results are not rerun. SAF3R is
local lazy per-frame Top4/ETH3D profile, not paper DINO-feature reproduction.
New paired speedup is N/A without contemporaneous matched Dense measurements.

`dependency_status.json` tracks waiting/validation/evaluation/complete stages;
`queue_status.json` tracks scene workers after validation. `comparison.md` and
`comparison.json` contain frozen-baseline comparisons; previous two revisions
remain available in adjacent output directories for final revision comparison.
