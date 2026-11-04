# Fixed-source unanchored clustering prototype

Workspace: `/data_NVME1/mmc_lyxiang/3D/SAF3R-source-local-unanchored`.
Branch: `codex/source-local-unanchored-20261105`.
Opt-in mode: `source_local_unanchored`.

This isolated prototype snapshots the current dirty Query-256 code; HEAD alone
is not the baseline. `source_local_baseline_manifest.json` records SHA256 hashes.
The integration-only diff is `outputs/source_local_unanchored_20261105/integration.patch`.
No changes were made in the original Query-group worktree.

## Algorithm

Every layout starts with exactly the same uniform, frame-local fine labels and
exact fine-cluster budget. After initialization, immutable sources are formed:

- `sequential`: groups of consecutive fine IDs, up to `source_width` clusters.
- `spatiotemporal`: compute fine cells' mean image coordinates, bin by consecutive
  frame chunks and rectangular image tiles, then split oversized tiles into
  at most `source_width` clusters. Frame/raster order is preserved inside a tile.
  `source_temporal_span` is an upper bound on the frame interval, not a guarantee
  that every source contains that many frames. Membership follows fine-cell
  centers; boundary cells can cross an image-tile boundary.

Each layer, including L0, runs source-local centroid Lloyd updates. All tokens
may change fine cluster, including former representatives. Existing labels win
exact distance ties. Empty clusters receive high-error donor tokens from the
same source. One arbitrary ordinary member per donor is protected only during
that repair; the protection expires before the next assignment. At the end,
choose the nearest real member to each current centroid as representative.

This removes global L0 assignment, the global center-distance/Top-K graph,
per-token neighbor-center feature gathers, and unused certification calculations.
It changes the candidate domain, so it is an algorithm variant, not an equivalent
implementation of adaptive-local clustering. Nonempty clustering does not imply
quality preservation. Image-coordinate continuity does not imply 3D continuity.

The current implementation is a bounded-batch PyTorch prototype, including
CPU layout construction and GPU dynamic indexing/sorting for empty repair.
These operations remain optimization opportunities. It is not a fused-kernel
performance ceiling. Source layout initialization must count in full-forward timing.

## Frozen scope and experiments

Query group remains 256; maximum Key retention remains 20%; clustering iterations
remain 2. The loss, four states, budgets, coarse evidence, priors, history sampling,
selected-Key unit weights and attention path are unchanged. This task does not
merge the concurrent K/V, history-export or Graph implementation optimization.

Twelve opt-in JSON configurations cover source width 16/32/64, sequential layout,
and spatiotemporal layout with span 1/2/4. Width is a number of candidate CLUSTERS,
not a Query-group size. Before a larger campaign, vary iteration count 1/2/3 as a
separate factor; avoid simultaneously changing the attention kernel or retention.

CPU diagnostics: 23 checks passed, including evolving random features,
coincident centers, representative migration, empty repair, source invariance,
nonempty/unique real representatives and actual 128-frame static geometry.
GPU diagnostics: all 12 configurations passed invariants on synthetic 128-frame
28x37-patch, 64D features. One warmup plus three repetitions; each repetition
contains initialization and three clustering updates. Raw initialization times,
update times and peak allocated memory are saved separately. GPU 6 strict-context
monitor reported no foreign process. These are clustering-only measurements;
features are synthetic and held fixed up to a common offset across updates.
They do not establish real-sequence accuracy, optimal parameters or full-model speed.

All six quality fields and Dense speedup are N/A with reasons in the raw and
human-readable diagnostic reports. These are not complete quality evaluations.

Future real evaluations use only heads/chess/stairs with 128 frames each, exactly
matching prior frame manifests/checkpoint/preprocessing and pose/reconstruction
settings. Report all predeclared variants; do not choose parameters against
7Scenes test truth. Include AUC@3, AUC@30, ACC/Comp/CD/F1 and initialization-inclusive
full-forward timing. Reuse unchanged Dense/SAF3R quality controls; absent newly
paired Dense timings, speedup is N/A with a reason. Three-scene means are not
seven-scene dataset results.

## CUDA Graph equivalence

Current `reuse_window_graph.eager()` contains the selector math; `Window` captures
that same callable. Replay is execution of captured kernels on updated runtime
buffers, not reuse of an old mask. Before each window, update prior/pi/history
IDs/allowed/uniform probabilities. Before each layer, update current features,
labels/representatives/population and tensor geometry. Structural changes require
a compatible cache key and graph or recapture. Python decisions/scalars and kernel
launch structure are frozen at capture and must not depend on unrepresented
changing data. Tail buffers must have valid sentinels/counts.

Previous-layer dependence is explicit history: previous attention evidence plus
current-layer clustering/coarse evidence -> current prior -> current gate,
four-state errors, budget and pack -> current support -> current attention and
next-layer evidence. The Graph cache and historical evidence are different objects.
Capture warmup is implementation overhead, not extra algorithm layers.

Replay alone should preserve the algorithm. Compare eager/captured support IDs,
counts, weights and diagnostics for changing layers/inputs/tails, then attention,
history and complete forwards. Same-kernel capture should match eager; changes
in reductions or kernel math require separate numerical tolerance checks because
small changes can alter discrete selection. The new clustering tests are not
Graph equivalence tests and do not establish full-forward Graph equivalence.

## Completed real-sequence campaign

The 20261105 campaign completed all 42/42 quality runs: 12 predeclared clustering configurations plus adaptive-local Graph/eager, each on heads/chess/stairs with 128 frames. See `outputs/source_local_quality_20261105/report_zh.md` (Chinese), `report.md` (all per-scene metrics), `validation.json`, and `graph_report.md`. Same physical GPU per scene, exact matched inputs/protocol and clean strict GPU monitors. All six quality metrics were measured. Dense speedup is N/A because unchanged Dense was not rerun. Quality is one final output per run; observed floating-point atomic nondeterminism means differences are not proof of stable accuracy gains. Graph/eager support equality is not strict: differences were observed in 12/621 real windows; repeated eager also differed in 11/621 windows.
