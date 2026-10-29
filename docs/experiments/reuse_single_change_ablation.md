# Independent reuse exploration ablations

Dataset: 7Scenes, all seven scenes, uniform 128 frames. Every method is evaluated against contemporaneous Dense and original reuse_explore on identical per-scene inputs/hardware/protocol. Retention remains at most 20%; no truth-driven parameter changes. One warmup and three complete timed forwards, including grouping/permutation/history transfer, allocation, attention and evidence. Quality: pose AUC@3/AUC@30, unposed reconstruction ACC/Comp/CD/F1; metres, F1 threshold 0.05m, downsample 0.0078125m, CD=(ACC+Comp)/2 unsquared.

Independent configurations:

- original: all original defaults (fixed groups, minimum error, second selected-QK evidence pass).
- shift8: only query_shift=8. Circular flattened-query groups use offsets 0,8,16,24,0,... by layer. Q is reordered for execution and output restored. All queries occur once, including short final groups and special queries. History proxies are transferred using overlap fractions between new and previous query groups. Key histories are unioned, identical IDs coalesced, with no new budget truncation; previous canonical masses use the same overlap weighting. This is a transfer of group-mean proxies, not a recovery of individual-query probabilities. Canonical regions and Value clustering do not move.
- max_error: only allocation_objective=max_error. Keep the original cluster target, state estimates, costs, representatives and admissibility. For each cluster transform finite errors E to max(E)-E; inadmissible infinite states stay infinite. Use the unchanged discrete Lagrange solver and <=budget constraint to approximately maximize total selected error. L0 retains the original representative-only seed. Drop remains a legitimate choice; unused budget/all-dropped clusters are measured, never silently filled. This is not an exact knapsack maximum.
- single_qk: only evidence_mode=saved_logits. The original online-softmax attention kernel saves FP32 selected logits. A second normalization/reduction kernel uses final LSE and saved logits to produce the same head/query-mean selected-Key probabilities. No second QK multiplication; this is two kernels with one QK traversal, not a fully fused single kernel or a zero-overhead probability export. Scratch has size heads * window_queries * selected_keys * 4 bytes, and counts in peak memory; there is no full-sequence dense NxN allocation.

Numerical validation: original CPU/CUDA support/output tests plus shifted-group overlap/duplicate coalescing, all-24-layer shifted output restoration/budget, maximum-error admissibility and budget (including drop-worst), and GPU saved-logit evidence versus explicit reference and original second-QK kernel. Small real-model diagnostics are not quality evaluations; all six fields remain unavailable with an explicit reason.

The preliminary farthest-representative interpretation was discarded following the user's clarification. Its diagnostic directory is retained only as an unused preliminary diagnostic; it is not part of this experiment.

Run: scripts/run_reuse_ablation_eval.py --output-dir outputs/reuse_single_change_7scenes128_20261029

## Canonical-grid-only extension

Additional independent variants `grid4` and `grid8` use (grid_y,grid_x)=(4,4) and (8,8); everything else equals original (fixed query groups, minimum proxy error, original second-QK evidence, same Value clustering and 20% retention cap). Grid dimensions specify regions per frame, not pixel size. Grid partitions remain fixed through all layers. Region counts and persistent canonical matrices grow 4x/16x versus 2x2; the current pooling/probe and overlap costs must be measured, not inferred as speedups.

Full evaluation: outputs/reuse_canonical_grid_7scenes128_20261029. Same-scene Dense/original references from the current campaign are symlinked with original source snapshots and paired timing records, on the same GPU and identical input/protocol. New methods wait for the earlier campaign to finish to avoid GPU timing contention. All seven scenes and six metrics required for macro results.

A supplemental CUDA unit test overlapped heads/max_error on GPU0. That run is archived under timing_excluded and replaced by a full isolated re-evaluation before the grid campaign; its old timing is excluded.
