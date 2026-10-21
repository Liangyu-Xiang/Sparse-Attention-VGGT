# Full-global SAF3R versus History80, 7Scenes and ETH3D

## User request and frozen scope

Apply each method to ALL24 global attention layers0–23; no Dense prefix or
deep-only substitution. Frame/local blocks remain unchanged. Compare actual
attention outputs with independent original VGGT Dense on identical frame lists,
checkpoint and preprocessing. Use all currently idle GPUs0–6, one scene bundle
per GPU, methods sequential within a bundle. Other jobs are never interrupted.

Methods:
- Dense original VGGT, checkpoint model_tracker_fixed_e20.pt, BF16 inference.
- Complete SAF3R repository default `eval_saf3r_vggt.yaml`: ETH3D-calibrated
 24-layer head profile,lazy first-correspondence-head Top4. This is NOT the
 earlier lazy Top128 variant, and is NOT claimed identical to paper DINO settings.
 First-frame/special policies,all_to_first,stride,query-probe,frame-local and
 DINO-index-reusing heads all use the repository implementation. Caches reset
 for each new input; missing optional fused selector uses existing equivalent
 blockwise selector fallback, not a change of retention.
- History80 exactly `parent_region_history80.json`: target20% total edges,
 specials inside budget,manual Attention ratios L0=0,L1–9=.2,L10–16=.8,L17–23=.2,
80% outgoing history,whole Parent selection,full sorting,original expanded
Tensor Core execution,no shift,no adaptive gate,no population compensation.
 Not the later Attention-first or Oracle schemes. Actual complete-Parent
 overshoot is recorded; methods are not equal-budget comparisons.

Datasets use configured SAF3R/DA3-Bench default scene entries,seed0,uniform max128:
-7Scenes: chess/fire/heads/office/pumpkin/redkitchen seq-01,stairs seq-02;
128 frames each. This is all seven default scene entries,not every original video.
-ETH3D: courtyard38,electro39,kicker31,pipes14,relief19,delivery_area40,facade76,
office26,playground32,relief_2=20,terrains42 valid images. All valid images used;
no repetition to manufacture128 frames. Existing default filters retained.
Terrace/meadow are excluded by the repository's existing default protocol,not
silently treated as evaluated. Exact image paths/indices are saved before running.

## Captures and comparisons

Every method records all24x16 actual head AV outputs via output-projection-input
hooks. Also capture complete attention-module outputs AFTER output projection,
BEFORE residual addition. These are different positions,reported separately.
Head comparison never assumes an output head after projection remains separate.

Two references:
1. rollout_vs_dense: same indexed layer/head on a separate complete Dense
trajectory,including accumulated upstream feature drift.
2. same_input_vs_dense: extra Dense SDPA on each sparse layer's OWN input,Q/K/V.
This isolates the current layer conditional on that input. Inputs differ between
the two sparse methods after layer0; this is not a common Dense teacher trajectory.
Functional linear projection creates the module-output counterfactual without
recursively invoking observation hooks. It does not replace actual model outputs.

Metrics: relative Frobenius L2=||sparse-Dense||_F/||Dense||_F,mean per-Query vector
cosine,flattened cosine,mean row-relative error,output norm ratio,raw energies.
All Queries included,with separate all/patch/special scopes. FP32 products,
FP64 accumulated statistics,streamed in Query chunks. No NxN attention matrices.
Dense BF16 AV/module outputs saved for exact reference (~13GB per128-frame scene).
Dataset means weight every paired scene equally,not frames,pixels or energies.
7Scenes and ETH3D are summarized separately; partial subsets explicitly labelled.

## Execution and artifacts

Worker: scripts/evaluate_full_global_output_comparison.py.
Queue/report: scripts/run_full_global_output_comparison.py.
Root: outputs/global_attention/full_saf3r_history80_20261021.
Two-frame gates: same root with `_smoke`,7Scenes heads and ETH3D courtyard.
Gate must cover Dense/SAF3R/History80,24 layers x16 heads,both capture positions,
finite model outputs and SAF3R lazy cache propagation.10 unit tests cover
capture positions,head metrics,macro weighting,partial coverage and frozen configs.

54 formal method runs =18 scenes x3 methods. Fresh models and reset state for
each method; no previous output overwritten. All source hashes frozen before
the campaign; queued jobs reject source changes rather than silently changing
method. Per-method raw results,input/config,source snapshots,status,output metrics
and scene CSV/Markdown are saved. Root dataset_summary.json,layer_macro.csv,
quality_per_scene.csv,dataset_report.md and per-dataset24-layer curves update
as scene bundles finish. Complete status requires all configured scenes.

Each full forward also evaluates pose and recon_unposed (depth not requested).
Save AUC@3/AUC@30,ACC/Comp/CD/F1 for each method/scene and equal-scene quality
macro on complete paired quality scenes. Pending/OOM/interrupted diagnostics
retain all six N/A fields with concrete reasons; output completeness and quality
completeness are distinct. A quality failure does not discard valid output stats.
7Scenes threshold0.05m,voxel0.0078125m; ETH3D threshold0.25m,voxel0.0390625m.
Distances metres,CD=(ACC+Comp)/2 unsquared; AUC/F1 fractions in JSON,percent in tables.

Not a speed benchmark: instrumented cold forward includes capture I/O,comparisons
and extra Dense counterfactuals. Record time/peak memory diagnostically; do not
divide these times for speedup or compare them to historical warmed runs.
Sparse speedup N/A without a separate paired warmed uninstrumented benchmark;
Dense1.00x is only its reference identity. Peak memory includes diagnostic buffers.
No truth-driven parameter tuning or Dense fallback is authorized.

## Completed results (2026-10-21)

Final auditable report view:
`outputs/global_attention/full_saf3r_history80_20261021_final/dataset_report.md`.
All7 default7Scenes scene entries and all11 configured ETH3D scene entries have
complete24x16 head captures,24 projected module outputs,and all six quality metrics
for each of Dense,SAF3R,History80 (54 paired logical method runs).

Original ETH3D relief_2 and terrains attempts encountered CUDA OOM. Their failures
are preserved in the original campaign; fresh-process repeats on idleGPU0/1 use
identical sources,inputs,methods and budgets. Repeated Dense six-metric results
match original Dense exactly. Consolidation validates frozen source hashes,exact
input geometry/frame paths/checkpoint,and capture/quality completeness before
creating report symlinks. `provenance.json` records each selected source. No
failure directory is overwritten. Other GPU jobs were observed during the
campaign; this alone does not prove the cause of both failures.

Quality equal-scene macros (AUC/F1 percent; ACC/Comp/CD metres):

| Dataset | Method | AUC@3 | AUC@30 | ACC | Comp | CD | F1 | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|---|
|7Scenes|Dense|23.04|84.29|0.1480|0.1243|0.1361|48.05|1.00x reference|
|7Scenes|SAF3R|23.60|84.43|0.1297|0.1244|0.1271|49.14|N/A: instrumented|
|7Scenes|History80|20.93|81.12|0.1404|0.1392|0.1398|43.45|N/A: instrumented|
|ETH3D|Dense|29.27|81.82|0.3037|0.8090|0.5563|63.04|1.00x reference|
|ETH3D|SAF3R|23.46|80.16|0.3108|0.8410|0.5759|60.64|N/A: instrumented|
|ETH3D|History80|15.34|72.05|0.4377|1.3547|0.8962|46.46|N/A: instrumented|

These are local matched measurements,not published-paper reference rows.
History80's AUC30/F1 changes against local Dense are -3.17/-4.60pp on7Scenes,
and -9.77/-16.58pp onETH3D. No quality gate is relaxed.

Projected module outputs versus independent Dense rollout reveal a consistent
pattern: History80 is closer in shallow layers but worse in middle/deep layers,
especially18–21. At layer19,SAF3R/History80 relative error and Query cosine are
124.12%/.4556 versus153.37%/.3206 (7Scenes),73.82%/.8199 versus154.32%/.3864
(ETH3D). These are attention-branch output metrics before residual addition,
not final task-score changes. Relative error can exceed100%.

This is not solely upstream accumulation: at layer18,using each method's own
current input,Dense-counterfactual relative error is50.76% versus107.24% on7Scenes,
and39.16% versus109.75% onETH3D (SAF3R versusHistory80). Since these own inputs
differ across methods,this is not a common-teacher controlled ablation and does
not by itself identify whether routing,Value representation or head sharing is
the underlying cause. No production method was changed as part of this diagnosis.

Final report includes both comparison types at every layer,all/patch/special
scopes,raw per-head AV CSV,full source/input provenance and four24-layer plots.
`segment_summary.json` is an equal-layer average of per-layer equal-scene metrics,
not an energy-pooled error.11 metric/capture/config/reporting tests pass.
