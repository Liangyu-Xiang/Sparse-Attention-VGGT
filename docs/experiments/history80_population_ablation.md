# History80 residual cluster-population compensation

This is a one-variable experiment against the complete all24-global-layer
SAF3R/History80 dataset comparison. It does not change budgets,clustering,
region selection criterion,manual branch ratios,80% history or the checkpoint.
Frame/local attention remains untouched. It is not an Oracle or Attention-first
variant. The default production configuration remains uncompensated; enable
`ValueRegionAVConfig(population_compensation=True)` explicitly.
Reusable opt-in config: `configs/evaluation/parent_region_history80_population.json`.

## Compensation semantics

Previous region sensing already used log(region population). That is not final
Value representative softmax compensation. For a current Value cluster C with
representative r and Query-group Attention support A,assign the representative
weight w=|C|-|C intersect A|,and add log(w) to its QK logit before the SINGLE
shared online softmax. Attention real Keys and all special Keys retain unit
weight. Selection excludes Value representatives from A,so w is at least1.
Every original patch member therefore contributes mass exactly once: either as
a retained real Attention Key or through its cluster representative. This does
not claim exact Dense reconstruction unless members also share Key and Value.
Representatives still use their actual K/V,not mean K/V.

Mass uses the actual current-layer merged Value partition,not the base fine
partition when fewer Value representatives are used. Population residuals vary
by Query group because Attention support differs; actual head/Q weights remain
separate. Final Triton execution applies FP32 log weights during same-pass QK,
and history evidence uses these compensated probabilities. It never multiplies
AV after normalization or normalizes the two branches independently.

The optional bias is bounded to a Query-group window,shape [groups,common Keys],
not a Token x Token mask. Population counting,overlap subtraction and weighting
are onGPU. The disabled constexpr path retains the original kernel arithmetic.
Descriptor execution is explicitly unsupported rather than silently unweighted.

## Protocol and artifacts

Baseline: outputs/global_attention/full_saf3r_history80_20261021_final.
Campaign: outputs/global_attention/history80_population_20261021.
Smoke: same campaign root with `_smoke`,7Scenes heads2frames,all24 layers.
Runner: scripts/run_history80_population_comparison.py.

Reuse the exact prior Dense references/quality and SAF3R quality/output statistics
through read-only baseline links. Run only compensated History80 on all7 default
7Scenes entries (128uniform frames) and all11 configured ETH3D entries (14–76
actual valid frames). Do not manufacture128 frames for ETH3D. Input paths,
indices,geometry and checkpoint must equal prior Dense before inference.
Freeze source hashes and save per-scene snapshots; no prior result overwritten.

Output comparisons capture all24x16 actual head AV and projected module outputs
before residual,with independent Dense rollout AND each sparse layer's own-input
Dense counterfactual. The latter is not a common-teacher trajectory. All Queries,
all/patch/special scopes; equal-scene macros separately for each dataset.

Every formal method run evaluates pose and recon_unposed:AUC@3,AUC@30,ACC,Comp,
CD,F1. Reconstruction metres,CD=(ACC+Comp)/2 unsquared.7Scenes threshold0.05m,
downsample0.0078125m; ETH3D threshold0.25m,downsample0.0390625m.
Save all six N/A fields with concrete reason for diagnostics/failures,not invented
quality. No parameters selected against7Scenes or ETH3D truth in this experiment.

Instrumented time/peak include extra Dense probes and output comparisons. They
are NOT valid deployment timings. Sparse end-to-end speedup is N/A absent paired
warmed uninstrumented measurements,Dense1.00x reference. Reused baselines are
explicitly not contemporaneous speed measurements. Complete quality evaluations
and output diagnoses are reported separately from runtime.

`population_comparison.md` compares original and compensated History80 against
the same Dense/SAF3R baselines,with six-metric macros and24-layer tables/plots.
`dataset_report.md`,per-scene comparison.json/CSV,output_metrics.json and raw
result.json preserve complete details. Partial macros are labelled and are not
compared to a complete baseline as if dataset coverage were equal.

## Completed results (2026-10-21)

All18 changed-method scene runs succeeded:7/7 default7Scenes entries and11/11
configured ETH3D entries. All24 layers and16 heads captured; six quality metrics
complete.18 new History80 forwards plus reused exact-input baselines,not54 new
forwards.57 CPU/GPU tests passed,including all24 compensated wrapper layers,
residual population conservation,mixed support overlap,specials,tail tiles,
single online softmax and compensated evidence versus explicit references.

Quality equal-scene macros (AUC/F1 percent,distances metres):

| Dataset | Method | AUC@3 | AUC@30 | ACC | Comp | CD | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
|7Scenes|Dense|23.04|84.29|0.1480|0.1243|0.1361|48.05|
|7Scenes|SAF3R lazy Top4|23.60|84.43|0.1297|0.1244|0.1271|49.14|
|7Scenes|History80 original|20.93|81.12|0.1404|0.1392|0.1398|43.45|
|7Scenes|History80 population|21.66|83.82|0.1342|0.1244|0.1293|49.48|
|ETH3D|Dense|29.27|81.82|0.3037|0.8090|0.5563|63.04|
|ETH3D|SAF3R lazy Top4|23.46|80.16|0.3108|0.8410|0.5759|60.64|
|ETH3D|History80 original|15.34|72.05|0.4377|1.3547|0.8962|46.46|
|ETH3D|History80 population|21.52|80.15|0.2984|0.7922|0.5453|64.58|

Population compensation substantially reduces deep-layer output bias. AtL18,
own-input Dense-counterfactual projected-output relative error falls from107.24%
to4.37% on7Scenes,and109.75% to3.28% onETH3D. Independent-rolloutL19 error falls
from153.37% to21.72% on7Scenes,and154.32% to40.09% onETH3D. Own-input reference
changes when the sparse trajectory changes: this is not a common-teacher
controlled comparison. Histories evolve from compensated same-pass evidence.

This does NOT establish lossless task accuracy. AUC30 still trails local Dense
by0.4707pp on7Scenes and1.6648pp onETH3D,so the prior0.2pp macro gate FAILS on
both datasets. MacroF1 improves over Dense by1.4358pp/1.5432pp,but scene gates
still fail (7Scenes heads,office,stairs; ETH3D details in quality_gates.json).
No gates were relaxed. Depth gate unavailable because depth was not evaluated;
speed/memory gates unavailable because these are instrumented diagnostics.
Dense speedup1.00x reference; all sparse speedups N/A without paired warmed
uninstrumented forwards. Baselines are local,not paper reference scores.

Complete results: `population_comparison.md` (four methods and24-layer tables),
`population_per_scene.md`/`population_quality_per_scene.csv` (all72 rows including
reused baselines),`head_macro.csv`,`layer_macro.csv`,`quality_gates.json`.
Per-method raw metrics,complete frame lists,configurations,source snapshots,
actual density and compensated-history metadata are retained. No original
uncompensated results or default configuration were overwritten.
