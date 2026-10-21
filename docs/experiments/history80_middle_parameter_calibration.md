# Middle-layer parameter calibration after population compensation

Goal:reduce L10–16 attention-module output error and increase cosine,without
changing20% total edge target,Value/Attention algorithm,population compensation,
or shallow/deep parameter settings. Frame/local attention remains unchanged.

## Predeclared calibration

Dense common-input teacher,not each candidate's different rollout. Each
candidate receives the SAME original Dense input at each layer. Shared fine
Value update uses that input; each candidate has its own evolving compensated
history. Score projected attention outputs BEFORE residual addition.

ETH3D courtyard,office,relief_2:uniform16 frames,seed0,same profile checkpoint
and preprocessing as previous experiments.16 evenly spaced full32-Query groups;
even groups for training constraint,odd groups for validation selection. Exact
frame and Query lists,source snapshots,raw layer scores saved. No task truth,
pose/reconstruction metrics or7Scenes inputs used for selection.

ETH3D calibration scenes overlap prior ETH3D evaluation. Therefore ETH3D followup
is a calibration-domain check,NOT independent held-out evidence.7Scenes is the
untouched validation dataset. Default known-view filters remain enabled.

16 candidates:Attention alpha in0,.2,.4,.6,.8,1;history in0,.4,.8,except pure
Value alpha0 only requires history0. One pair shared by all middle layers,
not per-scene schedules. Outgoing history is limited using the NEXT destination
layer's incoming fraction and KA. L9 therefore sends the correct L10 budget;
L16 sends the original L17 budget. Other destination layers retain80% history.

Selection:minimum equal-scene/equal-middle-layer validation relative L2,
requiring training relative L2 and validation cosine no worse than baseline.
Teacher stops beforeL17; these are diagnostics,not completed quality runs.
All candidate runs output six unavailable fields with a concrete diagnostic
reason and no claimed speedup. Dense teacher is offline calibration only;
deployed attention does not use Dense guidance/fallback.

## Frozen selected configuration

Selected a0.2_h0.8:middle Attention20%,Value80%,history80%.
Original middle Attention80%,Value20%,history80%.
Other layers unchanged:L0 pureValue,L1–9 andL17–23 Attention20%/Value80%.
All total budgets20%;compensation enabled throughout.

Held-out Query-group calibration relative L2:23.0205% ->15.8272%;
cosine:.970874 ->.987201. History0 atalpha.2 scores15.85%/.9872,close but not
the predeclared optimum. Do not exaggerate this small history preference.
This selection no longer enforces middle Attention dominance:it was an
initial assumption,not an evidence-backed hard constraint.

Artifacts:outputs/global_attention/history80_middle_calibration_20261021,
selection.json and selected_config.json. Reusable exact configuration:
configs/evaluation/parent_region_history80_population_middle20.json.
Tuple/list differences after JSON serialization are semantically irrelevant.

## Full-rollout validation

Run only the frozen selected candidate,not a7Scenes truth-driven parameter
search,on all7 default7Scenes128-frame entries and all11 configured ETH3D
entries14–76 frames. Reuse exact prior Dense/SAF3R and population baseline
references through explicit links. All24 global layers use the sparse method.
Capture every layer and head,own-input Dense local counterfactual and independent
Dense rollout. Own-input references are different across methods,unlike the
common-input calibration teacher. Compare actual density,not just targets.

Original campaign:outputs/global_attention/history80_middle_tuned_20261021.
Preserve any OOM/failed attempt;repeat unchanged configuration in `_retry`,then
construct an auditable `_final` report view with explicit source provenance.
Office initially failed OOM while another workload also occupied itsGPU;
the repeat does not lower frames,budget or quality gates.

Every full quality run includes AUC@3,AUC@30,unposedACC,Comp,CD,F1;distance
metres,CD=(ACC+Comp)/2 unsquared.7Scenes F1threshold.05m/downsample.0078125m;
ETH3D .25m/.0390625m. All source/config/frame lists frozen. No parameters
retuned using7Scenes scores even if the selected candidate disappoints.

Times/peaks include capture and extra Dense comparisons,not deployment
benchmarks. Dense1.00x reference;sparse speedup N/A absent same-hardware warmed
uninstrumented paired timings. Reused baselines are not contemporaneous speed
comparisons. Output-closeness improvements need not imply task-score gains;
report all six metrics and original gates rather than hiding regressions.

## Completed validation and disposition

Final auditable view:
outputs/global_attention/history80_middle_tuned_20261021_final.
All7/7 default7Scenes scenes and11/11 configured ETH3D scenes complete,all24
layers/all16 heads/all six quality metrics. Office's unchanged retry completes
the paired set;the failed original remains recorded.58 CPU/GPU tests pass,
including middle-only incoming-history schedules and next-destination budgets.

Equal-scene then equal-middle-layer means of projected module-output metrics:

| Dataset | Reference | Before error % | After error % | Before cosine | After cosine |
|---|---|---:|---:|---:|---:|
|7Scenes|Own-input Dense|19.5203|11.0413|0.978081|0.993547|
|ETH3D calibration domain|Own-input Dense|21.5758|15.2671|0.973605|0.987824|
|7Scenes|Independent Dense rollout|23.7065|13.0862|0.969311|0.990634|
|ETH3D calibration domain|Independent Dense rollout|27.2443|19.7265|0.960700|0.979301|

Actual own-input references differ across sparse trajectories;only the offline
teacher shares a common Dense input across candidates. All target budgets remain
20%,with complete-Parent overshoot reported,not equal-executed-FLOP claims.

Six-metric quality macros (AUC/F1 percent,distances metres):

| Dataset | Method | AUC@3 | AUC@30 | ACC | Comp | CD | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
|7Scenes|Dense|23.04|84.29|0.1480|0.1243|0.1361|48.05|
|7Scenes|SAF3R lazy Top4|23.60|84.43|0.1297|0.1244|0.1271|49.14|
|7Scenes|Before:population A80/V20 middle|21.66|83.82|0.1342|0.1244|0.1293|49.48|
|7Scenes|After:population A20/V80 middle|22.59|83.93|0.1406|0.1298|0.1352|47.70|
|ETH3D calibration domain|Dense|29.27|81.82|0.3037|0.8090|0.5563|63.04|
|ETH3D calibration domain|SAF3R lazy Top4|23.46|80.16|0.3108|0.8410|0.5759|60.64|
|ETH3D calibration domain|Before:population A80/V20 middle|21.52|80.15|0.2984|0.7922|0.5453|64.58|
|ETH3D calibration domain|After:population A20/V80 middle|20.76|79.07|0.3260|0.8788|0.6024|55.60|

The requested output-error/cosine goal improves on both domains,but task quality
does NOT uniformly improve.7Scenes AUC3/AUC30 increase,but F1 falls1.7849pp.
ETH3D F1 falls8.9800pp,and AUC30 falls1.0826pp against the previous population
configuration. The selected candidate is therefore EXPERIMENTAL ONLY,not a
replacement for the default or the best-task-quality configuration. No7Scenes
truth-driven reselection ofalpha.4/.6 or other candidates was performed.

Against Dense,the retained0.2pp macroAUC30 gate fails on both datasets;ETH3D's
0.5pp macroF1 gate also fails. Per-scene gates fail too. Seequality_gates.json.
Depth unavailable;speed/memory gates unavailable owing to diagnostic capture.
Dense1.00x reference;all sparse speedups N/A without paired warmed full forwards.

Final population_comparison.md is titled middle-parameter ablation and contains
both comparison types at every layer,four-method quality tables and plots.
population_per_scene.md/CSV include all72 rows with reused baselines labelled;
head_macro.csv/layer_macro.csv contain the actual per-head/layer comparisons.
provenance.json maps unchanged repeat sources while preserving failed evidence.
No default configuration or completed prior result was overwritten.
