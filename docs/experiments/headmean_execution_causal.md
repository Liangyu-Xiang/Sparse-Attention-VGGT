# Separating head-mean estimation from grid-grouped single-QK execution

The combined implementation changed two mechanisms. This diagnostic separates them with a fixed2×2 design on ETH3D's full11 scenes (all available14–76 frames). The original per-head/two-QK and combined head-mean/one-QK results are reused. Only the two missing variants run, once per scene, with no budget, loss or GT-driven tuning. Compensation remains enabled throughout.

|Variant|Estimator|Execution|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|
|---|---|---|---:|---:|---:|---:|---:|---:|
|A|per-head|original two-QK|24.41852|80.00208|0.35150|0.90507|0.62828|56.71405|
|B|head-mean|original two-QK|24.27828|79.15001|0.30339|0.83636|0.56988|60.06878|
|C|per-head|grid-grouped one-QK|24.10529|77.95064|0.40451|0.96617|0.68534|53.68323|
|D|head-mean|grid-grouped one-QK|22.34480|78.41895|0.35583|0.89430|0.62507|54.91852|

Both independent comparisons locate reconstruction damage in the new execution path: C-A raises CD0.057058m and reduces F1 3.030817pp; D-B raises CD0.055188m and reduces F1 5.150262pp. Head averaging improves reconstruction in both execution paths (B-A and D-C), but is not universally pose-neutral: B-A reduces AUC@3 0.140242pp and AUC@30 0.852068pp. The changes interact: for AUC@3, D-B-C+A=-1.620243pp; for F1=-2.119445pp. Do not attribute the combined result solely to head averaging or claim its pose scores are unchanged.

The execution treatment includes Key regrouping, changed online-softmax tile boundaries/order and compact grid probability reduction. Mathematically equivalent real-number formulas do not imply bitwise-identical BF16 attention outputs. This study identifies the affected execution path; it does not separately quantify the end-to-end contribution of each floating-point rounding operation.

All22 new-run input/checkpoint/model/protocol/frozen-source/GPU overlap/retention checks pass. Full six-metric per-scene records, conditional contrasts, raw results and source archives: `outputs/headmean_execution_causal_20261102`. CD=(ACC+Comp)/2, unsquared metres; F1 threshold0.25m, downsample0.0390625m. Timings of the four quality rows are different batches, not a paired speed experiment. Dense speedup remains N/A because Dense was not rerun.

## Why single QK became slower

`outputs/single_qk_real_window_causal_v2_20261102/result.json` measures identical real heads128 L12 inputs on the same GPU:16 heads,2048 queries,133248 Keys,64 support groups of26009 slots,512 grids. Both wrappers return Attention output, weighted/uncompensated grid summaries and raw probe logits; old summary aggregation is INCLUDED. Two warmups and three alternating measured samples each:

- Original two-QK wrapper:3.670571ms median.
- Grid-grouped single-QK wrapper:6.498241ms median,1.77036× the latency (local wrapper speed ratio0.564856×).
- Old patch Key tiles:26048; new:37066,42.30% more. Separate grid segments create partially empty64-Key tiles. Additional costs include GPU sorting/gathers, variable segment loops, log-sum statistics and reduction. Fewer logical QK evaluations do not guarantee a faster implementation.
- Probe raw logits are identical on this window. Grid statistics relative L2=2.29157e-7. Attention output relative L2=3.58396e-4 (0.03584%), max absolute difference0.015625. This demonstrates an execution-induced numerical difference despite closely matching probability summaries; it does not by itself prove which individual rounding operation causes every scene-level change.

The numerical/timing diagnostic has all six quality fields unavailable and no Dense speedup, with explicit reasons. It is NOT full-forward timing or another quality run. The earlier v1 diagnostic excluded old grid aggregation from its timed wrapper and is retained for provenance; v2 is the fair same-output-contract comparison.

## Publication state

Current default code remains the evaluated compensated head-mean/single-QK implementation uploaded at502f26b8. These independent variants and all adverse findings are published alongside it; the experiment does not silently select or deploy a new default using test scores. A reasonable subsequent direction is to preserve original Key order/softmax accumulation while removing repeated QK, rather than retaining the current expensive regrouping strategy. No such additional implementation or quality claim is made here.
