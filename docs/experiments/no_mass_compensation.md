# Probability compensation ablation and execution roles

Parent: evaluated head-mean/single-QK method in `outputs/headmean_single_qk_quality_20261102`. This is a single-factor ablation: set `disable_population_compensation=True`, bypass `history_rule.transfer`, and apply unit weights to ALL selected Keys, including L0. Keep the original losses, cluster update, History qualification, Exact eligibility, budget, head-mean estimator, probe count and attention backend. Remove the inapplicable mass-conservation diagnostic rather than claiming a zero error. Do not retune on GT.

Reproduction: `scripts/setup_no_mass_compensation.py`, `scripts/run_no_mass_compensation_quality.py`, `scripts/report_no_mass_compensation_quality.py`. Raw results and immutable source archive: `outputs/no_mass_compensation_quality_v2_20261102`. Only this changed method is evaluated; Dense, SAF3R and the parent are reused. Default retained parent is not overwritten.

The first attempt failed before producing any GT metrics because the bypass left the per-window weight variable uninitialized. It is retained under `outputs/no_mass_compensation_quality_20261102`, with all new-run scores unavailable and concrete exception reasons. The corrected two-frame, 24-layer preflight monkeypatched transfer to raise if called and passed with zero calls: `outputs/no_mass_compensation_preflight_20261102/validation.json`. No GT-based parameter selection occurred.

## Producer versus consumer: do not count the same operation twice

At layer l, the consumer takes the preceding layer's cached conditional statistics and the CURRENT layer's coarse-grid observation, then computes current probability estimates used in selection. After executing the selected attention, the producer estimates layer-l Dense normalization/grid mass and caches head-mean conditional statistics for layer l+1.

```
producer(l-1) -> cached conditional statistics
                         + current coarse observation(l)
               -> consumer(l) -> state choice -> sparse Attention(l)
               -> producer(l) -> cached statistics for l+1
```

The producer does not know the next layer's probabilities. It is incorrect to describe producer(l) as already computing those probabilities. L0 has no consumer; L23 has no producer. For heads128 there are 66 windows per layer: producer has 1518 calls on L0–22, consumer has 1518 calls on L1–23. They are different non-overlapping executions. Never charge the same producer interval a second time to the next layer's consumer.

The old 1.374212s segment-reduction kernel was wholly inside the producer. The old history-update operation was inside the consumer and performed a different computation. The earlier exclusive audit has zero duplicated intervals. Actual redundant computation still exists elsewhere: legacy coarse/Exact eligibility and M5c coarse observation remain separate pathways (including different numeric/probability conventions). This is execution duplication, not a duplicated reporting category. It cannot be removed merely by deleting a table row.

New changed-method heads128 diagnostic: `outputs/no_mass_compensation_profile_20261102`. Producer total2.523047s, consumer0.714658s, core sparse Attention/probe/grid fused kernel9.321678s. Only add independent exclusive rows; do not add parent and child scopes. Full instrumented forward26.982962s; global Attention total22.503277s; exclusive rows sum exactly22.503277s. Six quality fields and Dense speedup are N/A for this diagnostic, with reasons in raw JSON.

## Cluster cost and possible simplifications

In the new diagnostic all cluster operations together take1.066034s (11.44% of the fused core, or10.12% of the complete attention wrapper10.537230s). Its children are NOT additional costs to add: neighbor-center graph0.446518s; two local passes0.400050s; L0 two global searches0.168358s; empty-cluster repair0.014831s; centroid reductions0.009864s; remainder0.026413s.

Centers and memberships still update every layer. Candidate sets include the current center among32 candidates. The neighbor graph currently uses a global center-distance search; local token assignment alone does not make graph construction local. Simplification directions, not changes in this ablation:

- Equivalent engineering: batch diagnostic scalar extraction, remove unused local-mode certificates, fuse gathered distances/reductions.
- Approximation: refresh local memberships using current centers but reuse a bounded neighbor-candidate graph, or build approximate center neighbors without all center pairs. This does not freeze the cluster or Rep identity but changes candidate coverage and needs validation.
- One rather than two local passes is an algorithmic approximation and needs an independent evaluation.

## History membership, loss and ownership

For query group g and cluster c, qualified Keys are preceding-layer executed history members whose estimated current probability is above pi[g,c]/population[c], plus any historical Key equal to the current Rep even when it does not pass that threshold. The current method retains ALL qualified history members; the legacy `history_per_cluster=4` field is not an active Top4 truncation in this state implementation.

Nearest means minimum squared L2 distance between CURRENT layer RMS-scaled head-mean Values within that cluster, not physical/image distance or Key-logit similarity. The History-state loss is sum_j prior[g,j] * min_h ||z[j]-z[h]||^2 over cluster members and qualified history h. A selected history Key represents itself (zero distance). High-probability members therefore matter more.

`owners[g,j]` is the representative history Key assigned to member j, separately per query group. It is used after choosing the History state to aggregate j's probability to that history Key. It is not a persistent cluster label or a cross-layer head match. In this single-factor ablation the unchanged loss kernel still emits owners, but transfer no longer consumes them; omitting this now-unused output is a possible equivalent implementation optimization, not part of the measured ablation.

## Which operations construct the mask?

- Current-method selection inputs: cluster update, current coarse evidence, incoming history transfer, qualification, state losses, Exact eligibility and budget solving. These are required by THIS selection rule, not by sparse attention in general.
- Selected-support packing: converts chosen states to executable Key indices; some index organization is common to SAF3R too, but per-token expansion/prefix sums are an implementation choice.
- Mass transfer/logit compensation: modifies attention weights after support choice, not mask topology. This ablation removes it.
- Outgoing Dense probability calibration/conditionals/cache: prepares the next layer's mask evidence; not required to finish this layer's output. Omitted at the final layer.
- FP16/u14 packing: storage optimization, not part of the mathematical state rule. With head-mean evidence already fitting on GPU, plain FP16 may avoid codec work at the price of a small cache increase; not changed here.
- Diagnostics/conservation checks: validation and reporting, not mask-selection equations. Can be accumulated on GPU and read in batches.
- QKV, Q/K normalization, RoPE and output projection: original model operations, not mask construction.

The locally compared SAF3R lazy first-correspondence Top4 path obtains correspondence indices from its first relevant head and propagates them to later blocks under a fixed head-pattern profile. It still has candidate selection/index organization and sparse attention, but does not implement our Value clusters, Drop/Rep/History/Exact losses, dual cluster-budget allocation, M5c cross-layer calibration or probability-mass transfer. This is a local implementation comparison, not an assertion about all SAF3R variants or the paper's DINO settings.

## CPU involvement

QK, Value distances, scatter/gather, reductions, Top-K and Triton kernels execute on GPU. Python loops/control launch them from CPU. `float(tensor)`, `int(tensor)` and `bool(tensor.any())` copy GPU scalars and synchronize; `tensor.float()`/`tensor.int()` are device-preserving dtype conversions, not CPU calculations. Safety checks and logging create many synchronization points; this is avoidable critical-path scheduling overhead rather than large matrices being evaluated on CPU. The head-mean method has no CPU probability cache or probability-bank PCIe transfer. This ablation does not mix in unrelated clustering/check optimizations.

## Completed ablation results

All18 scenes passed the frozen-source/input/protocol/GPU-overlap checks. All432 layers have compensation disabled, no second QK and no CPU history bank. The original parent was not rerun or overwritten. Quality is one evaluation per scene; no retuning.

|Dataset/variant|AUC@3 %|AUC@30 %|ACC m|Comp m|CD m|F1 %|Mean forward s|Peak allocated GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|7scenes/ready_previous|22.64190|84.43857|0.14410|0.12298|0.13354|46.12994|27.77073|17.21150|
|7scenes/no_mass_compensation|21.05479|83.70946|0.14451|0.13880|0.14166|46.17526|26.28885|17.20946|
|eth3d/ready_previous|22.34480|78.41895|0.35583|0.89430|0.62507|54.91852|4.47692|11.23081|
|eth3d/no_mass_compensation|16.03639|77.46168|0.42201|1.00321|0.71261|45.94018|4.54003|11.22896|

Parent timings are historical, not paired. New Dense speedup is N/A. 7Scenes uses7 scenes of128 frames; ETH3D uses11 scenes of14–76 frames. Removing compensation hurts 7Scenes AUC/Comp/CD and severely hurts ETH3D pose/reconstruction. It does not demonstrate stable end-to-end speed improvement. Keep as an ablation, not a quality-preserving replacement.
