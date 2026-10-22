# Reuse + residual exploration: independent reference prototype

## Status

New opt-in entry `install_reuse_residual_exploration`, not a replacement or
mutation of History80, Attention-first, or the failed cross_layer_budget proxy.
Config: configs/evaluation/reuse_residual_exploration.json. No7Scenes truth is
used to select splits, thresholds or this config. Initial correctness tests:
6 CPU/CUDA tests, plus5 reporting tests, passed. A real2-frame heads diagnostic
also passed all24 global layers. The seven-scene128-frame QUALITY evaluation
has completed all28 matched runs; a two-frame diagnostic is not its result.

Completed results: `outputs/global_attention/reuse_explore_20261021/quality128/report.md`
and `summary.json`, with raw per-run metrics, exact inputs and frozen source
snapshots. The new prototype achieves macro AUC@3 22.2060%, AUC@30 83.8300%,
ACC0.121830m, Comp0.112915m, CD0.117373m and F1 50.4427%.
Matched Dense achieves23.0420%,84.2927%,0.148037m,0.124358m,0.136197m
and48.0537%, respectively. Mean scene median complete-forward time is116.742s
versus Dense17.118s (0.1466x speedup); mean scene peak allocated memory is
15.495GiB versus14.359GiB. AUC@30,2x speed and memory gates FAIL; average
reconstruction improvement does not hide per-scene regressions. This is a
reference prototype, not a successful acceleration release.

## Interpretation of the referenced conversation

The final conversation defines reuse utilities but explicitly leaves their
comparison with exploration utilities as a design question. This implementation
chooses one concrete engineering prototype, not an allegedly fully specified
optimal algorithm from that conversation.

There is NO layer A/V split table, fixed history fraction, head routing,
cross-head matching, Dense seed, exact Dense Top-K selector or Dense fallback.
All24 global layers use this entry; frame/local attention is unchanged.
L0 is the spatially initialized population-corrected Value representative seed.

The total patch cost cap is `floor(0.2*N)-specials`, with specials always retained
and counted in20%. Discrete region choices may leave unused budget. This is
therefore **AT MOST20% final edges**, not a promise of exactly20% or20% total
arithmetic. No oversized exact cluster is silently cut into individual tokens.

## Partitions and state

1. Fixed canonical grid:2x2 spatial cells per frame, preserving frame identity,
   unchanged from L0 to L23, independent of dynamic Value labels.
2. Dynamic fine Value partition: arithmetic16-head mean of raw V,64-D;
   base cluster count equals the maximum patch budget. Use the existing
   FullCoverageValue spatial initialization,2 synchronous source-restricted
   updates and centroid-nearest REAL member representatives.
3. Query groups: fixed32 neighboring Queries,4 real probe Queries per group;
   each final Query/head still has its own QK weights and output.

Persist per group:

- Genuine explicitly computed Key IDs plus FP16 **group-mean conditional patch
  probability**, not H x Q x K probabilities and not frozen current-layer
  weights. Population-weighted Value representatives cannot become history.
- FP16 canonical probability of the actual EXECUTED sparse support, aggregated
  by retained Key location and normalized over patch Keys (specials excluded).
- FP16 canonical probability contributed by genuine explicit edges only.

The first canonical distribution includes representative probabilities at their
actual retained Key locations. If a representative stands for members spanning
canonical cells, it does NOT redistribute that probability to those members.
It is an exact aggregation of the executed conditional sparse distribution,
not Dense region mass or an exact physical-population-region estimate.
This mismatch with current pooled-Key coarse estimates is a measurable proxy
limitation, not hidden under the word "true".

For128 frames with133248 total tokens,4164 Query groups and512 regions,
one FP16 grid matrix is4.26 decimal MB; the two saved matrices total8.53 MB.
The history bank has group x patch_budget int32 IDs and FP16 masses; its
capacity cost is about0.65 decimal GB, NOT4MB. Both are reported at runtime.

## Current observation and reuse

Pool all real patch Keys once into the FIXED canonical grid. For each group's
4 probes, perform per-head softmax with canonical population correction, then
average heads/probes to obtain current p[g,r]. Do not average Q/K across heads.

Previous P[g,r] is from the executed sparse pass. Region stability is
`beta = min(1, p/P)` when P>0, otherwise0. Current utility of a genuine previous
edge is `u_e = beta[grid(e)] * previous_probability_e`. No exact current QK is
used to validate these historical edges before final attention.

Re-map history IDs to CURRENT fine labels. Retain up to4 highest-utility
candidates per current fine cluster for bounded state enumeration. This is an
explicit computational approximation/cap, not an80% reuse policy. Exclude the
current fine representative from history candidates so a reuse+rep state never
duplicates that Key. Record cap-discarded counts. A current pure-rep layer
emits no genuine history.

## Residual localization and representative-mass estimation

The coarse demand used to localize probes is
`q = [p - beta * previous_genuine_grid_mass]+`.
It is the residual after the POTENTIAL history evidence, not after the final
joint selection. Actual selected reuse can be smaller. No iterative second
localization pass is hidden in this prototype.

Build a sparse current fine-cluster/canonical-cell overlap table from token
membership. It maps canonical mass to fine mass:
`pi_C = sum_r p_r * |C intersect grid_r| / |grid_r|`.
Map q the same way. Probe real representatives only for the highest-demand25%
of current fine clusters per group. This fraction controls estimator cost,
NOT the final A/V allocation. Outside this candidate set, estimate representative
mass using the coarse proxy; this can miss sharp sub-region logits.

For candidate clusters, evaluate the CURRENT representative QK on4 probes per
head, add log(cluster population), compare against the coarse log-partition,
average probabilities, then jointly renormalize estimated masses over all fine
clusters. Thus representative mass `pi_hat` is NOT assumed equal to pi as in
the previous failed Value-only distortion proxy.

## Common-scale operating points

Current fine means are head-mean V vectors. Let target contribution be
`t_C = pi_C * mean_V_C`. This is a low-resolution surrogate for true
attention-conditioned contribution, not an exact Dense output.

Each cluster offers:

- drop: cost0, error `norm(t_C)`;
- representative: cost1, error `norm(t_C - pi_hat_C*V_rep)`;
- reuse t edges, t=1..4: cost t, error against the sum of predicted historical
  contributions using CURRENT V at those edge IDs;
- reuse t edges + representative of residual n-t members: cost t+1, predicted
  residual representative mass `pi_hat_C * (n-t)/n`;
- exact expansion: cost n, proxy error0, eligible only in fine-probed regions.

Historical predicted masses inside a cluster are jointly capped at pi_C to
avoid double-counting the same regional mass. Complete expansion emits every
real cluster member and removes separate representative/reuse emissions there.
All gains live in vector-output norm units; no separate hand-set A/V utilities.

Budget solver: per group, independently minimize
`error(state_C) + lambda * cost(state_C)` over cluster states;16 bisections find
a feasible total cost <=B. This is a separable Lagrangian approximation to
discrete rate-distortion/knapsack, NOT an exact global optimizer. It can leave
unused budget, recorded rather than filled by undocumented fallback edges.
The exact/reuse/representative fractions are outputs, not inputs.

## Final attention and population accounting

Pack chosen states into compact group indices, never a full N x N Mask.
No duplicate Keys: reuse candidates exclude current rep, and exact-state
clusters emit only their full members. Genuine edges carry unit population;
the representative in a reuse+rep state carries n-t, not n. Dropped members
are intentionally unrepresented unless that state contains a residual rep.

One unified per-head/per-Query softmax/AV includes specials, explicit Keys and
population-corrected representatives. There are no separately normalized branch
outputs and no output mixing coefficients. Each layer recomputes final QK.

Tensor Core online attention returns output/LSE. A **SECOND selected-QK pass**
aggregates per-Key probabilities over heads/32 Queries into group x selected
Keys. It avoids massive retained per-head logits but duplicates sparse QK.
This is explicit validation overhead included in full-forward timing, NOT a
deployment-speed claim. It must eventually be replaced/approximated/benchmarked
if the mechanism proves useful. Never call final20% edges20% computation.

## Validation and evaluation

Tests compare all24 CPU and BF16 CUDA outputs against explicit population-
weighted attention with identical supports, including tails, padding, uniform
softmax, budget feasibility, duplicates, population n-t, genuine-only history,
constant canonical coordinates and state reset. Additional tests verify real
mass-based stability, no fixed history count and drop behavior at zero gain.

Real2-frame diagnostic: outputs/global_attention/reuse_explore_20261021/
diagnostic_heads2. Initial reuse COUNT logging there had a missing reduction
over the4 history-candidate slots. Actual supports/outputs were unaffected;
the counter was fixed and a sum-of-three-modes invariant added BEFORE freezing
the128-frame evaluation. Do not use its old reuse_edges_mean without correction.

Frozen128-frame queue: outputs/global_attention/reuse_explore_20261021/quality128.
Four methods per each of seven scenes: matched Dense, repository SAF3R lazy
correspondence Top-4, original History80+population compensation, this prototype.
Each has one warmup and3 measured complete forwards. Same exact frame lists,
checkpoint, preprocessing, pose and recon_unposed settings; all initialization,
clustering, pooling, routing, history and evidence costs count. No per-layer
timing synchronization. Quality evaluation follows AFTER model/GPU release.
Seven GPUs each own one scene; at most one timed GPU process per card, with a
second CPU-quality process allowed. No constant human monitoring required.

Every run writes six-metric quality.json/quality.md, raw metrics, input/config,
source snapshots, layered decisions, timing samples and peak allocated/reserved
memory. Incomplete/OOM runs retain concrete N/A reasons; no frame/budget changes.
Report pose AUC@3/AUC@30, reconstruction ACC/Comp/CD/F1 (metres,0.05m F1 threshold,
0.0078125m downsample,CD=(ACC+Comp)/2 unsquared), and paired Dense speedup.
Macro results require all7 scenes. SAF3R is explicitly NOT the paper DINO Top-K
configuration. No published/local score shifts or paper-H100 speed comparison.

Quality, >=2x speed and memory gates are separate. Depth AbsRel is not evaluated
in this experiment, so complete historical acceptance is not claimed.
Failure of the prototype is reported as such; no test-truth-driven retuning.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
 /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
 scripts/run_reuse_exploration_eval.py \
 --output-dir outputs/global_attention/reuse_explore_20261021/quality128
```

Do not launch a duplicate while the coordinator is running. report-only mode
regenerates tables from completed files without running or modifying models.
