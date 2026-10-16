# Hierarchical Value + validated Attention (layer-shared)

## Current protocol: ALL 24 global layers, uniform Value-only layer 0

The user corrected the scope: the current default sparsifies **all global
layers 0--23**, not just natural L3--L9. Frame/local attention is unchanged.
Total QK density remains 10% including mandatory special keys; selection and
history remain layer-shared (head-mean Q/K/V, not per-head state).

Layer 0 uses equal-occupancy contiguous fine groups in frame-major patch-token
order (`floor(token_id*K_f/N)`). Each group's representative is its nearest real
token to the centroid. Uniformly spread fine groups supply the retained patch
keys, so equal cluster-size ties cannot concentrate the initial support at the
start of the sequence. Its Attention budget is explicitly zero, and it skips
coarse QK/reliability evaluation. Coarse grouping is still computed on its fine
representatives to seed layer 1. Layers 1--23 perform the recursive local update
and the unmodified adaptive Attention/Value algorithm. There is no dense seed.

Current config: `configs/evaluation/eval_hierarchical_value_vggt.yaml`.
Historical reproducible L3--L9 config:
`configs/evaluation/eval_hierarchical_value_vggt_l3_9.yaml`.
Current results go to `outputs/hierarchical_value/7scenes_n128_r10_all24_uniform/`.
The runner's `--sparse-start/--sparse-end` are **one-based** configuration bounds
(1 and 24); JSON also records zero-based layers to avoid ambiguity.

The full-range SAF3R comparator now uses its official lazy-TopK aggregator and
all 24 profile entries. Correspondence TopK is 128 **per frame** and now active;
it was inactive in the earlier L3--L9-only comparison. SAF3R is not matched to
the new method's fixed 10% total budget. Timing hooks cover all 24 global modules,
and each method restores the original aggregator before the next comparison.
90 relevant tests passed; one optional dependency test skipped.

The all-layer experiment and independent retained diagnostics are now complete.
Formal all-24-global-layer results (seven scenes, 128 frames each):

| Method | Global layers 0--23 ms | Overall s | AUC30 % | AUC5 % |
|---|---:|---:|---:|---:|
| Dense | 11319.962 | 19.142 | 84.293 | 36.647 |
| SAF3R, correspondence TopK128/frame | 4080.761 | 11.996 | 84.576 | 38.137 |
| Uniform-start hierarchy, 10% | 7733.934 | 16.251 | 40.393 | 9.454 |

This is a negative accuracy result, NOT a usable full-layer acceleration claim.
The layer-14 Attention budget is only 1 or 3; all other layers have K_A=0.
Layer 15 therefore discards the new history instead of reusing it.
Detailed retained tensor schema, independent synchronized timings, dense output
errors and small-input oracles: `docs/experiments/hierarchical_value_diagnostics.md`.

## Historical L3--L9 protocol and results (superseded scope)

Implements the final algorithm in the referenced “几何注意力加速策略” conversation,
with the user's subsequent changes: 10% **total** QK edges, one shared selection
per layer, not independent per-head clustering, budgets, or history.

## Implemented protocol

- Only natural global L3--L9 change; all frame attention and other global layers
  remain dense. L3 initializes directly from its current V, with no L2 PAM seed.
- Head-mean raw V determines fine Value clusters (round(0.15 * patch N)).
- Coarse count is ceil(patch N / 128). Deterministic, projection-stratified,
  anchored Lloyd clustering runs three iterations, using bounded-block FP32
  distances. This is lightweight initialization, **not optimal PAM**.
- Initial fine clusters subdivide coarse populations proportionally; all groups
  get at least one fine cluster. Two local synchronous refinements follow.
- Subsequent layers run two synchronous fine updates, considering all fine
  representatives in each token's **original previous coarse group**, fixed
  across both iterations. Previous real representative indices anchor membership
  to keep K fixed. Centroid-nearest-real-token updates representatives in O(Nd).
- Every layer reclusters its fine representatives into current coarse groups.
  Whole fine clusters move together; coarse grouping cannot split a fine cluster.
- Mean normalized/RoPE Q and mean normalized/RoPE K compute shared coarse
  attention. Pooled K is token-population weighted. Log population corrects
  unequal group sizes. No coarse matrix or weights are saved across layers.
- Layer concentration C and specificity S follow the referenced entropy formula;
  alpha=C*S and K_A=round(alpha * patch edge budget). K_V=patch budget-K_A.
- Mandatory special keys (5 per frame) are subtracted from round(0.10*N_total),
  so the **final total connection retention is 10%**, not 10% plus specials.
  Special queries use the same selection algorithm and special key protocol.
- Hamilton/largest-remainder integer allocation redistributes mass from groups
  that hit token-population capacity. Per-query quotas sum exactly to K_A.
- In overfull history groups, deduplicate by current fine cluster, then prefer
  higher fine population; do not use previous attention weights or recompute
  history QK. Otherwise retain every historical edge in that group.
- Explore uncovered current fine representatives in descending cluster population.
  The source description misses a boundary case: a group may receive more slots
  than its fine representatives. Only **after exhausting representatives**, fill
  the group's remaining slots with deterministic unused real member tokens from
  the **same** group, with no additional fine QK. This preserves the described
  token-capacity allocation, exact edge count, and uniqueness. There is no random
  fallback or quota leak into other groups.
- V fill covers highest-population fine clusters not covered by A candidates.
- Shared indices are applied to the original per-head Q/K/V. Heads/output are
  **not averaged**. A Triton online-softmax kernel computes sparse QK exactly
  once and returns those same logits, alongside multi-head output. No dense mask
  or second fine QK check is used.
- Mean exact per-head logits rank the current patch support. Only top-K_A patch
  indices become next-layer history; V edges can be promoted. The temporary
  A/V origin mask tracks provenance in each query block; only validated indices
  survive cross-layer, not weights, coarse attention, or the old whole support.
- When K_A=0, every query has the identical Value-only support and history is
  empty. This mathematically identical case uses shared gathered-K/V SDPA,
  avoiding per-query index construction and unneeded fine-logit exports.
- Full inputs reset all state. History has shape [query, K_A], no head dimension.

## Memory and performance caveats

Per-query selection/attention uses bounded query chunks (128 by default). It does
not allocate N_query*N_key dense masks or H*N_query*K full-sequence logit arrays.
Coarse attention is N_query*ceil(N/128); history can still be large if alpha is
large, and all actual sparse edges must still be evaluated. Index sorting,
allocation, GPU/host scalar synchronization, and regrouping are included in
timing; this correctness-first prototype is not a fused production selector.
Coarse representative clustering costs O(iterations*K_f*K_c*d), not O(Nd).
The once-per-sequence lightweight initialization is **included** in measurement.

## Validation

Config: `configs/evaluation/eval_hierarchical_value_vggt.yaml`.
Runner: `scripts/benchmark_hierarchical_value.py`.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python -m pytest -q \
  tests/test_hierarchical_value_attention.py tests/test_layer_value_attention.py \
  tests/test_value_representatives_gpu.py tests/test_value_clustering_gpu.py

PYTHONPATH=. /data/mmc_syang/miniconda3/envs/fastvggt/bin/python \
  scripts/benchmark_hierarchical_value.py --gpus 0 1 2 3 4 5 6
```

The 7Scenes protocol uses all seven benchmark scenes, each uniformly sampled
to exactly 128 distinct frames (stairs seq-02, others seq-01), matching the
repository reader and sampler. Compare dense, original SAF3R only on L3--L9,
and this hierarchy on the same scene/GPU/process/checkpoint. SAF3R retains its
own original profile and is **not budget-matched to 10%**. Configured topk=128
does not control L3--L9 (no DINO heads there). Pose AUC is the implemented GT
validation; no GT depth/reconstruction evaluation is implied.

Per-scene inputs, poses, warmed full-forward/module timing samples, peak memory,
per-layer C/S/alpha/budgets and promoted V-edge counts are saved under
`outputs/hierarchical_value/7scenes_n128_r10/`.

## Completed 128-frame validation

All seven scenes completed, each with 128 frames (896 images total). One warmup,
three measured forwards per method, same checkpoint/input/GPU/process per scene.
88 related tests passed; one optional-dependency test skipped. Compile checks and
`git diff --check` passed. The earlier, superseded pre-shared-fastpath partial
run was preserved in `outputs/hierarchical_value/7scenes_n128_r10_pre_shared_fastpath/`;
its timing samples are not mixed into the final report.

| Method | L3--L9 ms | Overall s | AUC30 % | AUC5 % |
|---|---:|---:|---:|---:|
| Dense | 3299.688 | 19.047 | 84.293 | 36.647 |
| SAF3R L3--L9 | 304.046 | 16.029 | 84.408 | 36.761 |
| Hierarchical Value | 1017.031 | 16.895 | 84.344 | 36.723 |

Selected-module speedup over dense is 3.244x; overall speedup is 1.127x.
Selected modules still take 3.345x SAF3R's time; SAF3R's budget is not matched.

Each layer has 132608 patch tokens, 19891 fine clusters, 1036 coarse groups,
640 mandatory special keys, 12685 selected patch keys, and 13325 total keys.
The final connection fraction is 10.000150% (integer rounding of a 10% budget).

**Important limitation:** in ALL 49 scene/layer cases, alpha=C*S rounded the
Attention budget to zero. Alpha ranges approximately 3.4e-7 to 3.5e-6 under
mean-Q/mean-K coarse attention. These runs therefore execute Value-only coverage,
with empty cross-layer Attention history and the shared SDPA fast path. They
validate the hierarchy and Value coverage, but **do not validate a nontrivial
Attention reuse/explore/history benefit on this dataset**. That branch is covered
by synthetic tests (including promotion/discard and exact returned-logit checks).
The formula was not modified and no arbitrary minimum A budget was imposed.

Full results: `outputs/hierarchical_value/7scenes_n128_r10/report.md`;
per-layer budgets/C/S are in `layers.csv`.
