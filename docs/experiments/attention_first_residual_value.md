# Independent Attention-first / residual-Value scheme

This is a new opt-in implementation, not a change to the existing manual-ratio,
Parent-region, or Dense-teacher experiments. Global blocks 0–23 are replaced;
frame/local attention, model weights, heads and output projection are unchanged.
Batch-one inference only. No Dense seed, Dense probes, manual layer-ratio table,
or ground-truth-dependent controller is used.

## 1. Total budget and current Value structure

Let N include all tokens and S include all five special tokens per frame.
The configuration specifies 24 total retention rates r. Each query retains all
special Keys and exactly B=floor(r*N)-S unique patch Keys. Default r=20%.
The Attention/Value ratios below apply to B, not to output-vector weighting.

Head-mean current V maintains F=max(B over the configured 24 layers) base fine
clusters, not just the current Value-branch budget. Layer 0 initializes uniform
frame-local raster intervals, then selects centroid-nearest real members.
Later layers perform two anchored synchronous restricted migrations and GPU
centroid-nearest real-member updates. Each fine cluster remains nonempty.
Migration candidates are the other fine IDs in its fixed 32-fine source group.
These source groups are the Parents; this is a restricted hierarchy, not global
k-medoids or a globally rebuilt nearest-neighbour hierarchy.

Per-head Keys of the current fine representatives are averaged once per Parent.
Their populations count real patch members, not just representatives.

## 2. Allocate budgets before selecting history

Sample up to 32 uniformly spaced Query groups. Each group consists of 32
consecutive real Queries; probes are its offsets 4, 12, 20, 28, clamped at the
last token. Each probe/head computes softmax(Q*pooledK/sqrt(d)+log(population));
then probabilities are averaged over heads and probes. Q is never averaged
before QK, and heads retain their own Q/K/V for final attention.

Signals, all obtained without Dense attention:

- c: mean total-variation distance between this coarse distribution and the
  real-member population prior. Flat per-Key scores give c=0.
- h: positive lift of predicted probability on the previous support pool over
  its uniform-Key mass. This uses counts of history Keys in each Parent;
  it assumes uniform probability within a Parent and is only a proxy.
- d: current fine-representative squared Value error divided by global Value
  variance, clipped to [0,1]. Constant V gives d=0.
- Previous-layer c and d, initially the current signals.

Default demands and allocation:

    A_need = c + 0.5 * h * previous_c
    V_need = 1-c + 0.5 * (d + previous_d)
    alpha = A_need / (A_need + V_need)
    K_A = clamp(round(alpha * B), 0, B-1)
    K_V = B-K_A

The implementation exposes the history and Value demand coefficients. These
are transparent heuristic scales, **not calibrated optimality or quality
guarantees**. Diffuse coarse scores favour Value; concentrated scores and
useful history favour Attention; poor Value approximation increases Value
demand. At least one residual Value representative is reserved.

Ratios are sequence- and layer-dependent but shared across Query groups at a
layer. Selection itself remains Query-group specific. This uses coarse
patch-region probabilities, not the exact Dense distribution including special
Keys. A single explicit scalar-vector GPU-to-CPU synchronization is used per
layer for dynamic buffer sizes; it is included in full-forward timing.

## 3. Attention first: history, then exploration

After allocating K_A, hard-retain the leading min(pool size, floor(0.8*K_A))
historical Keys per Query group. Budget shrinkage discards excess low-ranked
history instead of forcing the budget to expand. Layer 0 has no history.

Score current Parents with the current group's four probes, using the same
per-head softmax-then-average calculation. Rank by probability per real Key,
not by total Parent mass alone. Exclude already retained historical Keys.
Append complete ranked Parent member lists until the remaining budget is
covered. Uniformly subsample only the boundary Parent to obtain exactly K_A.
There is no local true-Key QK probe, child/fine refinement, or global token sort.
In particular, **no Value representative is preselected to exclude candidates
from Attention**. The exact boundary cap reserves the promised Value budget;
this differs intentionally from the previous whole-Parent overshoot policy.

## 4. Value after Attention: cover residual structures

A fine Value cluster is considered covered if Attention selected any real Key
in it. Other members in such a cluster are not treated as a new residual
structure merely because their token IDs differ. This is a Value-similarity
approximation; one member is not proof of exact coverage of all its Values.

For each Query group independently:

1. Mark fine clusters touched by its Attention support.
2. Partition all untouched fine IDs, in fine-ID order, into exactly K_V nonempty
   residual groups. No top-population truncation is used.
3. Compute each residual group's mean from sums and populations of **all real
   members** in its fine clusters.
4. Among the base fine representatives in that residual group, select the real
   token nearest its mean, breaking ties by real token index.

The last step is candidate-limited, not exact medoid search or an exact search
over all member tokens. Every untouched fine cluster maps to a residual group
with one selected representative. Attention and residual Value supports are
disjoint by construction. The candidate supply cannot run out: F>=B, at most
K_A fine clusters are touched, so at least F-K_A>=K_V remain.

## 5. One final attention and causal history

Union the K_A Attention Keys, K_V residual Value Keys, and all special Keys.
Run **one** per-head softmax and AV over this union, using current-layer real
K/V. Do not separately normalize or weight branch outputs. There is no added
cluster-population correction in final logits; residual coverage does not claim
to reproduce Dense AV exactly.

The verified Tensor Core 32-Query x 64-Key streaming kernel is reused. Specials
form the common bank; A+V form compact group-specific indices. No NxN mask or
full selected logits are saved.

History evidence comes from this same QK pass. Rank 64-Key support tiles by
their conditional sparse probability mass, averaged over heads and Queries;
retain their real patch members in tile order, truncating the last tile if
needed. Specials and padding never become history. This is tile-level evidence,
not token-level confidence or Dense mass. Both A and V Keys can become history,
including after a Value-only layer.

The outgoing history pool capacity is bounded by current B and 80% of the next
configured maximum Attention capacity B_next-1. The next layer allocates its
actual budget before accepting history. No next-layer feature or ratio is read.
All grouping, history and gate state resets for every new input sequence.

## 6. Cost and diagnostics

This implementation prioritizes validating the requested order. It makes no
2x speed or unchanged-quality claim. Bounded windows (default 8 Query groups)
use group-by-patch membership workspaces and group-by-fine prefix sums, not a
full token-by-token mask. Residual group-dependent Value selection costs roughly
O(number_of_Query_groups * F * d); it can still be significant at long context.
Parent ranking, dynamic sizes and one synchronization per layer are also costs.
Full initialization, pooling, gate, selection, history and reordering are timed.

Saved layer records contain signals, actual A/V counts and ratios, history
used/discarded, remaining exploration, fine/Parent counts, residual coverage,
unique density, final padded QK and coarse probe QK. These count final QK and
probe QK separately; a 20% support does not mean 20% total arithmetic.
Optional captures save compact supports and residual coverage certificates,
not rendered NxN masks. Capture is diagnostic and increases memory/time.

## 7. Run and reporting

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_attention_first_residual_value.py \
  --output-dir outputs/attention_first_residual_value/7scenes128 \
  --mode dense --scene chess --frames 128

CUDA_VISIBLE_DEVICES=4 python scripts/evaluate_attention_first_residual_value.py \
  --output-dir outputs/attention_first_residual_value/7scenes128 \
  --mode attention_first --scene chess --frames 128 \
  --config configs/evaluation/attention_first_residual_value.json \
  --baseline-result outputs/attention_first_residual_value/7scenes128/dense/chess/result.json
```

Use an idle GPU. Defaults are one warmup and three measured full forwards,
then pose, depth and unposed reconstruction evaluation. Every result includes
AUC@3/AUC@30/ACC/Comp/CD/F1, timing, peak memory and matched Dense speedup when
available. Pose-only diagnostics explicitly mark reconstruction metrics N/A.
Matching checks require identical input, checkpoint, hardware and protocol.
Two-frame smoke tests validate execution only, not 128-frame dataset quality.
Thresholds must be calibrated independently, not searched against 7Scenes GT.
