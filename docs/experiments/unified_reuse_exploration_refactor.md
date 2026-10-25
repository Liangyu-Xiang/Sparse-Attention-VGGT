# Unified History Reuse + Current-Layer Value Exploration Refactor

## Goal

This branch refactors the empirically stronger `reuse_residual_exploration`
prototype instead of introducing a new sparse-attention method from scratch.
The final sparse mask is still obtained under one shared QK budget, but History
reuse and current-layer exploration now have deliberately different roles.

History is a cheap relation-reuse mechanism. Exploration is a current-layer
representation mechanism. They compete for the same budget; neither receives a
preset fraction.

## Core principles

1. The unavailable ideal target is the Dense attention output. Do not fabricate
   a per-cluster output target such as `pi * mean(V)`.
2. Previous-layer attention probability is valid only in that layer. It can
   provide relative salience among already-computed historical relations, but
   it is never interpreted as the current-layer edge probability.
3. Current coarse attention supplies current-layer importance.
4. Current Value geometry supplies current-layer redundancy for exploration.
5. A historical edge must be reusable directly by identity. It should not first
   be remapped into a Fine cluster and then re-evaluated as if it were another
   local Value candidate.
6. History and exploration share one budget. The final History count is an
   optimization outcome rather than a threshold or fixed percentage.
7. No explicit residual of the form
   `current coarse mass - previous edge probability` is constructed.
8. Mask construction must stay substantially cheaper than the final sparse
   attention.

## Problems in the original high-quality prototype

### Artificial cluster-output target

The previous selector used

```
target_C = pi_C * mean_V_C
```

and judged Drop / Rep / History / History+Rep / Exact states by their distance
to this fabricated target. This can retain a History edge because its Value
happens to resemble the cluster mean, which is not the reason History exists.

### History was made cluster-local again

The previous implementation remapped every old History ID into the *current*
Fine Value partition, sorted History inside each Fine cluster, kept at most
`history_per_cluster=4`, and then enumerated History prefixes as cluster-local
operating points.

That largely defeats the purpose of History reuse. A previously computed QK
relation should be cheap because its identity and old salience are already
known. Treating it as "a few more local candidates to test inside every Value
cluster" makes History behave too much like another exploration mechanism.

The refactor removes this remapping/cap from History selection entirely.

### Cross-layer probability transfer

Old attention values are no longer used as a prediction of current attention
mass. They are only historical salience evidence.

### Residual subtraction and Fine QK probing

The old residual

```
[p_current - beta * previous_genuine_mass]+
```

and the subsequent top-25% Fine representative QK probe are removed.

### Mismatched coarse estimators

Adjacent-layer persistence now compares the same canonical pooled-QK estimator
on both layers, instead of comparing current pooled-QK probability against a
previous sparse-support aggregation.

### Full repeated evidence QK

The original reference implementation recomputed almost the complete selected
sparse QK support after attention solely to recover next-layer History weights.
The refactor estimates outgoing relation salience on the same four probes used
by coarse routing, reducing this evidence QK from 32/32 queries to 4/32 queries.

## Preserved components

The following parts of the high-quality prototype remain:

- FullCoverageValue and its local cross-layer Value update;
- real-token Fine representatives;
- fixed low-resolution canonical observation;
- 32-query groups with four probes;
- population compensation;
- one final unified sparse softmax/AV;
- current Value Drop / Rep / Exact operating points;
- the Lagrangian budget search.

The point is to replace the selector semantics while preserving the parts that
already produced strong reconstruction quality.

## Direct History relation pool

History is now defined exactly as the previous layer's **executed sparse patch
support**.  If a patch key participated in QK in layer l-1, it is a History
candidate in layer l, independent of how it entered the previous mask:

- Value representative;
- Exact member;
- previously reused History edge.

This definition deliberately separates **where an edge came from** from
**whether the relation was actually computed**.  Rep/Exact are construction
mechanisms; History is an execution record.

For each previous executed edge e in canonical region r, its previous measured
patch attention is divided by its represented population to obtain a
single-edge salience proxy.  These proxies are normalized only within the old
region:

```
s_e = old_single_edge_salience_e / sum_{j in H_r} old_single_edge_salience_j
```

The current layer supplies its own coarse region importance `p_r`, and the
adjacent-layer comparison of the same canonical estimator supplies persistence
`beta_r`.

The History relation gain is therefore

```
G_H(e) = p_r(current) * beta_r * s_e(previous)
```

There is **no explicit-relation coverage gate**.  Coverage was removed because
it depended on which edge types the previous selector had already labeled as
"genuine", creating a bootstrap loop.  Sharpening is instead expected to emerge
implicitly:

1. early layers expose a broad previous support;
2. old low-salience edges receive small individual gains;
3. current coarse changes suppress stale regions through `beta_r` and
   `p_r(current)`;
4. retained edges are re-evaluated by current QK;
5. their new measured salience is used in the next layer;
6. new Rep/Exact exploration edges also join the next History bank automatically.

Thus a relation may become progressively stronger or disappear without moving
the previous absolute softmax probability into the current layer.

History IDs are never remapped into Fine clusters before scoring.  Each
historical relation is a two-state global item:

```
drop : cost 0, loss G_H(e)
keep : cost 1, loss 0
```

History and exploration still share the same total budget.  The number of
selected History edges is an optimization outcome, not a fixed ratio.

## Current-layer exploration

Fine cluster C receives current importance `pi_C` by mapping the current
canonical distribution through cluster/canonical overlap.

The current implementation retains the cosine-Value distortion used by the
first refactor:

```
D_rep(C) = mean_{j in C} (1 - cos(V_j, V_rep(C))) / 2
```

This metric choice is intentionally treated as an ablation-level detail rather
than a core principle. Raw-L2 or norm-aware alternatives can be evaluated
without changing the overall selector.

Exploration has only current-layer states:

```
Drop : error pi_C, cost 0
Rep  : error pi_C * D_rep(C), cost 1
Exact: error 0, cost |C|
```

Exact remains restricted to the top `candidate_fraction` clusters by current
coarse importance to preserve the high-quality prototype's expressive range
without reintroducing Fine-QK probing.

## Shared budget competition

History relation items and Fine Value exploration items are concatenated into
one state table and passed to the same Lagrangian budget solver.

Therefore the final mask can automatically contain:

- many History edges when old relations remain valuable;
- more new representatives when current Value coverage matters more;
- Exact expansions for a small number of current high-importance clusters.

There is still no History-first fixed allocation and no fixed History ratio.
The distinction is that History enters this competition *directly as relations*,
not as cluster-local Value candidates.

## Population accounting after selection

Only after a History edge has been selected is its token mapped to the current
Fine label for packing/population accounting.

This mapping is not used to decide whether the History edge is valuable.

If a selected History key happens to equal the current Fine representative, the
two physical roles are merged into one sparse key and their represented
population is combined. Exact expansion suppresses duplicate History members.

For a merged History+representative key, outgoing next-layer relation salience
uses `selected_mass / population` as a per-member approximation, preventing
aggregate population mass from being mistaken for genuine edge probability.

## Compact History state

The original prototype stored every query group with the full patch-budget
History width, producing a large padded History bank.

The refactor stores all executed patch relations from the previous layer and
removes only invalid padding. No Fine-cluster regrouping is used. Because the
full previous support can be close to the sparse budget width, this version can
increase History memory and selector work; the next experiment explicitly
profiles that cost before any efficiency optimization.

This directly reduces:

- History memory;
- invalid padded candidates scanned by the next-layer budget solver;
- the computational cost of making History an independent relation pool.

## Computational path removed

The refactored selector no longer performs:

- previous/current probability residual subtraction;
- residual top-k Fine localization;
- Fine representative QK probes;
- `pi_hat` representative-mass estimation;
- per-Fine-cluster History sorting and cap;
- full repeated selected-QK evidence.

The intended mask-construction path is now:

```
current canonical QK
        |
        +--> current Fine importance + current Value exploration states
        |
previous compact History bank
        +--> direct relation gains
        |
        v
shared dual budget
        |
        v
population-aware sparse support
        |
        v
one sparse attention
```

This is still a refactor of the existing high-quality prototype, not a separate
method family.

## Next diagnostic round

The executable diagnostic package and Codex handoff for the next round are in
`docs/experiments/unified_next_round_handoff.md`.  Run those diagnostics before
making another production-selector revision.

## Physical History–Rep budget accounting

The all-executed-edge experiment exposed a large optimizer/packing mismatch: a
physical key that was both previous History and the current Value representative
was charged twice by the independent state solver and deduplicated only after
selection.  The current branch contains a causal accounting ablation that merges
those roles before optimization while leaving History scoring and Rep/Exact
semantics unchanged.

Execution handoff:
`docs/experiments/physical_history_rep_budget_handoff.md`.

## Efficient History candidate reuse follow-up

The next implementation step is documented in:

`docs/experiments/efficient_history_candidate_reuse_plan.md`

The key change is to cap each Fine cluster at a small fixed number of concrete
relation candidate slots.  Reusable History fills those slots without current
QK probing; only the remaining slots trigger fresh current-layer QK discovery.
The final History/Fresh/Rep/Exact decision remains one joint global budget
optimization.
