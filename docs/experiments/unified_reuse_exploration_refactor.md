# Unified History Reuse + Current-Layer Value Exploration Refactor

## Motivation

This branch refactors the high-quality `reuse_residual_exploration` prototype
rather than starting from a new sparse-attention method. The objective is to
preserve components that already worked empirically while correcting the parts
whose semantics conflict with the current design principles.

The final sparse-support problem is viewed as one shared budget allocation:
reliable old relations and current-layer Value representatives compete jointly.
History is not hard-kept by a fixed fraction or threshold, and exploration is
not a second stage executed after History.

## Principles

1. The inaccessible ideal target is the Dense attention output. We do not invent
   a per-cluster output target such as `pi * mean(V)`.
2. Previous-layer attention probabilities are valid only in the previous layer.
   They may rank old genuine edges by historical salience, but are never treated
   as predicted current-layer attention mass.
3. Cross-layer coarse attention is used to judge persistence of the old relation
   prior. The same canonical estimator must be compared across adjacent layers.
4. Current coarse attention answers **where current information matters**.
5. Current Value geometry answers **which keys can represent redundant current
   information**.
6. History and exploration share one budget. The number of retained History
   edges is an output of the selector, not a preset ratio.
7. Once a History key is selected, its current-layer Value is also an anchor and
   therefore changes the value of further exploration.
8. No explicit residual of the form
   `current coarse mass - old edge probability` is constructed.
9. Support construction must stay much cheaper than the final sparse attention.

## What was wrong in the previous prototype

### Artificial cluster-output target

The previous operating-point loss used

```
target_C = pi_C * mean_V_C
```

and evaluated Drop / Rep / History / History+Rep / Exact by their distance to
this artificial vector target. This makes the representative artificially
favored and, more importantly, lets a History edge be retained because it
happens to approximate the current cluster mean. That is not the intended
semantics of History.

### Cross-layer probability transfer

The previous History utility was numerically reused as a predicted current
contribution. Old attention probability can rank old edges, but its absolute
value is not a current-layer probability because Q/K, support, and the softmax
denominator all change.

### Residual subtraction

The previous coarse demand used

```
q = [p_current - beta * previous_genuine_mass]+
```

which subtracts quantities that do not live in the same current-layer
probability space. This residual is removed.

### Fine representative QK probing

The top 25% residual Fine clusters previously received extra representative QK
probes to estimate `pi_hat`. This adds mask-construction cost and couples the
selector to another fine-grained attention search. The refactor removes this
entire branch.

### Mismatched cross-layer coarse estimators

Current `p` came from pooled canonical QK while the saved previous
distribution came from the executed sparse support aggregated by key location.
The refactor stores the same canonical coarse estimator every layer, so
persistence compares like with like.

### Full repeated evidence QK

The previous implementation recomputed QK on all selected patch keys after the
final sparse attention in order to obtain outgoing History salience. The
refactor replaces this with a 4/32-query probe evidence pass. It remains an
explicit approximation, but its QK cost is one eighth of the previous evidence
pass and it is used only for next-layer relative salience.

## What is preserved

The following components are intentionally retained:

- FullCoverageValue and its cross-layer local Value updates.
- Current real-token representatives.
- Fixed canonical low-resolution observation.
- Query groups of 32 with four probe queries.
- Population compensation for Value representatives.
- Genuine-edge-only History bookkeeping.
- The operating-point representation
  (Drop / Rep / History prefix / History+Rep / Exact).
- The shared Lagrangian budget solver, so History/Rep/Exact ratios remain
  adaptive outcomes rather than scheduled hyperparameters.
- One unified sparse softmax/AV execution.

This keeps the empirically successful architecture around the selector while
changing the semantics of the selector itself.

## Refactored common-scale objective

For Fine cluster C, current coarse attention is mapped to a current importance

```
pi_C
```

with no subtraction of History.

Current Values are unit-normalized. For a selected anchor s, the Value
representation distortion of cluster C is the mean cosine distance

```
D_C(s) = mean_{j in C} (1 - cos(V_j, V_s)) / 2
```

which is available cheaply from the mean of unit-normalized member Values. The
real cluster representative has a precomputed `D_rep`. A selected History
edge is evaluated with its **current** Value, so it can also act as an anchor.

For genuine History edge e, the reuse prior is

```
r_e = previous_salience_e * persistence_region(e)
```

where `previous_salience_e` is used only as a previous-layer ranking signal
and `persistence` comes from adjacent-layer canonical coarse distributions.
The product is a reuse prior, not a current attention probability.

For a History prefix, relation survival uses diminishing returns

```
R = 1 - product_e (1 - r_e)
```

and the common bounded cluster loss is

```
L_C = pi_C * D_C * (1 - R)
```

Interpretation:

- Drop: `D=1, R=0`.
- Rep: reduces current Value distortion only.
- History: receives an independent relation-persistence benefit and its current
  Value becomes an anchor.
- History+Rep: combines both roles.
- Exact: zero representation loss, but keeps its full QK cost.

All operating points are therefore compared in the same current-importance
scale without constructing a fake output vector or predicting current edge
probability from an old one.

## Exact-state eligibility

Exact expansion is still retained because it was important to the quality of
the previous prototype. It is no longer enabled by residual Fine-QK probes.
Instead only the top `candidate_fraction` clusters by **current coarse
importance** are eligible for Exact. The field name is retained for config
compatibility; it no longer controls any fine-level QK probing.

## Expected computational changes

Removed per layer:

- residual Fine-cluster mapping and residual top-k;
- QK probes on 25% of Fine representatives;
- representative-mass estimation and `pi_hat`;
- full second selected-QK evidence pass;
- previous genuine-grid residual state.

Still present:

- one tiny canonical coarse QK observation;
- Value update / clustering;
- History candidate ranking;
- operating-point construction and dual budget search;
- final sparse attention;
- 4/32-query probe evidence for next-layer historical salience.

The refactor is intended as a direct replacement candidate for the current
high-quality prototype, not a new method family. History80 and the original
prototype remain available from earlier commits for controlled comparison.
