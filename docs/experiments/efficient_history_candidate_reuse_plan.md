# Efficient History Reuse: Fixed Candidate Slots with Adaptive Fresh QK

## Goal

The current joint History/Exploration selector has the right optimization structure,
but it still has two efficiency problems:

1. a cluster may generate many History-prefix operating points;
2. History does not directly reduce the amount of current-layer QK exploration.

The implementation goal is therefore:

```
more reusable History
        ->
fewer fresh QK probes
        ->
lower selector cost
```

while keeping **one joint optimization objective**.  History is NOT selected first
as a final decision.

---

## Core idea

Give every current Value cluster a small fixed number `K` of **concrete relation
candidate slots**.

Without History, all `K` slots must be filled by fresh current-layer QK
exploration.

With History, reliable previous-layer relations occupy some of these slots for
free.  Only the remaining slots trigger fresh QK probing.

For cluster `C`:

```
K = relation_candidate_slots

K_H(C) = number of slots supplied by History
K_F(C) = K - K_H(C)
```

Only `K_F(C)` fresh candidates are discovered in the current layer.

This is a candidate-generation rule only.  The final support is still chosen by
the same global operating-point optimizer.

---

## Step 1: compute reusable History coverage

The current code already produces History relation gains:

```
G_h = current coarse importance
      x cross-layer persistence
      x previous within-region salience
```

After mapping History edges to the current Fine Value cluster, define cluster
History coverage:

```
coverage_H(C) = clamp(
    sum_{h in H_C} G_h / (pi_C + eps),
    0,
    1
)
```

This quantity is used only to decide how much **new candidate acquisition** is
needed.  It does not directly decide the final sparse support.

---

## Step 2: convert History coverage into candidate slots

Use a fixed small `K`, initially proposed as:

```
relation_candidate_slots = 4
```

For each cluster:

```
K_H = round(K * coverage_H)
K_F = K - K_H
```

Then:

- keep the top `K_H` History candidates by History gain;
- discover only `K_F` fresh current-layer relations.

Examples for `K=4`:

| History coverage | History candidates | Fresh candidates |
|---:|---:|---:|
| 0.00 | 0 | 4 |
| 0.25 | 1 | 3 |
| 0.50 | 2 | 2 |
| 0.75 | 3 | 1 |
| 1.00 | 4 | 0 |

This is **not a fixed History retention ratio**.  It is only a bounded candidate
construction rule.

The final optimizer can still choose:

- no concrete relation;
- one or more History/Fresh relations;
- the Value representative;
- concrete relations + representative;
- Exact.

---

## Step 3: fresh QK exploration only fills missing slots

Fresh exploration must have cost proportional to `K_F`.

Do not compute QK against every member of every cluster.

For a cluster with `K_F > 0`:

1. exclude the retained History candidate keys;
2. select a small deterministic pre-candidate set of size

```
M_F = min(cluster residual size, fresh_oversample * K_F)
```

with proposed initial:

```
fresh_oversample = 4
```

3. use spatially/uniformly distributed residual members for this pre-candidate
   set;
4. compute current QK only for these `M_F` keys using the existing 4 query
   probes of the 32-query group;
5. keep the top `K_F` fresh relations.

Therefore fresh selector QK cost is approximately:

```
fresh_probe_qk
    = 4 * sum_C M_F(C)
    ~ O(sum_C K_F(C))
```

and naturally decreases when History coverage increases.

The initial implementation should use uniform/spatial residual sampling because
it is cheap and does not introduce another expensive learned/search stage.

---

## Step 4: put History and Fresh relations on one comparable current scale

History candidates already have `G_h`.

For fresh candidates, compute current probe logits and normalize them inside the
fresh candidate set.  Assign the residual current importance:

```
residual_mass_C = pi_C * (1 - coverage_H(C))
```

Then:

```
G_f = residual_mass_C * softmax(fresh_probe_logits)
```

Thus both candidate types have the same interpretation:

```
G = current-layer relation utility mass
```

- History: inferred cheaply from previous real QK evidence + current persistence;
- Fresh: measured from current QK only where History is insufficient.

Merge the two candidate sets and sort them by `G`.

---

## Step 5: use a bounded number of joint operating points

After merging, each cluster has at most `K` concrete relation candidates.

For sorted candidates

```
e_1, e_2, ..., e_K
```

construct only:

```
Drop
Rep
EdgePrefix(1)
EdgePrefix(1) + Rep
...
EdgePrefix(K)
EdgePrefix(K) + Rep
Exact
```

Maximum state count:

```
2K + 3
```

For `K=4`, this is only 11 states per cluster.

This replaces the current unbounded state count caused by enumerating every
available History edge.

The existing `budget_states()` solver remains unchanged and chooses all cluster
states together under one global budget.

There is no sequential “select History first, then select Exploration” decision.

---

# Required code changes

Primary file:

`saf3r/patch/reuse_residual_exploration.py`

## Config

Add:

```python
relation_candidate_slots: int = 4
fresh_oversample: int = 4
```

Remove the current assumption that all History candidates become prefix states.

## Replace `history_cluster_prefixes()`

Replace it with:

```python
def build_relation_candidate_slots(
    history_ids,
    history_gain,
    history_valid,
    labels,
    pi,
    q_probe,
    k,
    patches,
    slots,
    oversample,
):
    ...
```

Outputs per query-group / Fine cluster:

```
candidate_ids      # [G, F, K]
candidate_gain     # [G, F, K]
candidate_source   # HISTORY / FRESH / INVALID
candidate_valid
history_coverage
fresh_probe_pairs
```

Responsibilities:

1. map History to current Fine clusters;
2. compute `coverage_H`;
3. allocate `K_H / K_F`;
4. retain top `K_H` History candidates;
5. probe only enough current QK to fill `K_F`;
6. merge and sort the final K concrete candidates.

## Replace `joint_history_exploration_operating_points()`

Rename to:

```python
def relation_operating_points(...):
    ...
```

It no longer needs an arbitrary History width.

It receives exactly `K` concrete candidates and creates at most `2K+3`
states.

The final solver is still:

```python
selected, count = budget_states(
    errors,
    costs,
    budget,
    dual_iterations,
)
```

## Packing

Replace the History-specific packing logic with candidate-source-aware packing:

```python
pack_relation_support(...)
```

A concrete candidate is one physical QK key regardless of whether its source is:

- HISTORY;
- FRESH.

Population compensation remains unchanged for the representative role.

---

# Important distinction: acquisition cost vs final support cost

Two costs must be recorded separately.

### Final sparse-attention cost

Every selected concrete relation costs one physical QK edge:

```
support_qk_cost = selected sparse keys
```

This remains the global budget optimized by `budget_states()`.

### Candidate acquisition cost

History candidate:

```
acquisition_qk = 0
```

because the relation was already computed in the previous layer.

Fresh candidate:

```
acquisition_qk > 0
```

because current QK probing is required.

Do NOT add acquisition QK to the final support budget.  Record it separately as
selector overhead.

This separation is essential to demonstrate the computational value of History.

---

# Required diagnostics

Add layer metrics:

```
history_candidate_slots_mean
fresh_candidate_slots_mean
history_candidate_fraction_mean
fresh_probe_qk_per_head
fresh_probe_clusters_mean
zero_fresh_probe_clusters_mean
relation_states_per_cluster_mean
```

The central efficiency test is:

```
History candidate fraction increases
        ->
fresh candidate slots decrease
        ->
fresh_probe_qk_per_head decreases
```

This relationship must be measured directly.

---

# Required ablations

Use the same joint optimizer and the same `K`.

## A. No-History fresh baseline

```
K_H = 0
K_F = K
```

Every cluster discovers all concrete relation candidates from current QK.

This is the clean baseline for answering:

> What does History save compared with directly finding several concrete QK
> edges in every current cluster?

## B. History reuse

Adaptive `K_H / K_F` from History coverage.

Compare:

- quality;
- final sparse support density;
- selector QK pairs;
- selector latency;
- total latency.

## C. History-only candidate acquisition diagnostic

Do not add fresh probes.  This is only to measure the quality ceiling/failure
mode of History candidates, not the final method.

---

# Acceptance criteria

The implementation is useful only if all three hold:

1. **Quality:** History reuse does not materially degrade the joint method.
2. **Bounded states:** operating points per cluster are bounded by `2K+3`.
3. **Real computation reduction:** compared with the no-History fresh baseline,

```
fresh_probe_qk_per_head
```

must decrease as reusable History increases.

The intended behavior is:

```
more reusable History
        ->
more candidate slots filled for free
        ->
fewer current QK probes
        ->
fewer selector FLOPs
```

If History fraction rises but fresh QK probing does not fall, the implementation
has failed its main efficiency objective.

---

# Implementation order

1. Add config and diagnostics.
2. Implement fixed K History/Fresh candidate allocation.
3. Implement residual uniform pre-candidate sampling.
4. Implement current QK probing only for `K_F > 0`.
5. Replace unbounded History-prefix operating points with bounded `2K+3`
   relation states.
6. Keep `budget_states()` unchanged.
7. Add the no-History fresh baseline.
8. Run two-frame correctness first.
9. Only then run 128-frame quality/runtime.

Do not change query-group size, History evidence granularity, Value clustering,
canonical grid, or total retention in this experiment.
