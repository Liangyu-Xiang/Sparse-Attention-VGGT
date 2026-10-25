
# Efficient History Reuse: History Relations + Representative + Exact Fallback

## Goal

The selector must exploit History without introducing a second current-layer
fine-QK search.

The key rule is:

    Concrete QK relations can come only from History.

For the current layer, a Fine Value cluster has only three information sources:

1. History relations: concrete QK edges that were actually executed in the
   previous layer and are reused as candidate relations;
2. Value representative: one current cluster representative that approximates
   the residual members;
3. Exact expansion: if History + representative are still insufficient, expand
   the whole cluster.

There is no fresh per-cluster QK discovery.

The final decision remains one joint global operating-point optimization. There
is no sequential "select History first, then select Exploration" solver.

---

## Core difference from a no-History method

Without History, this method does NOT try to discover several specific current
QK edges inside every cluster. The no-History compressed choices are only:

    Drop
    Rep
    Exact

With History, previously executed QK edges become the only source of concrete
relations:

    previous executed QK
            ->
    History relation candidates

If History relations plus one Value representative can preserve the cluster, the
cluster remains compressed.

If they cannot, the method does not search for additional specific QK edges:

    History + Rep insufficient
            ->
    Exact cluster expansion

This gives History a clear role and keeps the selector cheap.

---

## Per-cluster joint operating states

For a current Fine cluster C, map all available History relations into the
cluster and rank them by the existing History reliability score G_h.

Let H_t denote the top-t reusable History relations in C.

The joint state family is:

    Drop
    Rep
    H_1
    H_1 + Rep
    H_2
    H_2 + Rep
    ...
    H_t
    H_t + Rep
    ...
    Exact

All states are compared inside the SAME budget_states() optimization.

Important:

- H_t is not selected in a separate first-stage solver;
- there is no current-layer fine-QK search to create new concrete edges;
- Exact is the only fallback when compressed states are insufficient.

---

## What History is allowed to do

History provides two things.

### 1. Concrete relation preservation

A retained History edge is a real previous-layer QK relation. Its usefulness is
estimated from:

    current coarse importance
    x cross-layer persistence
    x previous measured relation salience

Current Value similarity must not invalidate a reliable History relation.

### 2. Reduce the need for Exact expansion

History should make the compressed state more expressive.

For example, Rep alone may be insufficient for a difficult cluster, while:

    History_1 + History_2 + Rep

may preserve enough information to avoid:

    Exact(C)

The computational benefit of History therefore comes primarily from replacing a
large Exact state with a small number of reused relations plus one
representative.

---

## Efficiency target

The desired property is NOT:

    more History -> fewer fresh QK probes

because fresh per-cluster QK probing does not exist in this design.

The desired property is:

    more useful reusable History
            ->
    more clusters remain compressible
            ->
    fewer clusters choose Exact
            ->
    fewer physical QK edges are executed

For a cluster of size |C|:

    History_t + Rep cost ~= t + 1
    Exact cost            = |C|

When t + 1 is much smaller than |C|, useful History avoids an expensive Exact
expansion.

Therefore the key efficiency metric is the reduction in Exact expansion and
physical support.

---

## Controlling the number of operating points

The current implementation can still become expensive if every available History
prefix creates a separate operating point.

Do NOT solve this by adding fresh-QK candidates.

Instead, compress only the History prefix levels.

Initial proposal:

    history_operating_levels = 4

For each cluster, compute cumulative History gain:

    S_t = sum_{i=1..t} G_{h_i}

and retain at most four representative prefix lengths corresponding to roughly:

    25%, 50%, 75%, 100%

of the total reusable History gain in that cluster.

Example:

If 17 History edges exist, the retained prefix lengths might be:

    t = [1, 3, 7, 17]

depending on the cumulative gain distribution.

The prefix lengths are adaptive, but the number of operating levels is fixed.

Then the maximum state count becomes:

    Drop
    Rep
    4 x (History-prefix, History-prefix + Rep)
    Exact

so at most:

    2 * history_operating_levels + 3

states per cluster.

With four levels, this is at most 11 states per cluster, independent of how many
History edges are available.

This is the main selector-efficiency improvement.

---

# Required code changes

Primary file:

    saf3r/patch/reuse_residual_exploration.py

## Config

Add:

    history_operating_levels: int = 4

Do NOT add:

- fresh relation slots;
- fresh oversampling;
- per-cluster current-QK probes.

## Compress History prefix enumeration

history_cluster_prefixes() should still:

1. map History edges to current Fine clusters;
2. sort them by History reliability;
3. compute cumulative History gain.

But it should additionally return only a small number of prefix checkpoints:

    prefix_lengths      # [G, F, L]
    prefix_gain         # cumulative gain at each checkpoint
    prefix_valid

where:

    L <= history_operating_levels

The checkpoints are selected from cumulative History gain.

## Joint operating points

joint_history_exploration_operating_points() should use only those checkpoint
prefixes.

For every retained prefix H_t, create:

    H_t
    H_t + Rep

plus:

    Drop
    Rep
    Exact

All states go into the same budget_states() call.

There is no separate History optimization stage.

## Exact fallback

If no compressed state is good enough under the shared loss/cost objective, the
optimizer can choose Exact.

Exact executes every member of the cluster and therefore needs no additional
specific relation search.

---

# Required diagnostics

Add:

    history_candidates_mean
    history_prefix_levels_mean
    history_selected_mean
    history_plus_rep_clusters_mean
    exact_clusters_mean
    exact_edges_mean
    physical_budget_utilization_mean
    actual_unique_density

Also record:

    mean_cluster_size_when_exact
    mean_history_count_when_exact
    mean_history_count_when_compressed

The central stage-level relationship to verify is:

    reusable History utility increases
            ->
    History + Rep states are selected more often
            ->
    Exact clusters / Exact edges decrease

---

# Clean no-History baseline

The correct no-History baseline is NOT fresh concrete-QK discovery.

Disable History and give each cluster only:

    Drop
    Rep
    Exact

This answers the causal question:

Does previously computed History allow more clusters to avoid Exact expansion
than Value representation alone?

Compare:

- quality;
- Exact cluster ratio;
- Exact QK edges;
- final physical support;
- selector latency;
- total latency.

---

# Acceptance criteria

The redesign is useful only if:

1. operating-point count is bounded by a small constant per cluster;
2. no new per-cluster QK discovery is introduced;
3. History + Rep reduces Exact expansion compared with the no-History
   Drop/Rep/Exact baseline;
4. quality is maintained or improved;
5. selected physical support decreases when useful History can replace Exact.

The intended behavior is:

    History provides concrete relations
            +
    Value representative covers the residual
            ->
    keep cluster compressed

    otherwise
            ->
    Exact expansion

This is the implementation invariant for the next revision.

---

# Implementation order

1. Add history_operating_levels.
2. Compress arbitrary History prefixes into at most L cumulative-gain
   checkpoints.
3. Keep only Drop / Rep / H_t / H_t+Rep / Exact states.
4. Keep the existing single global budget_states() solver.
5. Add a no-History Drop / Rep / Exact control.
6. Add Exact-rate and physical-support diagnostics.
7. Run two-frame correctness first.
8. Then run 128-frame quality/runtime.

Do not change query-group size, History evidence granularity, current Value
clustering, canonical grid, total retention, or attention execution kernel in
this experiment.
