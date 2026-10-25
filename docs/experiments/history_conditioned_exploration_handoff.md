# History-Conditioned Exploration Handoff

## Scope

This branch restores the *structural* History -> Exploration dependence that was
lost in the independent unified selector, while keeping the cleaner semantics of
the newer implementation.

Branch:

`refactor/history-conditioned-exploration`

Base branch remains frozen for the physical-accounting experiment:

`refactor/unified-reuse-exploration`

Do not merge or retarget the running physical-accounting experiment to this
branch.

---

## Core hypothesis

The previous unified selector used approximately

```
History loss + independent Exploration loss
```

under one shared budget.  History and Exploration interacted mainly through
budget competition and literal physical-key overlap.

The intended structure is instead

```
History relation selection
        ↓
retained History support
        ↓
History-conditioned residual Exploration
```

A reliable History edge must never be removed merely because its Value is close
to a newly proposed representative.  The opposite should happen: if a retained
History key can represent the same current Value cluster, the *new*
representative should become redundant.

---

# What is retained from the newer implementation

The following semantics remain unchanged:

1. History is **all patch QK support actually executed in the previous layer**,
   independent of Rep / Exact / previous-History origin.
2. History reliability uses previous measured salience plus current coarse
   persistence.
3. No fixed History ratio is introduced.
4. No old `history_per_cluster=4` cap is restored.
5. No `pi * mean(V)` artificial output target is restored.
6. No `current_coarse - beta * previous_real` residual subtraction is restored.
7. No global fine-representative QK probe is restored.
8. Current coarse observation, Fine Value clustering, population compensation,
   total 20% retention, and query-group execution remain otherwise unchanged.

---

# New outer-History / inner-Exploration approximation

At a Lagrange shadow price `lambda`, History is selected first:

```
keep h iff HistoryGain(h) > lambda
```

Importantly, this decision contains **no current Value similarity term**.

Let the selected History members of current Value cluster C be H_C.  Define

```
R_C = C \ H_C
```

as the residual members not explicitly represented by History.

The inner Exploration states are then recomputed *after* H_C is fixed.

## State 0 — History-only / residual Drop

Keep the selected History keys as explicit unit-population QK relations and add
no new representative.

Proxy residual loss:

```
L_drop(C | H_C) = pi_C * |R_C| / |C|
```

Incremental cost:

```
0
```

## State 1 — History -> Representative

Among retained History keys in C, choose the one whose normalized current Value
is closest to the current cluster centre.

Promote that already-paid History key to population representative for R_C.

Proxy residual loss:

```
L_hist_rep(C | H_C)
    = pi_C * |R_C| / |C| * D_V(history_anchor, C)
```

Incremental QK cost:

```
0
```

This is the key interaction:

> reliable History can replace a redundant new representative without losing
> the historical relation or consuming another physical sparse slot.

## State 2 — New Representative

Add the current nominal Value representative only if it still reduces residual
distortion enough to justify one additional QK key.

```
L_new_rep(C | H_C)
    = pi_C * |R_C| / |C| * D_V(rep_C, C)
```

Incremental cost:

```
1
```

If the nominal representative is already a retained History key, this state is
disabled because History -> Representative is free.

## State 3 — Exact residual expansion

Add every member of R_C.

```
L_exact = 0
cost = |R_C|
```

Exact eligibility remains controlled by the existing current-importance
candidate rule; this experiment does not change that hyperparameter.

---

# Budget solution

The selector searches a shared shadow price to satisfy the physical patch-QK
budget.

At every price iteration:

1. select History solely from relation reliability;
2. map selected History into current Fine clusters;
3. recompute the four residual Exploration states conditioned on that History;
4. choose the best inner state for each cluster at the same price;
5. measure the resulting **physical** cost;
6. update the price.

This is an efficient flattened approximation to the conceptual bilevel form

```
outer: choose History
inner: choose optimal residual Exploration given History
```

without enumerating all History subsets or restoring the old artificial output
proxy.

---

# Important semantic check

Suppose a reliable History key h and a nominal representative r have nearly the
same current Value.

The required behavior is:

```
keep h
promote h to representative
do not add r
```

NOT:

```
compare h and r by Value similarity
discard h
keep r
```

Unit tests explicitly cover this case.

---

# Configurations

Conditioned method:

`configs/evaluation/reuse_residual_exploration.json`

with

```json
"history_conditioned_exploration": true
```

Matched independent-loss control:

`configs/evaluation/reuse_residual_exploration_independent.json`

with

```json
"history_conditioned_exploration": false,
"physical_overlap_accounting": true
```

The older no-physical-overlap control is also forced onto the legacy independent
path and should not be used as the primary comparison for this experiment.

---

# New diagnostics

Layer records include:

- `conditioned_history_distortion_mean`
- `history_selected_cluster_coverage_mean`
- `shadow_price_mean`
- `missed_history_gain_mean`
- `history_conditioned_clusters_mean`
- `history_only_clusters_mean`
- `history_plus_rep_clusters_mean`
- `history_plus_exact_clusters_mean`
- `history_promoted_representatives_mean`
- existing History candidates / reuse ratio / filter keep ratio
- physical budget utilization / optimizer-physical gap / unused budget

The most important new quantity is:

`history_promoted_representatives_mean`

It measures how often an already-retained History key directly replaces a new
Value representative.

---

# Stage 0 — correctness

Use a fresh worktree:

```bash
git fetch origin

git worktree add ../Sparse-Attention-VGGT-history-conditioned \
  origin/refactor/history-conditioned-exploration

cd ../Sparse-Attention-VGGT-history-conditioned

git rev-parse HEAD

python -m py_compile \
  saf3r/patch/reuse_residual_exploration.py \
  scripts/evaluate_reuse_exploration.py \
  scripts/diagnose_history_pipeline.py

pytest -q tests/test_reuse_residual_exploration.py
```

Stop on any failure.

Required semantic unit tests must establish:

1. Value redundancy does not remove a reliable History edge.
2. A retained History key can become representative at zero incremental QK cost.
3. Exact expansion pays only for members not already retained by History.
4. With no History, the inner states reduce to the ordinary Drop / Rep / Exact
   Value geometry.
5. Final packed support contains no duplicate physical keys and never exceeds
   the nominal budget.

---

# Stage 1 — matched small-sequence causal A/B

Run the same code and inputs with:

A. independent control

`configs/evaluation/reuse_residual_exploration_independent.json`

B. conditioned method

`configs/evaluation/reuse_residual_exploration.json`

Start with Chess, Fire and RedKitchen using the smallest existing diagnostic
protocol that exercises L9-L16 History.

Report, layer by layer:

| layer | mode | History selected | History clusters | History->Rep | New Rep | Exact | physical utilization | missed History gain | shadow price |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Acceptance checks:

- no duplicate physical support;
- physical support <= budget;
- History candidate semantics unchanged;
- selected History is not reduced by Value similarity;
- conditioned mode shows nonzero History->Rep use when retained History overlaps
  current Value clusters;
- new Rep count should fall when History->Rep rises, if coupling is functioning.

Do not tune any hyperparameter from this stage.

---

# Stage 2 — History quality diagnostic

Rerun the existing History pipeline diagnostic on Chess / Fire / RedKitchen.

The main question is whether the new coupling changes *which* History edges are
retained merely because the shared shadow price changes.

Report L9-L16:

- candidate density;
- selected density;
- Top20 recall;
- precision;
- enrichment;
- attention-mass coverage;
- filter keep ratio.

Do not claim better History filtering unless precision/enrichment improves.

---

# Stage 3 — 7Scenes128 quality

Only after Stage 0 and Stage 1 pass.

Use the identical 128-frame protocol from the previous all-edge and
physical-accounting experiments.

Compare at minimum:

1. Dense historical;
2. SAF3R historical;
3. old reuse+residual prototype;
4. previous unified 57bdb78;
5. all-edge/no-coverage;
6. independent physical-accounting control;
7. History-conditioned Exploration.

Required metrics:

- AUC@3
- AUC@30
- ACC
- Comp
- CD
- F1

Primary causal comparison:

```
independent physical accounting
    ->
History-conditioned Exploration
```

The History score, Value clustering, coarse observation and total retention must
remain frozen.

---

# Stage 4 — interaction analysis

The experiment succeeds conceptually only if the following causal pattern is
observed:

```
retained History ↑
      ↓
History-promoted representatives ↑
      ↓
new representative demand ↓
```

without a corresponding collapse in pose/reconstruction quality.

Report stage averages for L0-L8, L9-L16, L17-L23:

| stage | History frac | History clusters | History->Rep | New Rep | Exact | utilization | AUC/F1 contribution notes |
|---|---:|---:|---:|---:|---:|---:|---|

Also inspect per-scene Chess, Fire, Pumpkin, RedKitchen because previous versions
showed large reconstruction sensitivity there.

---

# Runtime interpretation

This branch is a **semantic / causal implementation**, not yet the final fast
kernel.

The current implementation recomputes History-conditioned cluster states during
the shadow-price iterations using scatter reductions.  Therefore runtime may be
high.

Do not optimize kernels before establishing whether the coupling improves
quality and support allocation.

If the method is validated, the next engineering step is to cache/reorder
History-by-cluster statistics so the nested price search does not repeatedly
rebuild them.

---

# What this experiment does NOT solve

This branch deliberately does not yet change:

- Query-group History granularity;
- 4/32 probe evidence;
- canonical-grid size;
- Fine-clustering source-width restriction;
- per-head History;
- exact current-QK verification of History;
- Exact eligibility fraction.

Those are separate questions and should remain frozen here.

---

# Required final report

Commit all results under:

`outputs/history_conditioned_exploration/`

Final report must state:

1. evaluated SHA;
2. Stage-0 test status;
3. whether reliable History survives Value redundancy;
4. L9-L16 History fraction;
5. L9-L16 History->Rep count/fraction;
6. change in New-Rep count;
7. physical budget utilization;
8. History precision/enrichment;
9. macro 7Scenes quality;
10. runtime/memory;
11. exactly one evidence-based next recommendation.

Do not implement the recommendation before review.
