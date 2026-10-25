# Physical History–Rep Budget Accounting Ablation Handoff

## Scope

This experiment evaluates one specific issue discovered in the completed
all-executed-edge History experiment:

> The same physical patch key can simultaneously be a previous History candidate
> and the current Value-cluster representative. The previous dual formulation
> charged these as two independent cost-1 decisions and only deduplicated them
> later during packing.

This caused a large mismatch between nominal 20% retention and the actual
executed sparse support.

The current branch introduces **physical History–Rep overlap accounting** while
keeping all other method components fixed.

Branch:

`refactor/unified-reuse-exploration`

Do not change History scoring, Value clustering, Rep/Exact criterion, query
grouping, population compensation, canonical observation, or retention in this
ablation.

---

## Frozen reference

The immediately preceding all-edge/no-coverage experiment is archived under:

`outputs/all_edge_history_no_coverage/`

Its evaluated method commit was:

`1a7bb41d955da284eded195a2e0b11faf49c2ec3`

Important reference behavior:

- nominal patch retention: 20%;
- L9–L16 actual unique density: about 16.49%;
- L9–L16 unused patch budget: about 4676 keys/group;
- L9–L16 History fraction of actual patch support: about 48.6%;
- L9–L16 History/Rep candidate overlap: about 13179 keys/group;
- AUC@3: 22.6513%;
- AUC@30: 84.4355%;
- CD: 0.12984 m;
- F1: 46.3144%;
- mean scene forward: 131.84 s.

Use the committed JSON/CSV as source of truth.

---

# Production delta under test

## Previous independent accounting

If the same physical key k was:

1. selected as a History edge, and
2. selected as the current representative of Value cluster C,

the optimizer paid:

```
History(k): cost 1
Rep(C=k):   cost 1
total optimizer cost: 2
```

but packing executed k only once.

This is why the optimizer could believe the 20% budget was exhausted while the
actual support remained around 16–17%.

## New merged physical state

When a History key is exactly the current representative of cluster C, the
standalone History item is disabled and its utility is transferred into the
cluster state.

The overlapping cluster has four states:

```
Drop:
    loss = HistoryGain(k) + DropLoss(C)
    physical cost = 0

History-only:
    loss = DropLoss(C)
    physical cost = 1

Rep:
    loss = RepLoss(C)
    physical cost = 1
    # the same key simultaneously satisfies the History relation

Exact:
    loss = 0
    physical cost = |C|
```

Thus the same physical QK edge is never charged twice merely because it has both
History and Rep semantics.

For non-overlapping History candidates and clusters, the previous states are
unchanged.

## Important limitation

This ablation fixes **History/current-Rep overlap only**.

If a selected non-representative History key lies inside a cluster selected as
Exact, Exact still pays |C| while History separately pays cost 1. That residual
overlap is recorded as:

`history_exact_overlap_selected_mean`

Do not fix it in this experiment unless it becomes quantitatively large.

---

# Configuration

Default evaluation config now enables:

```json
"physical_overlap_accounting": true
```

A frozen control config is provided:

`configs/evaluation/reuse_residual_exploration_no_physical_overlap.json`

This permits a same-code matched A/B diagnostic.

---

# New / important recorded quantities

Each layer now records:

- `history_candidates_mean`
- `reuse_edges_mean`
- `history_rep_candidate_overlap_mean`
- `history_rep_merges_mean`
- `history_exact_overlap_selected_mean`
- `population_representative_clusters_mean`
- `counts_mean`
- `optimizer_cost_mean`
- `physical_budget_utilization_mean`
- `optimizer_physical_gap_mean`
- `unused_budget_mean`
- `actual_unique_density`

Key definitions:

```
physical_budget_utilization = physical executed patch keys / nominal patch budget
optimizer_physical_gap = optimizer charged cost - physical executed patch keys
```

The central implementation hypothesis is:

```
physical_overlap_accounting:
    optimizer_physical_gap ↓
    unused_budget ↓
    physical_budget_utilization → 1
    actual_unique_density → nominal 20% regime
```

---

# Stage 0 — checkout and correctness

Use an isolated worktree.

```bash
git fetch origin

git worktree add ../Sparse-Attention-VGGT-physical-budget \
  origin/refactor/unified-reuse-exploration

cd ../Sparse-Attention-VGGT-physical-budget

git rev-parse HEAD
```

Record the exact HEAD.

Run:

```bash
python -m py_compile \
  saf3r/patch/reuse_residual_exploration.py \
  scripts/evaluate_reuse_exploration.py \
  scripts/diagnose_history_pipeline.py

pytest -q tests/test_reuse_residual_exploration.py
```

Do not continue if tests fail.

If a bug is found, fix only the physical-overlap accounting implementation and
commit the bug fix separately.

---

# Stage 1 — matched two-frame A/B sanity

Run the same current code twice on Chess:

## A. control: old independent accounting

```bash
python scripts/evaluate_reuse_exploration.py \
  --method reuse_explore \
  --scene chess \
  --frames 2 \
  --diagnostic \
  --capture \
  --config configs/evaluation/reuse_residual_exploration_no_physical_overlap.json \
  --output-dir outputs/physical_overlap_accounting/diagnostic_control_chess
```

## B. physical accounting

```bash
python scripts/evaluate_reuse_exploration.py \
  --method reuse_explore \
  --scene chess \
  --frames 2 \
  --diagnostic \
  --capture \
  --config configs/evaluation/reuse_residual_exploration.json \
  --output-dir outputs/physical_overlap_accounting/diagnostic_physical_chess
```

Compare L0–L23 layer records.

Required table:

| stage | mode | optimizer cost | physical keys | utilization | optimizer-physical gap | unused budget | History/Rep overlap | merged History+Rep | History/Exact residual overlap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Stages:

- L0–L8
- L9–L16
- L17–L23

### Stage-1 acceptance criterion

The physical-accounting version must materially reduce:

- `optimizer_physical_gap_mean`
- `unused_budget_mean`

and increase:

- `physical_budget_utilization_mean`

without exceeding the nominal patch budget.

If utilization remains near the previous ~82%, stop and inspect accounting
before running seven-scene quality.

Do not require exactly 100% utilization: the Lagrangian state solver can leave a
small integer/slack remainder. The question is whether the large ~18% loss caused
by duplicate charging is removed.

Also inspect:

`history_exact_overlap_selected_mean`

If it remains negligible, leave Exact overlap untouched.

---

# Stage 2 — History-filter diagnostic after accounting fix

The physical budget change alters the sparse trajectory, so rerun the existing
three-scene pipeline diagnostic:

```bash
for scene in chess fire redkitchen; do
  python scripts/diagnose_history_pipeline.py \
    --scene "$scene" \
    --num-frames 2 \
    --first-frame 0 \
    --frame-gap 20 \
    --target-retention 0.20 \
    --layer-start 9 \
    --layer-end 16 \
    --config configs/evaluation/reuse_residual_exploration.json \
    --output-dir outputs/physical_overlap_accounting/pipeline/$scene
done
```

Report macro L9–L16:

- History candidate density;
- selected History density;
- recall;
- precision;
- enrichment;
- attention-mass coverage.

This stage is not intended to fix filtering. It checks whether restoring the
physical budget changes the previous conclusion.

Do not tune History score from these results.

---

# Stage 3 — 7Scenes128 end-to-end quality

Only start if Stage 1 confirms the accounting fix behaves as intended.

Run all seven scenes:

- heads
- chess
- stairs
- fire
- office
- pumpkin
- redkitchen

Protocol must match the archived all-edge/no-coverage experiment:

- 128 frames;
- uniform sampling;
- seed 0;
- original checkpoint;
- total retention 20%;
- 1 warmup;
- 3 measured forwards;
- capture disabled;
- same pose / recon_unposed quality protocol.

Example:

```bash
python scripts/evaluate_reuse_exploration.py \
  --method reuse_explore \
  --scene chess \
  --frames 128 \
  --warmup 1 \
  --repeats 3 \
  --config configs/evaluation/reuse_residual_exploration.json \
  --output-dir outputs/physical_overlap_accounting/quality128/chess
```

Parallel scene dispatch on idle GPUs is allowed.

Do not alter any method hyperparameter.

---

# Stage 4 — aggregate layer behavior

Aggregate all seven `layers.json` files equally by scene.

For every layer L0–L23 report:

| layer | physical utilization | optimizer gap | unused budget | History candidates | selected History | History fraction | merged History+Rep | population Rep clusters | Exact | History/Exact residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Also report stage averages:

- L0–L8
- L9–L16
- L17–L23

Important distinctions:

### History fraction

Use physical selected History keys divided by final physical patch support.

### Representative role

`representatives_mean` is an additive physical-role count and does **not**
include a merged History+Rep key.

Use:

`population_representative_clusters_mean`

to measure how many clusters are actually represented by a population
representative, including merged History+Rep keys.

This avoids incorrectly concluding that Rep usage collapsed merely because its
physical key is counted under History.

---

# Stage 5 — quality comparison

Create:

- `outputs/physical_overlap_accounting/comparison.json`
- `outputs/physical_overlap_accounting/comparison.md`

Compare:

1. Dense historical;
2. SAF3R historical;
3. old reuse+residual historical;
4. previous unified 57bdb78;
5. all-edge/no-coverage;
6. new physical-overlap-accounting version.

Required macro metrics:

- AUC@3
- AUC@30
- ACC
- Comp
- CD
- F1
- mean scene forward seconds
- peak allocated / reserved memory
- actual unique density
- physical budget utilization
- History fraction

The main causal comparison is:

`all-edge/no-coverage -> physical-overlap-accounting`

Do not attribute differences to History scoring because it is unchanged.

---

# Stage 6 — causal analysis

Create:

`outputs/physical_overlap_accounting/analysis.md`

Use:

`docs/experiments/physical_history_rep_budget_analysis_template.md`

as the required report structure.

Answer the following.

## A. Was the budget-accounting hypothesis correct?

Quantify:

- old vs new optimizer-physical gap;
- old vs new unused budget;
- old vs new actual density;
- old vs new physical utilization.

## B. Did quality improve simply by using the intended physical budget?

Compare all six quality metrics.

Particular attention:

- AUC@3 vs SAF3R;
- AUC@30 vs SAF3R;
- CD/F1 vs old reuse+residual;
- scene-level Chess, Fire, Pumpkin, RedKitchen.

## C. How did the History/Rep balance change?

Report both:

- physical History fraction;
- population representative cluster count.

A merged History+Rep key has both semantic roles but only one physical cost.

## D. Did History filtering behavior change?

Use Stage-2 precision/enrichment.

Do not conclude that the filter improved merely because History fraction changed.

## E. Is residual History/Exact overlap material?

Use:

`history_exact_overlap_selected_mean`

If negligible, do not fix it next.

If large enough to explain a meaningful remaining budget gap, report it as the
next accounting issue.

## F. Runtime interpretation

This ablation fixes correctness of physical budget accounting. It does **not**
yet reduce the amount of Exploration work when History grows.

Therefore:

- quality/density changes are the primary objective;
- runtime may stay high or even increase slightly due the four-state merged
  accounting;
- do not treat this as the final efficient implementation.

---

# Stop / next-step rules

Do not modify the production method again before the report is complete.

### If utilization becomes close to nominal and AUC@3 improves

Keep physical overlap accounting permanently. Then study finer current-layer
History filtering evidence (for example smaller query groups / restricted QK
verification) in a separate experiment.

### If utilization improves but F1 remains weak

Keep the accounting fix. Return to Rep/Exact compressibility as a separate
problem.

### If utilization does not improve

Do not run another quality ablation. Debug the merged-state accounting.

### If residual History/Exact overlap becomes large

The next accounting experiment may merge Exact overlap, but only after measuring
its contribution.

---

# Required Codex deliverables

Commit to the same branch:

- Stage-0 logs;
- matched two-frame control/physical outputs;
- three-scene pipeline diagnostics;
- all seven 128-frame outputs;
- per-layer and stage aggregation;
- `comparison.md/json`;
- `analysis.md`.

Final Codex response must state:

1. evaluated commit SHA;
2. Stage-0 test result;
3. L9–L16 old/new physical budget utilization;
4. L9–L16 old/new optimizer-physical gap;
5. L9–L16 History fraction;
6. L9–L16 History precision/enrichment;
7. macro 7Scenes quality;
8. latency/memory;
9. residual History/Exact overlap;
10. exactly one evidence-based recommendation.

Do not implement that recommendation before review.
