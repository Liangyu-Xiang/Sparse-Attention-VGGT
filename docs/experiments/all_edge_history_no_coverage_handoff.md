# All-Executed-Edge History + No-Coverage Ablation Handoff

## Purpose

This handoff evaluates one causal production change on branch:

`refactor/unified-reuse-exploration`

Do not redesign the method during this experiment.

The previous tested unified implementation starved History because it stored only
edges labeled "genuine" by the selector.  Representatives that actually executed
QK were excluded, so early Rep-heavy layers could not seed History.

The new definition is:

> **History at layer l is the complete patch-key support that actually executed
> QK at layer l-1, independent of whether each key entered the old mask through
> Rep, Exact, or previous History.**

The previous explicit-relation `coverage` gate is also removed.

Everything else is intentionally held fixed.

---

## Production delta under test

### History write

After sparse attention at layer l, every valid patch support key is written to
the next History bank.

For population representatives, outgoing historical salience uses:

```
single_edge_salience = measured_sparse_mass / represented_population
```

This keeps the relation evidence closer to the actual QK edge instead of using
the aggregate population mass.

### History filtering at the next layer

For previous edge e in canonical region r:

```
s_e = old_single_edge_salience_e / sum_{j in H_r} old_single_edge_salience_j

G_H(e) = p_r(current) * beta_r * s_e
```

where:

- `p_r(current)` is current canonical coarse importance;
- `beta_r = min(1, p_r(current) / p_r(previous))` is the existing asymmetric
  cross-layer persistence factor;
- `s_e` is only a previous-layer relative salience inside the region.

The old absolute probability is never treated as a current-layer probability.

### Intended progressive behavior

The method is allowed to have poor History recall early.

The intended mechanism is:

```
broad previous sparse support
        ↓
current coarse region filtering + old edge salience
        ↓
selected History subset
        +
new Exploration Rep/Exact edges
        ↓
current QK is actually executed
        ↓
updated current salience becomes next-layer History evidence
        ↓
repeat
```

Therefore the key hypothesis is not "L9 History recall must already be high".

The hypothesis is:

> **As middle-layer Attention sharpens, selected History should become more
> concentrated on current high-weight QK edges, so History precision/enrichment
> and useful reuse should improve progressively.**

### Deliberately unchanged

Do NOT change any of these in this ablation:

- total retention = 20%;
- Value clustering;
- Rep/Exact objective;
- `candidate_fraction`;
- population compensation;
- group32 query sharing;
- four probe queries;
- canonical 2x2/frame observation;
- `beta` definition;
- dual budget solver;
- final sparse-attention kernel.

This experiment is only about:

1. **all executed previous patch edges are eligible History**;
2. **coverage gate removed**.

---

## Reference points

Use the already committed historical results for comparison.

### Previous unified version

The previous Stage-1 analysis was committed at:

`ba22bcec4b372ba63680df96332b6c0929e90689`

The corresponding previously evaluated production quality is the
`57bdb78` run under:

`outputs/global_attention/unified_refactor_57bdb78/`

Important previous behavior:

- L9-L12 History candidate density approximately zero;
- L9-L16 History reuse only around 1% scale;
- 7Scenes128 macro approximately:
  - AUC@3 22.30%
  - AUC@30 84.31%
  - CD 0.1326 m
  - F1 46.80%

Use the committed JSON/Markdown values rather than retyping numbers if there is
any discrepancy.

### Strong old prototype

`outputs/global_attention/reuse_explore_20261021/`

This is the older high-reconstruction reference, not the production baseline for
this causal ablation.

### SAF3R

Use the already committed matched SAF3R results.  Do not rerun SAF3R unless a
matched-input inconsistency is discovered.

---

# Stage 0 — checkout and validation

Use a fresh worktree if the server checkout has local changes.

```bash
git fetch origin

git worktree add ../Sparse-Attention-VGGT-allhist \
  origin/refactor/unified-reuse-exploration

cd ../Sparse-Attention-VGGT-allhist

git rev-parse HEAD
```

Record the exact HEAD SHA in the experiment README.

Run:

```bash
python -m py_compile \
  saf3r/patch/reuse_residual_exploration.py \
  scripts/diagnose_history_pipeline.py \
  scripts/evaluate_reuse_exploration.py

pytest -q tests/test_reuse_residual_exploration.py
```

If validation fails:

- fix only an implementation bug required to make the intended all-edge/no-
  coverage definition run;
- do not change the selector design;
- commit the bug fix separately;
- rerun Stage 0.

Do not start quality evaluation if Stage 0 fails.

---

# Stage 1 — two-frame History filtering diagnostic

Run Chess, Fire, RedKitchen with the same two-frame diagnostic protocol used in
the previous Stage-1 experiment.

Use a fresh output root:

`outputs/all_edge_history_no_coverage/`

Example:

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
    --output-dir outputs/all_edge_history_no_coverage/pipeline/$scene
done
```

The updated diagnostic records:

- recall;
- precision with respect to current Top20;
- enrichment over random Top20 membership;
- attention-mass coverage;
- candidate density;
- candidate key count.

## Stage-1 sanity check

With the new History definition, `history_candidates` in L9-L16 must no longer
be near zero.

Because History is the previous executed patch support, its candidate set should
closely correspond to `previous_full`, modulo representation/packing details.

If candidate density is still near zero, stop: this is an implementation bug.

## Required Stage-1 tables

For each scene and macro average, report for L9-L16:

| layer | previous_full density | History candidate density | selected History density | selected History recall | selected History precision | selected History enrichment | selected History mass |
|---|---:|---:|---:|---:|---:|---:|---:|

Also report:

`history_selected / history_candidates`

as the filtering keep ratio.

## How to interpret filtering

Do **not** require high initial recall.

Good evidence for the intended progressive filter would be some combination of:

- selected History precision/enrichment rises in L10-L15;
- selected History keeps a smaller subset than the full candidate bank while
  preserving disproportionately large current Top20 overlap or attention mass;
- History reuse ratio grows as middle-layer attention relations sharpen.

Warning signs:

### F1 — filter too weak

History reuse immediately becomes very large, but precision/enrichment stays
near random or quality degrades.

### F2 — filter too strong

Candidate bank is healthy (~previous support), but selected History remains near
zero at all middle layers.

### F3 — candidate quality is poor

Candidates and selected History are both present, but precision/enrichment do
not improve through the middle layers.

### F4 — desired progressive behavior

Early reuse may be modest, then selected History becomes increasingly enriched
for current high-attention edges around L10-L15.

Do not tune thresholds after seeing 7Scenes quality in this stage.

---

# Stage 2 — 7Scenes128 end-to-end evaluation

Only start after Stage 1 is valid.

Scenes:

`heads chess stairs fire office pumpkin redkitchen`

Run the same evaluator and timing protocol as the previous unified results:

- 128 frames;
- uniform sampling, seed 0;
- original checkpoint;
- 1 warmup;
- 3 measured complete forwards;
- capture disabled;
- same quality protocol.

Example for one scene:

```bash
python scripts/evaluate_reuse_exploration.py \
  --method reuse_explore \
  --scene chess \
  --frames 128 \
  --warmup 1 \
  --repeats 3 \
  --output-dir outputs/all_edge_history_no_coverage/quality128/chess
```

Parallelize scenes across idle GPUs if desired, but do not change the algorithm
or input protocol.

Required quality metrics:

- AUC@3
- AUC@30
- ACC
- Comp
- CD
- F1

Required timing/memory:

- median complete-forward seconds;
- cold forward;
- peak allocated/reserved memory;
- actual sparse density.

---

# Stage 3 — layerwise behavior analysis

For each 128-frame scene, read `layers.json`.

The current code records:

- `history_candidates_mean`
- `reuse_edges_mean`
- `history_reuse_ratio_mean`
- `history_filter_keep_ratio_mean`
- `history_rep_candidate_overlap_mean`
- `representatives_mean`
- `exact_edges_mean`
- `counts_mean`
- `stored_history_bytes`
- `actual_unique_density`

Aggregate equally across all seven scenes.

Produce this table for every L0-L23:

| layer | History candidates | selected History | History fraction of final support | filter keep ratio | History/Rep candidate overlap | unused budget | Rep | Exact | stored History MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Also produce stage averages:

- L0-L8
- L9-L16
- L17-L23

The central question is whether the middle layers now exhibit the expected
progressive History reuse rather than the previous ~zero candidate bank.

Do not impose a target such as "must reach 80%".  Measure the behavior first.

---

# Stage 4 — compare against frozen references

Create:

`outputs/all_edge_history_no_coverage/comparison.md`

and

`outputs/all_edge_history_no_coverage/comparison.json`

At minimum compare:

1. Dense;
2. SAF3R;
3. old high-quality reuse+residual prototype;
4. previous unified `57bdb78`;
5. new all-edge History + no-coverage version.

Use equal-scene macro averages.

For the new version include:

- quality metrics;
- full-forward latency;
- peak memory;
- average/middle-layer History fraction.

Important: if History memory rises strongly because the complete 20% previous
support is stored, report it explicitly.  Do not hide this cost.

Also report `history_rep_candidate_overlap_mean` and `unused_budget_mean`.
With all previous support eligible, the same physical key can appear both as a
History item and as a current representative candidate.  The current independent
dual formulation can over-count that shared cost before packing deduplicates it.
This is a diagnostic confound to measure in this ablation, not something to
silently fix at the same time.

---

# Stage 5 — causal analysis

Create:

`outputs/all_edge_history_no_coverage/analysis.md`

Answer these questions explicitly.

## A. Did the History bootstrap deadlock disappear?

Evidence:

- previous_full candidate density;
- history_candidates density;
- first layer where History candidates appear.

Expected semantic result: History should exist from the layer immediately after
the first sparse layer, because every executed patch edge is historical.

## B. Does implicit filtering work?

Use selected History:

- density;
- recall;
- precision;
- enrichment;
- attention-mass coverage.

Focus on L10-L15.

The desired evidence is not merely "more History".  It is that selected History
becomes increasingly concentrated on current high-attention relations as the
network deepens.

## C. How much History is actually selected?

Report actual final-support History fractions by layer.

Do not equate the size of the candidate bank with accepted History reuse.

## D. Does quality improve?

Compare directly to `57bdb78`.

Especially inspect:

- AUC@3: relation/correspondence sensitivity;
- CD/F1: reconstruction;
- per-scene RedKitchen / Fire / Pumpkin, which previously lost reconstruction
  quality.

## E. What is the efficiency cost?

Compare complete-forward latency and memory.

The current selector still evaluates all Fine exploration items, so this causal
ablation is NOT expected to fully realize the eventual property

`more reuse -> less exploration work`.

Do not redesign that part during this experiment.

The purpose here is first to establish whether the corrected History semantics
improves support/quality.

---

# Stop / follow-up rules

Do not make another production-method change before the report is complete.

After results, choose the next action from evidence:

### If candidate bank is healthy and filtering progressively enriches History,
and quality improves

Keep the all-edge History definition.  Next study how selected History can
reduce unresolved Exploration work.

### If candidate bank is healthy but selected History remains near zero

The next target is History score/budget calibration, not candidate generation.

### If selected History is large but enrichment/quality is poor

Filtering is too weak or group32/coarse evidence is insufficient.  Do not add
more History blindly.

### If History behavior improves but reconstruction remains weak

Keep History fixed and return to the Rep/Exact compressibility problem.  Do not
mix that change into this ablation.

### If runtime/memory becomes unacceptable

Report the cost first.  A compact/top-candidate History implementation is a
separate efficiency change and must be evaluated after the semantic ablation.

---

# Required Codex deliverables

Commit to the same branch:

- Stage-0 logs;
- all Stage-1 pipeline CSV/JSON;
- all seven 7Scenes128 outputs;
- layerwise aggregation;
- `comparison.md/json`;
- `analysis.md`;
- any bug-only diagnostic/test fixes.

In the final Codex response state:

1. exact evaluated commit SHA;
2. Stage-0 test status;
3. whether History candidates now equal/track the previous executed support;
4. L9-L16 selected History fractions;
5. L9-L16 precision/enrichment trend;
6. seven-scene macro quality;
7. full-forward latency and peak memory;
8. one evidence-based recommendation for the next step.

Do not implement the next recommendation until these results are reviewed.
