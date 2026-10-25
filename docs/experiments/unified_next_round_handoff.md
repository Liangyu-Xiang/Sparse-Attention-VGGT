> **Superseded for the current production ablation.** Stage-1 diagnostics from this handoff have already been completed and archived under `outputs/unified_next_round/`. For the current all-executed-edge History + no-coverage experiment, follow `docs/experiments/all_edge_history_no_coverage_handoff.md` instead.

# Unified Reuse: Next-Round Diagnostic Handoff

## Scope

This document is the execution handoff for the next experimental round on branch:

`refactor/unified-reuse-exploration`

The goal is **not** to immediately invent another selector.  First isolate four questions with controlled diagnostics:

1. How much adjacent-layer QK support is actually reusable in Dense VGGT?
2. Where does the current History pipeline lose that reusable support?
3. When can a Value cluster safely use one representative, and when should it stay Exact?
4. Does the current implementation satisfy the desired efficiency property that more History reuse reduces mask-construction work?

The default method must remain unchanged during Stage 1.  One capture-only instrumentation change has been added to
`reuse_residual_exploration.py`: when `capture=True`, each window now records
`history_candidates` and `history_selected`.  This has no effect when capture is disabled and does not change normal selection.

Population-compensation questions are explicitly out of scope for this round.

---

## New scripts

### A. Dense History reuse ceiling

`scripts/diagnose_history_reuse_ceiling.py`

Runs an **unmodified Dense VGGT** on a small clip.  For every adjacent layer pair it asks whether the previous layer's
per-head/per-query Top-r support contains the current layer's Top-20% support.

Default candidate sizes: 20%, 30%, 40%, 50%, 80%.

Outputs:

- `per_head.csv`
- `transitions.csv`
- `summary.json`
- `reuse_ceiling.png`

Primary quantities:

- `recall`: current Top-20% key recall from previous support;
- `attention_mass_coverage`: current dense patch-attention mass inside previous support;
- `same_prefix_jaccard`;
- query-independent key-saliency control.

This experiment answers whether the intuitive 50–80% reuse regime is supported by Dense Attention itself.

### B. Current History pipeline diagnosis

`scripts/diagnose_history_pipeline.py`

Runs the **current unified sparse method** with `capture=True`.  Before L9–L16 attention is executed, it also computes a
diagnostic full patch QK target on the same sparse network trajectory.

It compares:

1. `previous_full`: full sparse support from L-1;
2. `history_candidates`: current History bank before budget selection;
3. `history_selected`: History edges that survive the shared budget selector;
4. `final_support`: complete final support of the current layer.

Outputs:

- `pipeline.csv`
- `summary.json`

Interpretation:

- previous_full high, history_candidates low -> candidate-bank construction is losing useful edges;
- history_candidates high, history_selected low -> scoring/shared-budget selection is the bottleneck;
- previous_full already low -> the prior sparse layer failed to bootstrap the relations;
- always inspect recall together with candidate density.

This is the key experiment for explaining why the current end-to-end History ratio is only ~1–4% in middle layers even though Dense
cross-layer predictability can be much higher.

### C. Value-cluster Rep/Exact compressibility

`scripts/diagnose_value_cluster_compressibility.py`

Runs the current sparse method and evaluates the actual Fine Value clusters with diagnostic Q/K/V on L10–L15.

For the important cluster/query-group pairs it records:

- `value_rel_l2`
- `value_cosine_dist`
- `key_rel_l2`
- `key_cosine_dist`
- `qk_logit_std`
- `rep_affinity_error`
- `mass_preserving_value_error`
- current inferred state (`rep`, `exact_like`, or `other`)

Definitions:

`rep_affinity_error` compares the representative QK logit to the cluster's log-mean-exp QK affinity.  It asks whether one real Key
behaves like the whole cluster in QK space.

`mass_preserving_value_error` gives the representative the cluster's **true diagnostic attention mass**, then measures only how much
the Value contribution changes.  It therefore does not test population compensation.

The main question is whether clusters exist with:

`small Value distortion + large QK/affinity error`.

If yes, then Value similarity is suitable for *forming* clusters but is insufficient for deciding Rep versus Exact.

### D. Rank the Rep/Exact diagnostic signals

`scripts/summarize_value_compressibility.py`

Takes one or more `clusters.csv` files and ranks the cheap signals against:

`importance * mass_preserving_value_error`.

It reports rank correlation and precision/recall at the top diagnostic-error fraction.

This script is for selecting which signal deserves a later end-to-end ablation.  Do not choose a threshold from 7Scenes quality.

### E. History reuse / selector scaling benchmark

`scripts/benchmark_history_reuse_scaling.py`

A CUDA microbenchmark of the **current selector structure** at 128-frame dimensions.  It varies the number of History items while
keeping the complete Fine exploration table.

Outputs:

- `scaling.csv`
- `summary.json`

The expected current behavior is that exploration work stays almost constant and the solver can become more expensive as History is
added.  This intentionally demonstrates that the current implementation does **not yet** satisfy:

`History reuse up -> unresolved exploration down -> mask-construction time down`.

---

## Stage 0: checkout and validation

Use an isolated worktree if the server has local changes.

```bash
git fetch origin
git worktree add ../Sparse-Attention-VGGT-next origin/refactor/unified-reuse-exploration
cd ../Sparse-Attention-VGGT-next

git rev-parse HEAD

python -m py_compile \
  scripts/diagnose_history_reuse_ceiling.py \
  scripts/diagnose_history_pipeline.py \
  scripts/diagnose_value_cluster_compressibility.py \
  scripts/summarize_value_compressibility.py \
  scripts/benchmark_history_reuse_scaling.py \
  scripts/diagnose_query_group_granularity.py

pytest -q tests/test_reuse_residual_exploration.py
```

Do not modify the production selector in Stage 0/1.  If a diagnostic script has an implementation bug, fix only the diagnostic and
commit the fix separately.

---

## Stage 1A: Dense reuse ceiling

Start with three representative scenes:

- chess
- fire
- redkitchen

Use the existing small two-frame protocol first so the diagnostic QK matrices are cheap.

Example:

```bash
for scene in chess fire redkitchen; do
  python scripts/diagnose_history_reuse_ceiling.py     --scene $scene     --num-frames 2     --first-frame 0     --frame-gap 20     --target-retention 0.20     --candidate-retentions 0.20 0.30 0.40 0.50 0.80     --layer-start 8     --layer-end 17     --output-dir outputs/unified_next_round/reuse_ceiling/$scene
done
```

Required report:

- per-layer recall curves for each candidate ratio;
- L10->11, L11->12, L12->13, L13->14, L14->15 means;
- same-size Top20->Top20 overlap;
- Top40/50/80 candidate recall of current Top20;
- current attention-mass coverage;
- comparison to query-independent key-saliency control.

Do **not** interpret an 80%-sized candidate set with 90% recall as "90% of the final mask can be reused".  Candidate density must be
reported alongside recall.

A useful result would be, for example, middle-layer Top20->Top20 recall near or above 50%, or a moderately wider previous support
reaching 70–80%+ current recall.  These are observations, not hard acceptance thresholds.

---

## Stage 1B: locate History pipeline loss

Run the same clips:

```bash
for scene in chess fire redkitchen; do
  python scripts/diagnose_history_pipeline.py     --scene $scene     --num-frames 2     --first-frame 0     --frame-gap 20     --target-retention 0.20     --layer-start 9     --layer-end 16     --output-dir outputs/unified_next_round/history_pipeline/$scene
done
```

For every L9–L16 layer make one table:

| layer | previous_full recall | history_candidates recall | history_selected recall | final_support recall |
|---|---:|---:|---:|---:|

Also include candidate density and attention-mass coverage.

Decision tree:

### Case H1: previous_full is high, history_candidates drops strongly

The current definition of "genuine History" is too restrictive or the compact History bank discards the useful part of the previous
support.  Next method ablation should change **History candidate generation**, not the shared budget formula.

### Case H2: history_candidates is high, history_selected drops strongly

Candidate generation works, but the History score / coverage gate / global budget competition suppresses it.  Next ablation should
change **History selection**, not Value clustering.

### Case H3: previous_full itself is low

The previous sparse layer never discovered the correct support.  Focus on **exploration/bootstrap** or finer query/head granularity.

### Case H4: all support recall is high but AUC@3 is still weak

Then support recall alone is not the missing quantity; inspect per-query probability/output error and head/query sharing.

---

## Stage 1C: Rep versus Exact compressibility

Run the same three scenes first:

```bash
for scene in chess fire redkitchen; do
  python scripts/diagnose_value_cluster_compressibility.py     --scene $scene     --num-frames 2     --first-frame 0     --frame-gap 20     --layer-start 10     --layer-end 15     --top-clusters 64     --output-dir outputs/unified_next_round/compressibility/$scene
done
```

Then summarize:

```bash
python scripts/summarize_value_compressibility.py   outputs/unified_next_round/compressibility/chess/clusters.csv   outputs/unified_next_round/compressibility/fire/clusters.csv   outputs/unified_next_round/compressibility/redkitchen/clusters.csv   --positive-fraction 0.20   --output outputs/unified_next_round/compressibility/summary.json
```

Required analysis:

1. Correlation / top-20% detection quality for Value, Key, and QK-based signals.
2. Fraction of important clusters with low Value distortion but high affinity/logit error.
3. Compare current `rep` clusters and `exact_like` clusters in these metrics.
4. Produce at least these scatter plots:
   - Value distortion vs representative affinity error;
   - Value distortion vs mass-preserving contribution error;
   - QK logit std vs mass-preserving contribution error;
   - Key distortion vs mass-preserving contribution error.

The central hypothesis to test is:

> Value similarity is enough to define the cluster, but a separate Attention-side compressibility signal is needed to decide
> whether that cluster may be represented by one Key or should be Exact.

---

## Stage 1D: does more History currently make mask construction cheaper?

Run:

```bash
python scripts/benchmark_history_reuse_scaling.py   --frames 128   --reuse-ratios 0 0.25 0.50 0.75 0.80   --warmup 5   --repeats 30   --output-dir outputs/unified_next_round/history_scaling
```

Report:

- exploration ms/window;
- dual-solver ms/window;
- projected selector ms over all query groups;
- peak memory.

Expected current result:

- exploration cost approximately constant with History ratio;
- solver cost flat or increasing.

This is evidence of an **implementation/design inefficiency**, not evidence against reuse itself.

The final method should eventually satisfy a stronger property:

`reliable History fraction increases -> unresolved exploration region/candidate work decreases -> mask construction becomes cheaper`.

---

## Stage 1E: query-group granularity diagnostic

The script is already implemented:

`scripts/diagnose_query_group_granularity.py`

Run it at least on Chess; repeat on Fire/RedKitchen if the effect is large:

```bash
python scripts/diagnose_query_group_granularity.py \
  --scene chess \
  --num-frames 2 \
  --first-frame 0 \
  --frame-gap 20 \
  --retention 0.20 \
  --group-sizes 32 16 8 4 1 \
  --layer-start 9 \
  --layer-end 15 \
  --output-dir outputs/unified_next_round/query_granularity/chess
```

All variants use the same 20% candidate density.  The source support is built from previous Dense attention averaged over heads and
over the specified query group.  The target remains per-head/per-query current Top20.

If recall rises substantially from group32 to group8/group4/group1, that directly measures information loss caused by shared query
support.  This experiment deliberately keeps head sharing fixed; only after a query-group effect is established should a separate
head-group diagnostic be added.

Do not run five end-to-end variants until this offline diagnostic shows a meaningful recall gain.

---

## Stage 2: controlled end-to-end ablations (gated by Stage 1)

Do not start Stage 2 until Stage 1 results are committed and summarized.

### History ablation

Choose **one** change according to H1/H2/H3.

Do not simultaneously change History candidate generation, History score, query-group size, and Value compressibility.

The first History end-to-end test should answer one causal question only.

### Rep/Exact ablation

Keep Value clustering itself fixed.

Compare the current Value-only Rep/Exact behavior against the best cheap Attention-side signal identified in Stage 1C.

Desired variants:

- A: current Value-only baseline;
- B: Key-dispersion criterion, if Stage 1C supports it;
- C: low-cost QK-probe criterion, if Stage 1C supports it;
- D: diagnostic oracle on a tiny clip only, never a reported accelerator.

All variants must retain the same total sparse budget.  Rep/Exact allocation may change; total budget may not.

Only after a criterion is selected on calibration/held-out diagnostics should it be frozen and evaluated on 7Scenes128.

---

## Dataset / tuning rule

7Scenes small clips may be used to **explain the already observed failure** and reproduce qualitative phenomena.

Do not select thresholds or hyperparameters from 7Scenes test quality metrics.

For any criterion that becomes part of the final method:

1. determine its form / threshold on a calibration or held-out set;
2. freeze it;
3. then run 7Scenes128.

No AUC/F1 feedback should be used to tune the diagnostic criterion.

---

## Required Codex deliverables

Before proposing another method revision, commit:

`outputs/unified_next_round/`

with all raw CSV/JSON/figures plus:

`outputs/unified_next_round/analysis.md`

Use `docs/experiments/unified_next_round_analysis_template.md` as the report template.\n\nThe analysis must contain:\n
### History

- Dense reuse ceiling by layer and candidate ratio;
- previous_full -> history_candidates -> history_selected recall drop;
- actual candidate densities;
- a statement of which H1/H2/H3/H4 case is supported.

### Value compressibility

- metric ranking from `summarize_value_compressibility.py`;
- low-Value-distortion / high-Attention-error examples;
- whether current Rep decisions are over-aggressive;
- which single signal should be tested next, if any.

### Efficiency

- reuse ratio vs selector time table;
- explicit statement whether current implementation satisfies
  `more reuse -> lower mask construction time`.

### Recommendation

Propose at most **one History change** and **one Rep/Exact change** for the next end-to-end experiment.  Do not implement a broad
multi-module rewrite before this report is reviewed.

---

## Current hypotheses to verify, not assume

H-A: Middle-layer Dense support has enough adjacent-layer predictability to support 50–80% candidate reuse.

H-B: The current ~1–4% History ratio is caused by the History pipeline, not by absence of reusable relations.

H-C: The current selector overuses one-slot representatives because Value clustering already guarantees small Value distortion.

H-D: Old prototype reconstruction quality came partly from detecting clusters that should remain Exact, rather than from the old
artificial `pi * mean(V)` target itself.

H-E: Current History reuse does not reduce exploration work; a final efficient implementation must explicitly skip/reduce exploration
as reliable History coverage grows.

The purpose of this round is to confirm or reject these hypotheses separately.
