# Middle-layer Attention: two-frame results and intermediate masks

Chess seq-01 frames0/10. Global layers0–9 and17–23 remain Dense;
layers10–16 use the spatial region/history Attention branch.

## Read the results

- [20%budget: every intermediate stage, layers10–16](stages/budget20_B/report.md)
- [10%budget: every intermediate stage, layers10–16](stages/budget10_B/report.md)
- [A/B search, fixed/adaptive history and both final budgets](chess_two_frames/report/report.md)
- [Implementation and evaluation protocol](../../docs/experiments/region_history_middle_isolation.md)

Stage visualizations use B(two-level search),50%history transfer and the actual
sparse trajectory. Blue marks membership in ORIGINAL Query/Key coordinates.
Region membership is not an assertion that all individual QK edges were computed.
The initial coarse gate filters history only; exploration starts over all fine
cells not covered by retained history. Fine cells are connected frame-local grids.

Each stage run verified bitwise identity of final masks and model poses against
the previous unobserved run, plus exact cross-layer history handoff. Raw packed
stage masks and coarse probabilities are in each `exact_stages.npz`; per-stage
definitions and unpacking dimensions are in `result.json`.

The two-frame sweep includes20settings:2budgets x2search variants x
(0%,25%,50%,100%history transfer plus coarse-concentration-adaptive history).
Each setting has controlled-Dense-input and actual-sparse-trajectory diagnostics.
All final budgets include mandatory special Keys. Raw final indices and poses
are retained in each budget/variant directory. CSV tables are deliberately
versioned here despite the repository's general CSV ignore rule.

These are TWO-FRAME diagnostic results, not128-frame speed or7Scenes accuracy
claims. Neighborhood radius2/4recall must be compared with the included matched
random-support baseline. High neighborhood coverage alone does not show success.

## Validation

The focused selector/stage tests passed:12tests. The broader selector, prior
residual implementation and connectivity tests passed:29tests before the new
stage-capture test was added. The synchronization also tests the staged snapshot
independently so unrelated working-tree changes are not required.
