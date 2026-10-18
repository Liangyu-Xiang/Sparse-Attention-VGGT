# Region Attention corrections and controlled oracle diagnostics

[Full report](report/report.md) ·
[Implementation/protocol](../../docs/experiments/region_history_corrections.md)

[当前完整中文流程及128帧评估状态](../../docs/experiments/current_region_attention_workflow.md)

Two-frame Chess0/10, globalDense0–9 / sparse10–16 / Dense17–23,
final10%/20%budgets including specials,50%history transfer.

- `chess_two_frames/budget10` and `budget20`:14same-Dense-input methods;
  legacy reproduces previous masks bitwise and relative output errors to1e-5.
- `native_two_frames`:non-oracle corrected installer on the actual sparse
  backbone, A/B and both budgets, with per-Query hard-history/budget/special
  checks, Dense prefix identity and repeated pose identity.
- `report`:paired comparisons, per-layerCSV, raw/summaryJSON and old/new masks.

Exact mass, Dense-best-member and influence modes are DIAGNOSTIC Dense-information
experiments, not deployable fast selectors. Population bias is only approximate
mass. Local terminal refinement in the non-oracle installer incurs extra QK,
including repeated winning logits at final execution, and must not be advertised
as free. No128-frame speedup or7Scenes accuracy claim is made.

Previous experiment files were not overwritten. Error aggregation, history
freezing and oracle-access semantics are explicitly recorded in each result.
