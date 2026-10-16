# Synced evaluation artifacts

This directory contains the final v4 seven-scene, 128-frame evaluation, not the
failed v2/v3 attempts or the earlier v1 kernel/policy version.

Version-controlled artifacts:

- `report.md`, `summary.csv`, `aggregate.json`, `report_aggregate.json`: paired
  Dense, SAF3R, fixed10 and adaptive pose accuracy, runtime and actual density.
- `per_layer.csv`, `per_scene_layer_head.csv`: budgets, C/S, cross-layer head
  matching, reuse/exploration/promotion counters and history lifecycle.
- `diagnostics_report.md`, `diagnostics_aggregate.json`,
  `diagnostic_independent_errors.csv`, `diagnostic_stages.csv`: separate untimed
  64-query validation and stage timings. Online 32-query budget calibration is
  still INCLUDED in formal inference timings.
- `diagnostic_prediction_verification.json`: all 14 extra diagnostic predictions
  exactly match their respective formal predictions.
- Each scene's `result.json`: complete input frame paths/indices, model/runtime
  settings, formal repeat samples, pose metrics, decisions and pose fingerprints.
- Each scene's `source_manifest.json`: SHA256 hashes of the evaluated sources.
  Those source files are included in this repository. The local duplicate
  `source_snapshot/` is not uploaded; hashes match the evaluated code here.

Retained locally, not uploaded:

- Prediction `.npz` files and duplicate per-method/layer JSON files.
- Failed/preflight output directories and transient worker logs/status files.
- Raw Chess frames 0/20 QKV, RoPE positions, masks, hierarchies and histories in
  `outputs/adaptive_block_value/chess_0_20_replay/`.
- Raw controlled-sensing tensors under `outputs/coarse_sensing_review/`.
- Dataset images, depth, poses, meshes and model checkpoint weights.

Regenerate raw captures with `scripts/capture_adaptive_block_value.py`; reproduce
the formal evaluation with `scripts/benchmark_adaptive_block_value.py` and the
configuration/commands in [the scheme description](../../../docs/experiments/adaptive_block_value.md).
The Chinese description is [here](../../../docs/experiments/adaptive_block_value_zh.md).
The current method does NOT meet both Dense accuracy and SAF3R speed; this
limitation is deliberately preserved in the reports rather than hidden.
