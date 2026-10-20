# Mandatory quality comparison protocol

Every subsequent method/run in this project must output six quality fields:

| Field | Definition | Display |
|---|---|---|
| AUC@3 | Existing relative-pose error CDF AUC up to 3 degrees | percent, higher better |
| AUC@30 | Same pose protocol, up to 30 degrees | percent, higher better |
| ACC | Mean unsquared nearest-neighbour distance, predicted cloud to GT | metres, lower better |
| Comp | Mean unsquared nearest-neighbour distance, GT to prediction | metres, lower better |
| CD | (ACC + Comp) / 2, matching `recon_unposed.overall` | metres, lower better |
| F1 | Harmonic mean of reconstruction precision and recall | percent, higher better |

Also report end-to-end speedup relative to VGGT Dense: Dense warmed median
complete-forward seconds divided by method seconds, with identical input,
hardware and timing settings. Dense is 1.00x; values below 1x mean slower. Count
all initialization, clustering, selection, history and data-reordering costs.
Use one warmup plus three measured forwards and preserve raw samples. The
dataset-level ratio is mean Dense seconds / mean method seconds; it is not the
arithmetic mean of per-scene ratios. Do not use historical unmatched timings.

For 7Scenes, use the repository SAF3R/DA3-Bench **unposed** reconstruction:
predicted cameras, existing alignment and fusion, 0.05 m precision/recall
threshold, and 4/512 m evaluation voxel size. Do not switch to GT poses to
improve a result. Save the exact sampled input frame list and model checkpoint.

Use per-scene results and seven-scene equal-weight macro averages. Each scene is
the default dataset entry, sampled uniformly to 128 frames, seed 0; this does
not mean exhaustive testing of every original video sequence/frame. All three
methods must use identical input paths/indices, shapes and checkpoint. Label
SAF3R's head profile and Top-K explicitly: the existing project comparison uses
Top-128; the shipped official configuration uses Top-4 and is a different run.

The active evaluator `scripts/evaluate_fused_shared_av.py` now defaults to pose,
depth and recon_unposed and writes `quality.json` and `quality.md` beside every
result. Missing, nonfinite, unsupported, failed or diagnostic-only metrics must
be `null`/N/A with a reason, never zero-filled. Such a run is not a full quality
evaluation. `scripts/quality_metrics.py` defines the canonical schema.

Current comparison command:

```bash
/data/mmc_syang/miniconda3/envs/fastvggt/bin/python scripts/compare_three_method_quality.py \
  --output-dir outputs/parent_region_av/three_method_quality_7scenes128 \
  --gpus 4 5 6 --saf3r-topk 128 --include-current
```

With `--include-current`, this computes fresh VGGT Dense, SAF3R Top-128 and
History80 results, running all three methods of a scene sequentially on the
same GPU. The original frozen History80 results remain untouched and provide
the input/config reference. `--resume` validates and preserves completed runs.
The report checks exact inputs and excludes incomplete methods from full-dataset
macro rows. Without `--include-current`, the old current-method quality can be
shown but its historical timing is not used for a speedup claim. No parameter
tuning is performed on these test results.
