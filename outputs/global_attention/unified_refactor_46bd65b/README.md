# Updated unified branch evaluation: 46bd65b

Remote branch: `refactor/unified-reuse-exploration`.
Pinned commit: `46bd65b5af355ba6d383947e6f12c93fd847e62d`.
Previous evaluated commit: `3afe65a038b01e34732493beddf9bf9eb63af76c`.
Isolated checkout: `/data_NVME1/mmc_lyxiang/3D/SAF3R-unified-eval-46bd65b`.
No changes are pushed, no previous experiment results are overwritten.

## Frozen protocol

- Seven 7Scenes scenes; 128 uniformly sampled frames per scene. Each run checks
  exact checkpoint, frame list, and preprocessing metadata against the frozen
  historical Dense input.json before inference.
- Original checkpoint: `/data_NVME1/mmc_lyxiang/3D/SAF3R/checkpoints/vggt/model_tracker_fixed_e20.pt`.
- Evaluate pose and recon_unposed: AUC@3, AUC@30, ACC, Comp, CD, F1, plus equal-scene
  macro averages only when all seven scenes finish. AUC/F1 percentages, distances
  metres; CD=(ACC+Comp)/2 unsquared; threshold .05m, downsample .0078125m.
- Unmodified upstream default hyperparameters: total retention .20, source_width
  32, local_iterations 2, window_groups 8, canonical grid 2x2 per frame,
  candidate_fraction .25, dual_iterations 16. All24 global layers sparse.
- No test-truth tuning, Dense fallback, reduced frame count, or old baseline reruns.
- One warmup and three complete forward measurements include initialization,
  clustering, selectors, history, and evidence; no per-layer synchronization.
- GPUs4 and5 only, one process/card, UUID-bound. Busy cards wait. Do not reset a
  GPU/driver or stop unrelated processes. Driver/kernel-wait safeguards enabled.

## Changes evaluated

History is now a direct query-group relation candidate bank, without the former
per-Fine-cluster cap or Value-based screening. Current canonical probabilities
and persistence scale previous salience within a canonical region; relation and
Value Drop/Rep/Exact candidates compete for the same budget. History is compacted
once per layer. Shared representative duplicates are merged with population
accounting. These are upstream changes, not evaluation-time parameter tuning.

## Compatibility overlay

Same necessary overlay as the prior evaluation: disable autocast for FP32 sparse
mapping; pad the Triton probe dot to16 rows but retain only the original four
probes with denominator4. Evaluation CLI accepts the original checkpoint path
and strict expected input.json. Full diff and source hashes are saved.
Nine upstream CPU/GPU tests and two coordinator report tests passed before dataset launch. Two-frame preflight is
diagnostic only: all six quality fields unavailable with explicit reasons.

## Reports

`comparison.md` and `comparison.json` compare against frozen historical Dense,
SAF3R lazy per-frame Top4 (ETH3D profile), History80+population, and original
reuse+residual. This SAF3R is not paper DINO-feature TopK reproduction.
Prior branch results remain in `../unified_refactor_20261025/` for revision comparison.
`queue_status.json`, `logs/`, and per-scene `quality128/` expose progress/raw data.
No completed result is claimed until pose and reconstruction both finish.
Current-vs-Dense speedup is N/A without contemporaneous paired Dense timing;
historical timing ratios are not fresh speed measurements.
