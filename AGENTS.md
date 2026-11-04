# Project experiment reporting requirements

For all subsequent experiments, every method/run must report camera-pose AUC@3
and AUC@30, and unposed point-cloud reconstruction ACC, Comp, CD, and F1. This is
a user requirement, not an optional reporting preference.

- Compare the same checkpoint, exact input frame list, preprocessing, pose
  protocol, alignment/fusion settings, and reconstruction thresholds.
- Save raw metrics and a human-readable six-metric table for each run, per-scene
  rows, and equal-scene macro averages for dataset evaluations.
- Label AUC and F1 as fractions or percentages. Report reconstruction distances
  in metres for 7Scenes. CD here is (ACC + Comp) / 2 using unsquared distances,
  not their sum or squared Chamfer distance.
- Record the reconstruction threshold/downsampling and SAF3R Top-K/profile.
  Top-128 is not the official default Top-4 configuration.
- Default quality runs must include pose and recon_unposed; depth is additional.
  Kernel-only, diagnostic, OOM, interrupted, or explicitly pose-only runs must
  still output all six fields as unavailable with a concrete reason. Never
  invent missing scores or call such runs complete quality evaluations.
- Record runtime, peak memory, sampling scope and completeness separately from
  quality. Historical noisy timings are not contemporaneous speed comparisons.
- Every method/run must also report end-to-end speedup relative to original
  VGGT Dense: Dense warmed median full-forward seconds / method warmed median
  full-forward seconds, under the same inputs/hardware/timing protocol. Include
  initialization, clustering, routing, history and reordering. Report Dense as
  1.00x; below 1x means slower. Missing paired timings are N/A with a reason.
- Never tune parameters against 7Scenes test truth or silently change the method,
  retention, or quality gates to improve a comparison.
- Paper comparisons must separate published-reference rows from local measured
  rows, require all seven scenes for 7Scenes macro results, and keep128-frame
  sparse evaluation separate from300-frame single-sequence or dataset results.
  Label lazy first-correspondence-head Top-K separately from the paper's DINO
  feature Top-K; repository defaults are not proof of identical paper settings.
  Never shift/scale local scores to match published baselines. Report method
  degradation against local matched Dense, and baseline reproduction drift
  against published Dense as two distinct differences. H100 paper timings are
  not local RTX4090 timing baselines.

## Evaluation scope: changed methods only

- By default, rerun only the method changed by the user. Reuse existing complete results for unchanged Dense, SAF3R and other controls; do not rerun them merely to obtain contemporaneous timings.
- Rerun an unchanged control only when the user explicitly requests that rerun. Missing contemporaneous speed comparisons must be N/A with a reason; historical timing references must be labeled with their batch and never presented as paired measurements.

## Default scope during method validation (user update, 2026-11-04)

- Use the fixed difficult-sequence subset in `configs/evaluation/validation_subset.json`: 7Scenes `stairs`, `pumpkin`; ETH3D `relief`, `relief_2`. Do not run complete datasets unless explicitly requested.
- This subset was selected as the two lowest-F1 sequences per dataset from the existing sampled32 baseline, without using current candidate results to select sequences. Keep it fixed across later validations.
- Recompute every baseline/control average using exactly the same selected sequences. Never compare a candidate subset mean with a historical full-dataset mean.
- Report all six quality metrics, per-scene rows and equal-scene subset means. Label these as validation subset results, not all-seven-scene 7Scenes or full ETH3D dataset macro results. Retain all existing protocol, unit, missing-metric and runtime rules above.
