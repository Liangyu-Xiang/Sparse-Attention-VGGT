# Pinned unified-refactor 7Scenes evaluation

Requested branch: `refactor/unified-reuse-exploration`.
Pinned upstream: `3afe65a038b01e34732493beddf9bf9eb63af76c`.
Isolated detached checkout:
`/data_NVME1/mmc_lyxiang/3D/SAF3R-unified-eval-20261025`.
The primary worktree and remote branch are not overwritten or pushed.

## Experiment scope

Only the NEW method runs on all seven 7Scenes scenes, uniform 128 frames,
checkpoint `/data_NVME1/mmc_lyxiang/3D/SAF3R/checkpoints/vggt/model_tracker_fixed_e20.pt`.
Exact historical `input.json` equality is checked before every measured forward.
Pose and recon_unposed; AUC@3, AUC@30, ACC, Comp, CD, F1 required.
Reconstruction threshold .05m, downsampling .0078125m, CD=(ACC+Comp)/2 unsquared.
One warmup, three measured COMPLETE forwards; all initialization, clustering,
selection, state maintenance and probe evidence count. No per-layer sync.
No test-truth tuning, budget reduction, Dense fallback or changed layer schedule.

Strict GPUs 4 and 5, bound by UUID:

- GPU4: GPU-a67db5fc-c14e-4884-1909-a7c7050ef267
- GPU5: GPU-917c95c3-9023-f077-f577-dae0f7f9d717

One process/card at a time. Busy cards wait; no other GPU is substituted and
no unrelated job is stopped. Health/kernel-wait safeguards stop further dispatch
on driver failures; no automatic GPU reset or driver operation is attempted.

## Compatibility overlay, not method changes

Upstream initially fails its CUDA all-layer test because outer BF16 autocast
reaches sparse addmm, which is unsupported in the pinned torch environment.
Also the four-probe Triton dot has M=4, below Tensor Core tile requirements.
The evaluation checkout therefore applies ONLY:

1. Explicit FP32/autocast-disabled sparse overlap × canonical-probability mapping.
2. Pad probe dot to 16 rows, masking all but the SAME four probes. Denominator
   remains four; tail clamping is unchanged. Verified against explicit four-query
   attention probabilities, not against a 16-probe approximation.
3. Evaluation CLI checkpoint override and strict historical input equality,
   for isolated checkout/path parity; generic diagnostic-unavailable wording.

No loss, selection state, persistence formula or hyperparameter is changed.
See `evaluation_overlay.patch`, source hashes in `manifest.json`, and per-run
source snapshots. The report identifies the pinned branch WITH this overlay,
not an unmodified-upstream runtime claim.

Validation before dataset launch: ten upstream CPU/GPU/report tests and one
additional probe-probability test passed; real two-frame/all-24-layer model
preflight passed. Preflight is a diagnostic: six quality metrics and paired
Dense speedup are unavailable, not a quality evaluation.

## Baselines and reporting

Dense, SAF3R, History80+population, and original reuse+residual are reused from
`outputs/global_attention/reuse_explore_20261021/quality128`. They are NOT rerun.
Their 28 raw records are checked for complete quality, matching exact inputs and
protocols, then SHA256-recorded in the manifest.

SAF3R is the local lazy first-correspondence-head Top-4 with
`configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json`, not a paper DINO
Top-K reproduction. No paper/local score shifting or H100 timing comparison.

`comparison.md` / `comparison.json`: 35 per-scene method rows and equal-scene
macro ONLY after all seven scenes for the corresponding method complete.
`queue_status.json`: running/queued/completed scenes and worker PIDs.
`quality128/<scene>/`: raw six metrics, timing, memory, sources, configs, inputs.
`logs/<scene>.log`: live forward and evaluation output.
`completion.json`: whether all seven NEW quality evaluations completed.

Historical runtime/speedups belong to their original phase. New method full
runtime/peak memory are recorded, but valid end-to-end speedup vs Dense is N/A:
the user explicitly requested no Dense rerun, so there is no contemporaneous
same-UUID Dense timing. Do not silently label a historical-time ratio as a
new matched acceleration measurement. Quality comparisons remain valid under
the checked shared inputs/checkpoint/protocol.
