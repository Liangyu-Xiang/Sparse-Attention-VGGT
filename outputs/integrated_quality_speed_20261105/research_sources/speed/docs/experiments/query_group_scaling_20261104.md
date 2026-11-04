# Shared Value loss: actual Query-group size 128 and 256

Source branch: `codex/shared-value-cluster-scaling-20261104`, commit `d7f535ca`.
This isolated worktree adds `query_group_size` to the active shared Value sampled-history path.
No experiments use the old 32-query selection groups. The default remains compatible for existing callers.

## Controlled change

`configs/evaluation/reuse_shared_value_loss_q128.json` and `reuse_shared_value_loss_q256.json`
are copies of `reuse_shared_value_loss_256.json`, differing only in `query_group_size`.
The source configuration's suffix `256` refers to the HISTORY SAMPLING INTERVAL,
not to the previous Query-group size (which was 32).

The true grouping controls the number of selection decisions, previous-support rows,
coarse-probe positions, prior/state-loss rows, support packing and history consumers.
128 adjacent queries now share one selected support; 256 adjacent queries share one support
in the other intervention. Their weights and outputs remain individual per query/head.
The attention kernel's 32-row execution tiles index the enclosing 128/256-query support;
they never perform independent 32-query selection. This avoids confounding the experiment
with a separate matrix-tile tuning change.

The canonical 2x2-per-frame grid, flat Value clustering, nominal 20% Cluster count,
20% maximum selected-Key retention (specials included), four state choices, 16 dual
iterations, 64-group window, graph policy, and unit selected-Key weights are unchanged.
Coarse probes remain four evenly spaced actual queries, at group offsets
`group_size * [1,3,5,7] / 8`; tail positions clamp to the last actual query.
History uses the first probe in sampled groups, at a fixed 256-query interval:
every second group for size128, every group for size256. Each unsampled group retains
its own uniform coarse prior, without copying another group's support or evidence.
This necessarily increases the fraction of groups with actual measured history as
Query groups become larger; the physical history sampling interval is not retuned.

For 128 frames at 392x518: total tokens=133248, patch tokens=132608, special keys=640,
patch budget=26009 and Cluster count=26009. Size128 gives 1041 groups, size256 gives
521 groups (the last has 128 queries); both sample 521 groups for history.
No oversized cluster is silently truncated; discrete allocation may leave unused budget.

## Verification and execution

`tests/test_query_group_scaling.py`: six CUDA tests. Explicit PyTorch attention checks
actual 128/256-query supports, per-head normalizers and exported debiased log probabilities,
including padded/invalid keys and short tails. Both sizes additionally execute all 24
production global layers with sparse state propagation, unique-key and reset assertions.

Environment: `/data_SSD2/mmc_lyxiang/anaconda3/envs/3d/bin/python` (PyTorch2.7.1/cu118,
Triton3.3.1). The older `fastvggt` environment cannot import this branch's coarse
libdevice module; that failed implementation-validation attempt is archived separately.
It produced no quality or speed result and was not used for the experiment.

Final user-requested scope: only three 7Scenes scenes, heads/chess/stairs, with128 uniformly sampled frames each.
Run `python scripts/evaluate_query_group_scaling.py --gpus 3 4 5` in this worktree.
The initial broader queue was stopped at the user's request; completed or interrupted
out-of-scope runs are preserved and listed in `excluded_runs.json`, and are excluded from every final table.
No ETH3D or additional7Scenes scenes are part of the final comparison.
Each scene's two variants execute serially on the same physical GPU, with alternating
variant order across scenes. Strict GPU leases check no foreign context at launch and
monitor every0.5s, aborting if another GPU job joins during the timed phase.
One warmup and three complete BF16 forwards with TF32 disabled; all reset, clustering,
coarse observation, selector, history, graph and output costs count.

Per-run artifacts archive exact input paths/indices, checkpoint path, configuration,
source hashes/tar, all24 layer records, raw pose/recon_unposed metrics, timing samples,
peak allocated/reserved memory and six-metric tables. Monitors archive physical GPU UUIDs.
AUC/F1 are fractions in raw JSON and percentages in report tables; ACC/Comp/CD are metres.
CD is the average of unsquared ACC and Comp. Reconstruction thresholds/downsampling
are recorded per run. Three-scene equal averages are explicitly NOT the full7Scenes macro.

Unchanged Dense and SAF3R controls are not rerun, following this branch's AGENTS.md.
Without contemporaneous paired Dense timing, end-to-end Dense speedup is N/A with a reason.
Old RTX5090-D group32 timings are not a local RTX4090 baseline. No cross-hardware
speedup claim or unmatched accuracy degradation is manufactured.

Output: `outputs/query_group_scaling_20261104/report.md` and `summary.json`.

## Measured result

All six selected runs completed pose and recon_unposed, with all six metrics available.
All six strict lease monitors report no foreign GPU overlap. Source hashes and exact
paired inputs/protocols match. The first six correctness checks passed.

The equal-scene mean of warmed median full-forward times is13.73605s for group128
and13.96404s for group256. 256/128 speedup is0.98367x (about1.66% slower).
AUC@30 changes85.25919% ->85.01941%; F1 changes46.94925% ->46.47900%.
CD improves0.16649374m ->0.16249687m, primarily through stairs completeness.
Per-scene timing and quality changes are mixed; this sample does not establish
a consistent advantage for group256. It does not compare local speed against group32.
