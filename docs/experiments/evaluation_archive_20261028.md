# Evaluation archive: 2026-10-28

These experiments used production revision `06c5e0c760b34b34c3f1a715a99417dd0406fd76`.
Local evaluation provenance/reporting commit: `c88382bc45781be6270e4e7f76b929f13dcb5333`.
Per-run source snapshots/hashes identify the actual evaluated code. Later remote
History implementation plans must not be attributed to these measurements.
No production selector changes are included in this archive.

## Archived evaluations

1. Joint History/Exploration, all seven 7Scenes scenes, 128-frame inputs:
   `outputs/joint_history_06c5e0c/comparison.md` and `comparison.json`.
   Per-scene six metrics, paired Dense timing and GPU-contention records are retained.
2. Matched Dense/SAF3R on 7Scenes (7 scenes) and ETH3D (11 scenes):
   `outputs/saf3r_top4_two_datasets_20261027/summary.md` and `summary.json`.
   SAF3R uses lazy first-correspondence-head per-frame Top4 and the ETH3D profile,
   NOT the paper's DINO-feature selection. ETH3D actual inputs contain 14–76 frames.
   `progress.json` records the original attempt; `retry_progress.json` records the
   successful retries. Result completeness is determined from final `result.json`,
   not the historical failure status of an earlier attempt.
   Initially unmonitored timings are retained as raw observations, not validated
   uncontended dataset speedups. Monitored fire/electro retries are valid pairs.
3. ScanNet-50, public non-lazy DINO64 configuration, max2048 candidate cap:
   `outputs/scannet50_paper_dino64_20261027/summary.md` contains all available runs;
   `matched_partial_summary.md/json` and `matched_partial_per_scene.csv` contain
   the controlled comparison. This is NOT a complete ScanNet-50 evaluation and
   NOT a recovery of the paper's unpublished scene-specific K choices.

## ScanNet completeness and metric definitions

| Frame cap | Dense complete | SAF3R complete | Common completed scenes |
|---:|---:|---:|---:|
| 300 | 48/50 | 44/50 | 42 |
| 500 | 49/50 | 0/50 | 0 |
| 700 | 0/50 | 0/50 | 0 |

500-SAF3R and both700 configurations failed preflight with CUDA OOM, so the
remaining scenes for those configurations were not started. Additional300 and
500-Dense scene OOMs are retained in `failure.json`; `blocked_jobs.json` records
not-started jobs with six unavailable fields and concrete reasons. No frame,
resolution, retention or K fallback was applied. All processes have exited.

The42-scene common300-cap subset has equal-scene means:

| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 12.77721 | 80.47573 | 0.21545 | 0.24792 | 0.23168 | 15.87184 |
| DINO64 | 9.64000 | 77.28279 | 0.21235 | 0.24028 | 0.22632 | 15.35060 |

Timing is restricted to36 uncontended matched pairs: Dense72.8585s versus
SAF3R27.9239s, a2.6092x speedup (ratio of scene-mean warmed medians). Each run
uses one warmup and three measured complete forwards, including DINO selection.
This is NOT a full50-scene speedup or an H100 comparison. Max allocated memory
over the42 quality scenes is25.0903GiB Dense and28.6475GiB SAF3R.

ScanNet uses the official valid-pose uniform sampling, predicted-camera depth
unprojection, confidence>=1, first-camera gauge and FastVGGT bounding-box scale
and center alignment. No GT-camera depth fusion or ICP is introduced. The user
six-metric extension uses100k point cap,5cm voxel,5cm F1 threshold and0.5m distance
clipping. CD=(ACC+Comp)/2 with unsquared distances; official sum-CD is stored
separately. These F1/CD results must not be equated to another benchmark protocol.

All raw AUC/F1 values are fractions; human-readable tables use percentages.
Scene lists, frame manifests, metrics, failure logs, source hashes and small
pose/distance arrays are archived. Datasets and checkpoint weights are excluded.

## Entry points

- `scripts/run_joint_history_evaluation.py`
- `scripts/run_saf3r_two_dataset_eval.py`
- `scripts/resume_saf3r_two_dataset_eval.py`
- `scripts/run_scannet_paper64.py`
- `scripts/report_scannet_paper64_partial.py`

Runners require fresh output directories and the documented local dataset paths.
ScanNet numerical/reporting tests: `tests/test_scannet_paper64.py` (4 passed).
