# Published experiment artifacts

The selected implementation is documented in `docs/experiments/layer_shared_av_zh.md`.
Start with `report/report.md`; `selected_joint_v3_n128` contains the final seven-scene
128-frame results. Other named suites are baselines, calibration trials, or rejected
versions and must not be pooled with the final suite. Caveats are retained beside
the affected suites.

The repository retains lightweight results, configurations, pose predictions,
source hashes, and summary tables for provenance. Duplicate source snapshots,
large sampled-logit tensors, runtime logs, queue/status files, smoke runs, and
incomplete aborted-run outputs remain local. The aborted suite retains only its
explanatory marker.
Reproducing inference requires the external dataset, checkpoint, and GPU environment;
the offline calibration sweep additionally requires regenerating sampled logits.

The final method is experimental: it did not achieve usable accuracy or speedup.
