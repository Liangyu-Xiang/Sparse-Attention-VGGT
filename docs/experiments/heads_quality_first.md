# heads128: quality-first failure localization

This priority run does not benchmark speed. It uses the exact128-frame heads
input/checkpoint of the matched Dense baseline, one inference forward, no warmup
and no repeated timing. All six required quality metrics are evaluated; speedup
is unavailable by design. Concurrent elapsed seconds are not speed comparisons.

The first three jobs apply the old whole-Parent/history80 scheme only to:

| Ablation | Sparse-output global blocks | Original Dense-output blocks |
|---|---|---|
| Shallow only |0–9|10–23|
| Middle only |10–16|0–9 and17–23|
| Deep only |17–23|0–16|

The new Attention-first scheme receives the same three ablations next. Both
schemes maintain their own Value/history states at inactive layers but return
original Dense output there. They do not use Dense-derived historical masks.

Interpretation must distinguish:

1. Large degradation with only one stage sparse: evidence that this stage is
   sufficient to cause degradation under this rollout.
2. Each stage individually close to Dense but the full scheme much worse:
   evidence of cross-stage interactions/error accumulation, not proof that one
   isolated layer is the cause.
3. AUC versus reconstruction failures: report both, not only AUC30. A stage
   can preserve coarse pose while damaging AUC3, depth or reconstructed F1.
4. Layer-output divergence is observational; finer stage/substage interventions
   are needed before naming a specific failing layer or claiming a root cause.

The priority dispatcher pauses only the broad campaign controller's future
dispatch (SIGSTOP), not already-running children, and resumes it on exit. It
admits at most two CUDA processes/GPU and requires20GiB free before an old
scheme job starts. New-scheme jobs wait for an exclusive GPU because an earlier
standalone fire run hit OOM; observed current memory is not a peak guarantee.
No inputs, retention or mathematical branch rules are altered to hide OOMs.

Live artifacts:

`outputs/attention_first_residual_value/heads_priority_quality/status.json`

`outputs/attention_first_residual_value/heads_priority_quality/comparison.md`

The comparison initially contains matched Dense and the previously completed
full old scheme; stage rows appear only after full task metrics complete.
Missing rows remain explicitly pending, never interpreted as zero error or a
successful ablation. Source and schedule hashes are recorded in the manifest.
