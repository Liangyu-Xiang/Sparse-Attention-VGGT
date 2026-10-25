# All executed History / no coverage ablation

Pinned upstream HEAD:1a7bb41d955da284eded195a2e0b11faf49c2ec3.
Read the superseding all_edge_history_no_coverage_handoff.md completely.
All model changes are upstream changes. Local changes are diagnostic/reporting
bugs and evaluation compatibility only. No retention, solver, grouping, Value,
Rep/Exact, population compensation or final sparse-attention algorithm changes.
The archived four-effective-probe padded kernel and FP32 sparse mapping wrapper
are identical in semantics to previous evaluations. Exact historical checkpoint
path and input-list equality checks are enabled for the isolated worktree.

Stages:0validation ->1three two-frame pipeline diagnostics and strict bank sanity
check ->2seven-scene128 quality ->3layer analysis ->4comparison ->5causal report.
No quality job starts unless all three Stage1 pipelines are valid. Busy GPUs wait;
all idle non-quarantined cards may be used, one worker/card, no unrelated kills.
Two-frame inputs:seq-01 frames0,20; layers9–16; target patch Top20.
Quality:exact previous128-frame lists, uniform seed0;1warmup/3complete measured
forwards; original checkpoint; pose+recon_unposed; capture disabled.
ACC/Comp/CD metres; CD=(ACC+Comp)/2 unsquared; threshold.05m,downsample.0078125m.
Report all six metrics for every run, with explicit N/A reasons for diagnostics
or failures. No contemporaneous Dense rerun: fresh paired speedup is N/A;
historical timing comparisons are not new speedups. SAF3R local lazy per-frame
Top4/ETH3D profile, not paper DINO feature TopK reproduction.
Do not optimize dedup/budget-overcount confounds or full exploration cost away.
