# Integrated main, 20% clusters: equivalent inference engineering

Base e0107c38, exact frozen ETH3D cluster20 configuration. Independent branch
codex/cluster20-speed-20261105. VGGT checkpoint and native evaluation unchanged.
Cluster fraction .20, layer retention 10%/22%/10%, query groups256, current-layer
updates, cluster_population compensation, all four states and16 budget iterations
remain. No GT-guided parameter tuning. Dense and SAF3R quality are reused.

Opt-in `optimized.json` versus `reference.json` changes engineering only:

- Masked Triton stores replace boolean-index compaction in source clustering.
  All distance, tie, empty repair and representative arithmetic remains native.
- Keep CUDA allocator cache between aggregator blocks (released Python activations
  are still freed); original native behavior remains the default.
- Fuse LayerScale and residual addition with explicit FP32 multiply/add rounding.
- Reuse existing workspace `rope_fast.py`, preserving intermediate dtype rounding;
  derive native raster position extent freshly from each input shape. No head
  correspondence, altered probability formula, cross-layer frozen states or
  reduced probes are introduced.
- Equivalent attention launch uses two pipeline stages;24 real first-window tests
  had bitwise-equal output and history. Slower split/tile alternatives rejected
  using runtime and intermediate numerical checks only.
- Enable cuDNN runtime algorithm search for the native dense convolution heads;
  TF32 policy remains unchanged. Algorithm search/cold cost is recorded separately.

Validation:16 deterministic cluster trajectories,18 residual dtype/shape cases,
7 fused RoPE cases pass. Initial residual fusion lacked explicit rounding and was
rejected; corrected version passes bitwise comparisons. Original failed diagnostic
and deterministic-cuBLAS setup attempt remain in outputs.

Pilot on dedicated RTX5090D:reference heads128 median7.211s; allocator reuse6.800s;
residual/pipeline6.683s; convolution autotuning6.515s; final fused RoPE about6.12s.
These intermediate pilots used GPUs4/5/6 and are not all paired estimates of
individual incremental gains. Formal per-scene same-GPU SAF3R comparison and
six-metric quality gates are pending; do not claim completed acceptance.

User authorized SAF3R timing only, no Dense rerun. Speedup vs original VGGT Dense
therefore N/A, historical timings are labeled. SAF3R profile is repository lazy
first-correspondence Top4 per frame, not paper DINO TopK.

Campaign records: main workspace outputs/cluster20_speed_20261105. Fresh7Scenes
reference20% quality is necessary: prior complete7Scenes records were12% clusters;
ETH3D reference20% reuses the already complete11-scene batch. Only optimized
method geometry is rerun on ETH3D. All six metrics, per-scene data and equal-scene
macros will be retained, including failures. Acceptance: speed<=paired SAF3R,
128-frame full forward<9s, absolute relative macro ACC/Comp/CD/F1 change<2%.
Per-scene deviations are separately reported; they are not hidden in macros.

## First complete campaign and numerical stability

All43 new runs completed (25 complete quality runs +18 SAF3R timing-only runs),
with11 additional ETH3D reference rows reused. See
`outputs/cluster20_speed_20261105/report.md` for the full54 method/scene records.
All18 optimized scenes are faster than paired native SAF3R.7Scenes averages:
reference7.388654s,SAF3R6.779607s,first candidate6.161170s (1.1004x vs SAF3R).
ETH3D:SAF3R1.579174s,candidate1.272822s (1.2407x).

Strict absolute macro reconstruction-change gate: ETH3D passes (largest0.149%);
7Scenes fails because F1 improves by2.3224%, exceeding the absolute2% threshold.
7Scenes distance macro changes are ACC−1.280%,Comp+1.625%,CD+0.068%. There are
also substantive per-scene changes, including degradations; these are not hidden
by an improved mean F1. The candidate is NOT marked fully accepted.

Prefix on/off autotuning diagnostics were bitwise equal for the sampled first/last
frame patch projections and DINO outputs. This rules out the suggested prefix
explanation in that diagnostic, not every later convolution. Full-mask numerical
captures reveal original reference self-repeat differences starting at L14, before
head-mean Values change at L15. Final pose relative L2 is0.001732. This is evidence
of intrinsic numerical variability, not a proof that one specific reduction caused
all geometry differences. Details: numerical_stability.md and trajectory/result.json.

A separately frozen conservative candidate (`conservative.json`) retains only
allocator reuse,exact-tested RoPE and residual fusions. It restores original
Attention stage1,original source-clustering stores and native convolution selection.
Reference quality is reused; authorized SAF3R timing and only changed-method quality
are rerun. It is not a budget/loss/probability adjustment. Results remain pending.
`precision_preserving.json` describes a trajectory diagnostic, not a selected model.

Common-forward optimization is enabled for our candidate, while SAF3R retains its
original execution. This is an implementation runtime comparison, not proof that
our sparse selector is intrinsically faster than every optimized SAF3R implementation.

## Final conservative campaign

All36 new runs completed:18 changed-method quality runs +18 authorized SAF3R
timing-only runs. All reference quality is reused. Conservative full-forward
averages:7Scenes6.427234s vs SAF3R6.668902s (1.0376x);ETH3D1.323959s vs
SAF3R1.601425s (1.2096x). All18 per-scene warmed median speed comparisons pass.
7Scenes warm peak allocated14.979GiB;ETH3D10.959GiB. Mean cold forwards8.162s
and3.030s respectively;cold/warm samples remain separate.

Quality is NOT fully accepted:7Scenes macro Comp+2.7321% exceeds2%. Other
7Scenes macro reconstruction changes:ACC−0.6043%,CD+0.9442%,F1+0.4972%.
ETH3D macro reconstruction changes all stay below2%,but electro,relief,terrains
have per-scene exceptions. No candidate is promoted as meeting every constraint.

The initial prefix-autotuning hypothesis was not supported by the prefix diagnostic.
Reference self-repeat mask drift remains a material numerical-stability limitation;
its exact single-kernel cause is not established. Neither budget tuning nor
cherry-picking repeated quality predictions was used to hide these failures.

Chinese final report:outputs/cluster20_speed_20261105/final_report_zh.md.
Conservative detail:outputs/cluster20_speed_conservative_20261105/report.md.
Both archives include per-scene raw JSON,six-metric tables,CSV,source hashes and
GPU monitors. Reused references are materialized in the archive;large source tars
and diagnostic feature tensors remain in the original local outputs,with model
source preserved by Git commit and hashes. Conservative plan.created_utc inherited
the earlier plan timestamp;provenance_note.json preserves the correction using
the original manifest file mtime,without rewriting the frozen plan.
