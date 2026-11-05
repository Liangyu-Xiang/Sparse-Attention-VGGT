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
