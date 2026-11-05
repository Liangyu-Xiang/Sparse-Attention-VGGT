# Numerical stability investigation

Fresh-process autotuning on/off prefix comparison is bitwise equal for the sampled first/last-frame patch projection and final DINO prefix tokens. This diagnostic does not support attributing the quality difference to early convolution autotuning. It does not prove equivalence of every later convolution.

The original reference itself is not repeat-stable. Same checkpoint/input in one model, no algorithm/config changes: first different retained mask at L14; head-mean Values first differ at L15; final pose relative L2=0.0017321643. An optimized precision-path run first differs in mask at L12 and in Values at L13; a subsequent run first differs in mask at L17 and Values at L18. These comparisons retain all real supports; no GT or quality scoring was used.

This demonstrates intrinsic numerical variability, not proof that any specific atomic kernel caused all differences. The selector contains FP32 atomic reductions and discrete choices, which warrant further deterministic investigation. Repeated quality scores were not sampled to select a favorable result.

A second candidate restores original Attention stage1 and original source clustering stores, disables convolution autotuning, and retains only exact-tested RoPE/residual fusion plus allocator reuse. It is a separately frozen engineering candidate, not a budget/representation-objective adjustment. Reference quality is reused; only SAF3R timing and the changed method are rerun.

The first candidate fails the strictly frozen absolute <2% 7Scenes macro criterion because F1 improves by2.3224%; other macro distance changes are within2%. It additionally has real per-scene deviations above2%, including regressions. ETH3D macro and all per-scene point-cloud changes pass2%. These failures are retained and never relabeled as complete acceptance.
