# Overall90% sparsity experiment

This is a budget/cluster-count experiment on the historical estimator,using
`history_sparse90_selected_20261106.json`. It does not claim quality preservation
before the full six-metric test. Its report is
`outputs/history_sparse90_20261106/REPORT.md`.

## Budget definition

`total_edge_retention=True` interprets each layer retention as a cap on **unique
formal Attention QK edges after special-Query correction**. With N tokens,S
special Keys and D full-Key special Queries:

`cap=floor(rate*N*N)`
`ordinary_key_cap=floor((cap-D*N)/(N-D))`
`patch_budget=ordinary_key_cap-S`.

The selector enforces its ordinary per-Query Key cap. Thus final unique edges are
at most the requested layer cap,even though special Queries retain all Keys.
The old default interpretation is unchanged when the flag is false. Protected-
frame variants are explicitly excluded from this geometry formula.

Selected total-edge rates are8% forL0–5,12% forL6–17,8% forL18–23:average10%,
so overall sparsity is at least90%. This is not a requirement that every layer
have90% sparsity. Unused budget and integer floors can make actual sparsity
slightly greater. Auxiliary observations and physical duplicate work are charged
to runtime;the90% number is not a claim about all auxiliary FLOPs.

## Independent choice and precision

Compare four overall10% schedules (uniform,scaled former middle schedule,
8/12/8 and6/14/6) and6%/8%/10% cluster counts on existing ScanNet64/TUM64 RGB.
The proxy is24-layer local projected Attention error against same-QKV full-Key
Attention at deterministic heldout Queries. No target scene geometry/pose truth
is used. The best equal-sequence proxy is8% clusters plus8/12/8 edge rates.
Different TUM128RGB validates the selected candidate before target quality.

The90% candidate uses `raw_masked`,which preserves original FP16 relative-logit
evidence. It does not reuse the14% engineering prototype's BF16 probability
requantization. Same-QKV/same-Mask tests verify exported relative logits and
shifts are bitwise equal to the original gather exporter. Formal AV and selected
raw normalizers are also checked,including large compensation weights and tail
Queries. Current-Key features use the old FP32 projection rather than the fused
Tensor Core projection. The main-path optimization changes layout,lifetimes and
batching,not the estimator equations. The standalone9 exporter tests (including actual16-head/64-dimension multi-tile cases) plus12
integer-budget tests pass.

## Hard acceptance

Quality reference is the completed14% historical result at e4b72294,declared
before target scoring. All six dataset macro metrics must be non-worse on both
7Scenes7 and ETH3D11. A2% reconstruction check is only supplementary and cannot
replace this gate. Also disclose the completed14% engineering experiment and
fast14 quality,without rerunning either control quality.

The user authorized fast14 paired timing only. New quality and fast14 timing run
sequentially on the same physical GPU/input,one warmup and three complete
forwards. Dense/SAF3R are not rerun;Dense paired speedup is N/A. Report every
scene,macro mean,all six metrics,memory,actual densities and any failed gate.
No target-score retuning or choice of a best repeated quality run is allowed.
