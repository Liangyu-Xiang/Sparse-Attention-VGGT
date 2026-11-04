# Direct Sparse log-probability history

Independent branch `codex/direct-probability-history-20261104`, based on published
`6c9ca01c`. Opt-in config `configs/evaluation/reuse_direct_probability.json`.
No defaults or other worktrees changed.

The user authorized replacing the legacy v8 history with sparse probability,
calibration and missing-token compensation. The new history is a modeling change.
Parameters are frozen before quality evaluation; no ground-truth tuning.

For head h and selected key i, actual compensated sparse probability is
`P_hi=exp(z_hi+b_i-LSE_h)`. Export
`log a_i=logmeanexp_h(z_hi-LSE_h)` directly from the four-probe QK and final
Attention LSE. Thus a_i is the head-mean sparse P with shared assignment bias
removed. Evaluate the expression before exponentiation; log P must not be
rounded to half and then divided by exp(bias). Zero-assignment keys use the
finite raw/LSE score rather than undefined 0/0. Specials remain in the final
sparse denominator; there is no patch-only Softmax normalization.

For each Grid r, coarse missing-member score is
`log u_r=logmeanexp_h(pooled_grid_logit_hr-LSE_h)`. The existing current-layer
coarse QK is reused: this adds no independent Grid QK. Complete the old Grid
with `D_r=sum_selected(a_i)+(size_r-count_r)*u_r`. Use stable log-sum-exp,
then cache `log(a_i/D_r)` in signed FP16 and `u_r/D_r` in FP32. The log cache
avoids the FP16 probability underflow seen in the failed beta prototype.

At the next layer use the original M5c confidence
`ratio=(G_current+EPS)/(G_previous+EPS)`;
`beta=ratio if ratio<1 else ratio^-2`.
Selected member estimate is
`G_current*(beta*exp(cached_log_conditional)+(1-beta)/size)`;
unselected member estimate is
`G_current*(beta*cached_uniform_conditional+(1-beta)/size)`.
Then combine the four probes with the original unique-tail weights and construct
cluster statistics for the unchanged state loss and budget solver.

This avoids assigning the entire current Grid mass to selected members via a
hard cap. It changes v8's mean of per-head conditionals to a ratio of head-mean
scores. There is no claim to exact Dense or exact old-v8 probability recovery.
Residual candidate exploration uses four-probe Grid means instead of the old
all32-query Grid observation. Its compensated and unit-weight summaries are
computed from compact exported log scores, without another QK.

The main AV stays Grid-free Direct Triton Flash-style Attention. A separate
four-probe export still recomputes QK (the faster existing B strategy), including
Specials; this is not a claim of zero auxiliary QK. Removed: legacy all-query
selected-QK observation and a second four-probe cache QK. Compact per-Grid
reductions and current coarse Grid estimation remain necessary for calibration.

Retention20%,adaptive-local clustering,four probes,Drop/Rep/History/Exact,
16-round budget,state losses and compensation remain unchanged. The new cache
is bounded using the existing resident/spill mechanism; its format is signed
FP16 log values,not positive-only u14. Numerical tests cover bias60,zero
assignment weights,ordinary half-probability underflow,arbitrary Grid runs,
invalid support holes,empty/full/partial Grids and fresh-input CUDA Graph replay.

Validation records preserve two diagnostic harness failures (FP16 bitwise oracle
rounding and shadowed CLI args) and a pilot setup failure (missing dataset
symlink); none caused estimator or quality-gate changes. The final numerical
validation passes. These diagnostics have all six geometry fields N/A.

Full evaluation: changed method only,all7 7Scenes128-frame scenes and all11
ETH3D14–76-frame scenes,pose+recon_unposed. Same checkpoint,image lists and
protocols audited against the preceding Direct campaign. Raw six metrics,
per-scene tables,equal-scene macros,profile,source hashes and exclusive GPU4–7
monitors are saved in `outputs/direct_probability_quality_20261104`.
End-to-end Dense speedup N/A: no unchanged Dense rerun is authorized by default.
Historical timing values are identified by campaign and are not paired timings.
The main report must describe measured quality changes without calling a
completed evaluation a quality acceptance. Default backend stays unchanged.
