# History evidence engineering optimization

The independent `teacher_history_probability_tiles_w128_20261106.json` configuration
preserves the preceding teacher-calibrated history estimator,states,losses and
budgets. Default old backends remain available. Frozen results are in
`outputs/history_evidence_speed_20261106/REPORT.md`.

## Reuse the actual AV tile

For each original head/probe,formal sparse AV computes
`r_t(e)=exp(z(e)+log(w(e))-m_t)`. It already casts this tile to BF16 for the
Tensor Core AV product. Save precisely that BF16 tile and its FP32 shift `m_t`,
instead of gathering FP32 raw logits and encoding FP16 relative logits.

Accumulate the uncompensated selected partition alongside the original weighted
softmax: `D_raw <- alpha*D_raw + sum_e r_t(e)/w(e)`. Its final
`L_raw=m_final+log(D_raw)` allows recovery of the unweighted selected conditional:

`u(e)=mean_heads,probes[BF16(r_t(e))*exp(m_t-L_raw)/w(e)]`.

The exponential is per head/probe/tile,not per edge. The subsequent selected-mass
scaling and block-mass normalization are unchanged. Head averaging happens
before cross-layer transfer. No selected QK is recomputed to produce history.
Original special-Query Dense corrections still run,as in the matched fast14.

This is mathematical equivalence with a different rounding path,not bitwise
identity of all historical probabilities: BF16 AV tiles replace FP16 logit
encoding. Numeric tests cover the difference;full geometry is rerun once after
freezing. Formal AV uses the same BF16 tiles it did before.

## Layout,liveness and scheduling

Ordinary full Query groups use masked probe-row stores in the original MMA
layout. Only the short tail needs gather bookkeeping for repeated clamped probes.
Delay V loads until softmax/export preparation is ready,and perform the inverse-
weight reduction after AV,so the FP32 `p*inverse_weight` matrix need not coexist
with p still needed by AV. Independent no-GT microbenchmarks choose64-query,
128-key,4-warps,2-stage execution. Other layouts/precision/register caps/probe
splits were slower and are archived as diagnostics,not selected geometry runs.

Fuse current Key block means,8-anchor collection and head-sketch4 projection.
For VGGT16 heads the fixed projection coefficients are ±0.125,exact in BF16.
FP32 addition order differs slightly (tests ≤1e-6);the Value feature path is
unchanged. Compute the coarse observation for all groups once per layer,keep
its head-mean tensors directly as cache,and use views in the selector windows.

Increase the processing window from64 to128 independent Query groups. Each mask
is still shared by256 Query tokens. This changes batching and scratch memory,
not the support model. The nominal layer retention remains10/18/10%,average14%
(86% sparsity). Actual unique-edge density includes the original special-Query
correction and is reported per layer.

## Verification and timing

Unit tests: old attention/probability regressions plus probability-tile export,
posterior update and fused Key preparation (25 tests in total). Independent RGB
TUM128 validation compares same-QKV fast4 teacher probabilities and local AV.
Only no-GT numerical intermediates and timing choose implementations. Code and
configuration freeze precedes target quality;no target-score budget tuning.

The user explicitly authorized fast14 **timing only**. Each target scene runs
new-method quality and fast14 warmed complete-forward timing on the same GPU,
sequentially,with1 warmup and3 repeats. Unchanged fast14 geometry,SAF3R and Dense
results are reused. No paired Dense timing is authorized,so Dense speedup is N/A.
All diagnostic/timing-only records retain six quality fields as N/A with reasons.

Use `scripts/run_history_speed_evaluation.py --freeze <selection.json> --output
<fresh-output>` for the frozen18-scene evaluation plus authorized timing. Use
`scripts/report_history_evidence_speed.py --output <output>` to audit/report.
The CPU-only `run_profiler_gpu_exclusive.py` wrapper enforces GPU4/5/6 leases and
records competing CUDA applications. Full-forward timing includes all auxiliary
computation. The separate profile uses mutually exclusive stage events;the AV
row includes inline history export and is never counted again as a separate row.
