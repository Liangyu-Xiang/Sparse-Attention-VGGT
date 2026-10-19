# Fused full-coverage Value / evolving Attention

[Implementation, semantics and limitations](../../docs/experiments/fused_shared_av.md)

- `kernel_gate.json`: initial fixed-support kernel,2/50/128frames.
- `kernel_gate_pipelined.json`: pipelined kernel,128frames,10/15/20%budgets.
- `chess2_final_gate`: small-input all24-layer structural capture and timings.
- `chess50_final_gate`: current50-frame paired gate.
- `chess128_final_gate`: current128-frame paired gate, source snapshots match working implementation.
- `recheck_20261019`: paired300-frame Dense/fused, SAF3R Top-128 OOM, explicit official Top-4 comparison, and128-frame memory recheck. See [report](recheck_20261019/report.md).
- Other `first`, `v1`, `v2`, `gate`, `window128` paths are local development revisions, not all versioned; do not mix their timings/metrics.

Current128-frame manual20%scheme:0Value-only;1–9/17–23Attention20%;10–16Attention80%.
Dense/fused warmed median full forward16.6402/13.9401s; peak14.3586/14.9271GiB;
Chess AUC30 88.4662%/87.3659%. **NOT accepted**:2x speed, memory and scene accuracy guardrails not met.
These are operator/one-scene gates, NOT a completed seven-scene pose/depth/reconstruction evaluation.
Calibration, full evaluation and adaptive-budget experiments are intentionally locked by failed gates.

300-frame Chess: Dense70.530s/25.090GiB/AUC30 88.3168%; fused62.714s/26.532GiB/87.6535%;
SAF3R official Top-4:25.033s/26.480GiB/88.3653%. Top-128 is OOM, not silently replaced.
Releasing obsolete Top-128 selection indices before fresh selection corrects the128-frame
peak from29.882GiB to22.453GiB, with identical pose fingerprints. No warmup-selection reuse.
