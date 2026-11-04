# Direct Sparse selected-Key probability export

Final matched actual-data kernel benchmark:all1584 windows across24 global layers,heads128,all16Heads,same Q/K/V/support/bias,three samples per variant/window. Sums of window medians,NOT a complete forward or single-layer time. Alternate outputs never feed the original current Direct-probability trajectory. GPU7 exclusive,no Dense/SAF3R model reruns.

| Output | Query interval | 24-layer total s | Extra cost |
|---|---:|---:|---:|
| av_only_seconds | - | 2.864744289 | +0.000% |
| probe_probability_seconds | 32 | 3.710718651 | +29.531% |
| branch_inline_1_seconds | 32 | 3.689132036 | +28.777% |
| branch_inline_8_seconds | 256 | 2.947034430 | +2.873% |
| branch_inline_16_seconds | 512 | 2.929577986 | +2.263% |

The new branch path exports fixed-interval Query probabilities on every selected Key. Normalize each Head using its final compensated sparse LSE,then average Heads. Specials are retained in the denominator. AV covers all Queries. Final-probability reduction and all export launches are included in these timings.

Main QK is computed once per Query Group. Sampled Groups export FP16 online exponents and FP32 Tile reference maxima before AV;ordinary Groups use the usual AV body in the same launch. The small finalizer rescales with final LSE and averages Heads. There is no auxiliary probability QK,no Grid routing,no full QxK mask,and no FP32 raw-logit bank. Dense four-Probe output remains materially more expensive;the near-free claim only applies to the measured sparse Query sampling.

Scratch for one64-Group16-Head window:every256Queries6929152bytes(~6.61MiB),every512Queries3464576bytes(~3.30MiB). No current history sampling/default changed.

## Numerical validation

Numerical cases=160,fresh CUDA Graph cases=6; FP16/BF16,pure/compensated,zero weights,bias60,empty patch support with Specials,tail/duplicate Probes,one/four Probes,strides1/2/4/8/16 and non-contiguous Head windows. Real-data maximum errors: {"av_only_av_relative_l2": 0.0, "probe_probability_av_relative_l2": 0.0, "branch_inline_1_av_relative_l2": 0.0, "branch_inline_8_av_relative_l2": 0.0, "branch_inline_16_av_relative_l2": 0.0, "fast_probability_max_abs": 1.1533498764038086e-05, "fast_normalization_max_abs": 2.086162567138672e-05, "branch_stride_1_probability_max_abs": 1.1533498764038086e-05, "branch_stride_1_normalization_max_abs": 2.086162567138672e-05, "branch_stride_8_probability_max_abs": 7.927417755126953e-06, "branch_stride_8_normalization_max_abs": 2.092123031616211e-05, "branch_stride_16_probability_max_abs": 6.616115570068359e-06, "branch_stride_16_normalization_max_abs": 2.086162567138672e-05, "masked_probability_max_abs": 1.1533498764038086e-05}.

FP16 export storage is an explicit auxiliary probability quantization;do not claim bitwise probability equality. AV numerical equality in the actual all-window measurements is recorded as zero relative L2 for every window;this is not a bytewise dump comparison. Default debiased_log=False returns actual compensated sparse P;debiased_log=True is a separate log-score export with different numeric errors and is not the timed near-free output.

## Geometry and end-to-end limits

| AUC@3 % ↑ | AUC@30 % ↑ | ACC m ↓ | Comp m ↓ | CD m ↓ | F1 % ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| N/A | N/A | N/A | N/A | N/A | N/A |

CD = (ACC + Comp) / 2, using unsquared distances.

Speedup vs VGGT: N/A until matched Dense full-forward timing is available; see comparison report.

Unavailable: auc03: Kernel-only numerical/performance extension; changed Query sampling is not fed into model history; no pose/reconstruction evaluation; auc30: Kernel-only numerical/performance extension; changed Query sampling is not fed into model history; no pose/reconstruction evaluation; acc: Kernel-only numerical/performance extension; changed Query sampling is not fed into model history; no pose/reconstruction evaluation; comp: Kernel-only numerical/performance extension; changed Query sampling is not fed into model history; no pose/reconstruction evaluation; cd: Kernel-only numerical/performance extension; changed Query sampling is not fed into model history; no pose/reconstruction evaluation; f1: Kernel-only numerical/performance extension; changed Query sampling is not fed into model history; no pose/reconstruction evaluation


No new full-model forward median is measured in this kernel-only extension. The preceding model15.641640s belongs to direct_probability_quality_20261104 and is not a new timing or geometry result. New Query sampling has not been fed into history,so existing six-metric quality cannot validate such a modeling change. Dense end-to-end speedup N/A,not an operator ratio.

## Earlier attempts

Original Tile-normalized inline,Gather exponent reuse,masked stores and splitting AV into two launches are retained with their raw benchmarks. The first attempted one-Probe pointer store failed compilation and was fixed by explicit pointer broadcasting. One branch benchmark failed because instrumentation wrappers were imported as a device function;import order was corrected without changing kernel arithmetic. Failed runs preserve all six scores N/A and their logs. Synthetic preflight and first-window24-layer timings are separate from final full1584-window evidence.

Compiler metadata: {"av_only": {"_attention_mode0": {"registers": 128, "spills": 0, "shared_bytes": 45568}}, "probe_probability": {"_attention_mode0": {"registers": 128, "spills": 0, "shared_bytes": 45568}, "_probe": {"registers": 118, "spills": 0, "shared_bytes": 18432}}, "branch_inline_1": {"registers": 191, "spills": 0, "shared_bytes": 45568}, "branch_inline_8": {"registers": 192, "spills": 0, "shared_bytes": 45568}, "branch_inline_16": {"registers": 192, "spills": 0, "shared_bytes": 45568}}. This is not measured occupancy. Verified exclusive monitors: 13;kernel source hashes unchanged. All raw run six-metric statuses are in six_metrics_unavailable.csv;no GT parameter tuning or quality-gate changes.
