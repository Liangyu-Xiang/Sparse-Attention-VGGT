# Common VGGT path audit against SAF3R

User request:check whether the two methods have identical non-policy,
non-Attention work,and explain any additional differences. Base c85543db;
independent branch codex/saf3r-common-path-audit-20261104. No production forward,
mask,retention,quality gate or checkpoint was changed. New code only creates
runtime diagnostic scopes and a report. Raw events and instrumented forward
texts are in outputs/common_path_audit_20261104.

## Main evidence

Same heads128/checkpoint/preprocessing/GPU7,oneCUDA program per physical GPU.
Ours is the256-Query history configuration. Three uninstrumented warmed full
forwards precede a separate instrumented forward. Both models retain four DPT
intermediates and execute camera,depth and point heads.488 common module input/
output shape,dtype and stride comparisons pass;common module call counts match.

Common model operators including Block residuals:ours4.558671s,SAF3R4.551536s,
only7.135ms(.1568%)apart in the single profiled forward. This is practical
agreement,not evidence of mathematically exact runtime equality or a significant
small speed difference. Backbone,frame blocks,global FFN,QKV,norm/RoPE,projection
and prediction heads all use the same mathematical modules and dimensions.

Disjoint same-profile accounting:

| Bucket | Ours256 s | SAF3R s |
|---|---:|---:|
| Common model operators |4.558671|4.551536|
| Ordinary aggregator allocator gaps |.369718|.440114|
| Other casts/assembly/checks/dispatch |.206640|.142124|
| Subtotal excluding main policy/Attention paths |5.135029|5.133774|
| Ours policy construction/history/state/budget/transfer |5.948889|included below|
| Attention and its execution/route pipeline |2.999967|1.577925|
| Total profile GPU timeline |14.083885|6.711699|

The SAF3R1.577925s includes its TopK and other per-pattern preparation/gathers;
it is not pure AV time. Both entire Attention/route pipelines were excluded from
the shared-path comparison. No equal-retained-edge kernel comparison is claimed.

## Actual non-identical engineering work

Ours has .033805s of QKV preparation around the common module calls,versus
.000052s in SAF3R. Both norm/RoPE outputs are FP32. Ours then casts Q/K to BF16
and materializes strided BF16 V into contiguous Head-major storage. SAF3R retains
FP32 Q/K and interleaved BF16 V at this wrapper boundary;its downstream SDPA
under autocast and custom kernels may perform their own conversions. Do not infer
that it never casts,or that this is a controlled precision/quality ablation.

Ours copies1584 query-window outputs then reshapes: .024580s. SAF3R concatenates
16 per-Head results24times then reshapes: .017558s. The output dimensions match;
copy/dispatch organization differs. Our extra post-Attention validation/history
bookkeeping/statistics take .039276s of GPU timeline. Summed CPU latency .355s
includes waiting for earlier GPU work and cannot be added to device timings.

Both aggregators call empty_cache24times. Different live/cached allocations
explain why identical allocator calls need not cost exactly the same;reserved
bytes and per-call times are recorded. Ours additionally creates4 selector CUDA
Graphs in this profiled forward;their .034184s allocator interval is already
nested in policy construction. Do not count it again as common work.

The nearly equal5.135/5.134 totals include offsetting engineering differences.
The previous5.30s residual was neither an additional ours-only cost nor a proven
irreducible common-model constant. This audit replaces that ambiguous accounting.

## Protocol and scope

Fresh-routing warmed medians:ours13.839723s,SAF3R6.476984s. These belong to this
batch,not the earlier14.366136/6.505094s runs. All fresh SAF3R forwards compute
TopK once;the separately labeled cached-route single forward computes it zero
times and is not a complete independent-inference baseline.

Historical6.505094s provenance was checked:method saf3r_lazy4,source hash
7bab97c9e6170310dac033cc2f2de2e164db7f5a294e357becf10301118388e8
matches scripts/evaluate_saf3r_matched.py,which resets routes each iteration.
The initial concern about generic runners resetting only outside their loop does
not apply to this historical matched record. The risk remains in those separate
generic scripts,which were not changed by this audit.

All six geometric scores are N/A for the new diagnostic runs,with explicit
runtime-only reasons. Existing complete six-metric quality results remain
historical controls;no new quality campaign or Dense rerun was performed.
SAF3R is the local lazy first-correspondence-head Top4,ETH3D profile,not paper
DINO Top-K. Active Head/layer counts:stride212,all_to_first85,q_probe_topk75,
dino_topk7,global_to_frame5. No broadcast_first is active.

Reproduce each scripts/audit_common_path.py method under run_gpu_exclusive.py,
then python -m scripts.report_common_path_audit. The complete source snapshot,
488 shape checks,event trees,per-scope CUDA/CPU timing and both exclusive GPU
monitors accompany the report. Profile figures are24-layer totals,separate from
uninstrumented forward medians;CPU/GPU overlap must not be double counted.
