# VGGT common-path audit:ours256 versus SAF3R lazy Top4

The main hypothesis is supported:shared mathematical model work has the same operators,shapes,dtypes,strides and call counts,with practically identical measured time. This does not mean every engineering operation is identical or every timing sample must be exactly equal.

Same heads128 inputs/checkpoint/preprocessing/GPU7 RTX5090 D,oneCUDA program per physical GPU. No algorithm,mask,retention or checkpoint edits. One cold+three warmed uninstrumented full forwards per method,then one separately instrumented forward. Common shape/type/stride checks exclude pointer addresses. Both save_intermediates=False and both execute camera,depth and point heads;no extra model head or FFN exists only in ours.

## Exact common-module comparison

| operation | calls | ours_seconds | saf3r_seconds | difference_seconds |
| --- | --- | --- | --- | --- |
| image_backbone | 1 | 0.599961 | 0.600678 | -0.000718 |
| frame_block | 24 | 1.032085 | 1.055890 | -0.023805 |
| global_qkv | 24 | 0.099592 | 0.099359 | 0.000232 |
| global_q_norm | 24 | 0.078778 | 0.078680 | 0.000098 |
| global_k_norm | 24 | 0.078738 | 0.078674 | 0.000065 |
| global_rope | 48 | 0.231170 | 0.231292 | -0.000122 |
| global_proj | 24 | 0.035793 | 0.029756 | 0.006037 |
| global_proj_drop | 24 | 0.000026 | 0.000009 | 0.000017 |
| global_ffn | 24 | 0.285405 | 0.274643 | 0.010763 |
| global_norm1 | 24 | 0.017220 | 0.017247 | -0.000027 |
| global_norm2 | 24 | 0.017234 | 0.017260 | -0.000026 |
| camera_head | 1 | 0.008933 | 0.009362 | -0.000429 |
| depth_head | 1 | 0.994244 | 0.987489 | 0.006755 |
| point_head | 1 | 1.003959 | 0.995913 | 0.008046 |
| global_block_residual_and_layerscale | 24 | 0.075534 | 0.075285 | 0.000249 |

Common totals including Block residual/layerscale:ours=4.558671051s,SAF3R=4.551535951s,difference=0.007135101s(0.1568%). These are single-profile CUDA-event spans,not repeated-sample statistical evidence of a small speed difference.

## Disjoint accounting and exclusions

| method | profile_gpu_span_seconds | excluded_policy_seconds | excluded_attention_and_routing_seconds | common_model_seconds | common_allocator_gap_seconds | other_pipeline_seconds | remaining_after_exclusions_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours256 | 14.083885 | 5.948889 | 2.999967 | 4.558671 | 0.369718 | 0.206640 | 5.135029 |
| saf3r | 6.711699 | 0.000000 | 1.577925 | 4.551536 | 0.440114 | 0.142124 | 5.133774 |

Ours policy bucket includes clustering,pooling,Grid/history production,history consumption,state loss,budget,packing and transfer;it already includes nested history_prior and state_loss_budget_pack_transfer. SAF3R attention/routing bucket includes all384 head paths,TopK selection,per-pattern gathers/conversions and SDPA/Triton. Thus this comparison excludes the COMPLETE attention/routing pipelines rather than pretending equal retained-edge arithmetic or isolating every per-mode gather. It establishes common-path equivalence;it is not an equal-sparsity kernel-efficiency benchmark.

After these explicit exclusions,both remaining totals are about5.14s. Their close totals contain offsetting engineering differences;they do not prove each substep identical. The previous5.30s residual was never a pure irreducible shared-model constant.

## Non-identical engineering work

| operation | ours_seconds | saf3r_seconds | difference_seconds |
| --- | --- | --- | --- |
| qkv_prepare_views_casts_and_contiguous_extra | 0.033805 | 0.000052 | 0.033754 |
| output_assembly_and_layout | 0.024580 | 0.017558 | 0.007022 |
| post_attention_checks_and_stats | 0.039276 | 0.000000 | 0.039276 |
| other_wrapper_and_model_dispatch | 0.108979 | 0.124515 | -0.015536 |

QKV projections,norms and RoPE are common. At wrapper exit,ours converts FP32 Q/K to BF16 and materializes strided V as contiguous[H,N,D];SAF3R keeps FP32 Q/K and strided BF16 V at that boundary. Its downstream SDPA/autocast and custom kernels can perform their own conversions;do not claim it has no casts or infer a quality effect from this runtime-only audit. The0.034s is the extra preparation interval around common module calls,not all attention-pipeline conversion costs.

Ours writes1584 Query-window outputs into a full buffer,and then changes Head-major to token-major layout. SAF3R concatenates16 per-Head outputs24times,and performs the same final layout change. Both assemble the same output dimensions,but use different dispatch/copy strategies. Our post-Attention validation and statistics add a24-layer0.039s device-timeline interval;their0.355s summed CPU latency includes waiting for earlier GPU work and MUST NOT be added to GPU spans.

Both aggregators call empty_cache24times with the same source. Runtime allocator state differs;observed ordinary allocator gaps are listed above. Ours also captures4 new selector CUDA Graphs in this profiled forward,which call empty_cache;their time is nested inside the policy bucket and is not added again. Individual timings depend on cache/allocator context and normal measurement variation despite identical arithmetic.

## Attention patterns,kept separate from common work

| mode | head_calls | pipeline_seconds |
| --- | --- | --- |
| all_to_first | 85 | 0.019710 |
| stride | 212 | 0.981346 |
| global_to_frame | 5 | 0.001278 |
| q_probe_topk | 75 | 0.428507 |
| dino_topk | 7 | 0.147083 |

Active profile:212 stride heads,85 all_to_first,75 q_probe_topk,7 dino_topk and5 global_to_frame,384 Head/layer instances total. No broadcast_first head is active,so that optional attention-map-sharing mode does not explain this comparison. Regular slicing/SDPA and irregular shared-support gathers are different execution strategies;remaining Attention timing differences cannot be attributed to retained fraction alone without equal-support experiments.

## Fresh routing and historical protocol check

Fresh full-forward medians:ours256=13.839722960s,SAF3R=6.476984450s. Both include their current policy/routing work. SAF3R fused TopK executes once in EVERY cold/warmed/profiled forward. TopK scope in the profile=0.111933891s. The separate ONE-forward cached-routing diagnostic has zero TopK calls;it is not a complete independent-inference timing baseline.

Historical6.505093521s SAF3R was traced by its recorded source hash to evaluate_saf3r_matched.py,which resets routing inside every forward loop. Its stored driver hash matches the archived source. The initially suspected stale-route issue belongs to other generic runners which reset only outside their loops;it does NOT invalidate the historical6.51s matched run. These historical times still belong to a different batch.

## Scope,validation and reproduction

| AUC@3 % ↑ | AUC@30 % ↑ | ACC m ↓ | Comp m ↓ | CD m ↓ | F1 % ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| N/A | N/A | N/A | N/A | N/A | N/A |

CD = (ACC + Comp) / 2, using unsquared distances.

Speedup vs VGGT: N/A until matched Dense full-forward timing is available; see comparison report.

Unavailable: auc03: Runtime/code audit only;new diagnostic runs do not evaluate geometry;complete historical quality is unchanged and reused; auc30: Runtime/code audit only;new diagnostic runs do not evaluate geometry;complete historical quality is unchanged and reused; acc: Runtime/code audit only;new diagnostic runs do not evaluate geometry;complete historical quality is unchanged and reused; comp: Runtime/code audit only;new diagnostic runs do not evaluate geometry;complete historical quality is unchanged and reused; cd: Runtime/code audit only;new diagnostic runs do not evaluate geometry;complete historical quality is unchanged and reused; f1: Runtime/code audit only;new diagnostic runs do not evaluate geometry;complete historical quality is unchanged and reused


No fresh quality campaign or Dense rerun was requested/needed for this code/runtime inspection. All six fields are N/A for these diagnostic runs;complete historical quality is reused separately. No default source forward was edited:ours diagnostic forward is dynamically generated from the original with scope wrappers;SAF3R diagnostic forward preserves arithmetic with explicit scope wrappers. Uninstrumented medians were measured BEFORE installing module hooks/dynamic forwards/Tensor scalar instrumentation. Instrumented forward text,raw event trees,CPU latencies,tensor layouts,source snapshot and GPU monitors are retained. CPU waits overlap preceding GPU work;do not sum CPU/GPU measurements.

Independent checks:equal input/hardware/model flags;488 common Tensor metadata comparisons;equal module call counts;two GPU7 exclusive monitors;historical matched driver hash;one fresh TopK call per forward. Reproduce the two audit_common_path.py invocations under run_gpu_exclusive.py,then python -m scripts.report_common_path_audit.
