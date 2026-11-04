"""Compare identical model work separately from policy and attention pipelines."""
import csv,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/common_path_audit_20261104'
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def write_csv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in ['storage_ptr','data_ptr']}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def table(rows,keys):
 def cell(v):return 'N/A' if v is None else f'{v:.6f}' if isinstance(v,float) else str(v)
 return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(cell(r.get(k)) for k in keys)+' |' for r in rows])
def main():
 data={k:json.loads((OUT/k/'result.json').read_text()) for k in ['ours256','saf3r']};a=data['ours256'];b=data['saf3r'];assert a['complete'] and b['complete'];assert a['input']==b['input'] and a['hardware']==b['hardware'] and a['model_flags']==b['model_flags']
 groups={k:{r['name']:r for r in d['profile']['groups']} for k,d in data.items()};g=groups
 shared=['image_backbone','frame_block','global_qkv','global_q_norm','global_k_norm','global_rope','global_proj','global_proj_drop','global_ffn','global_norm1','global_norm2','camera_head','depth_head','point_head']
 common=[];checks=[]
 for name in shared:
  x=g['ours256'][name];y=g['saf3r'][name];assert x['calls']==y['calls']
  common.append(dict(operation=name,calls=x['calls'],ours_seconds=x['inclusive_seconds'],saf3r_seconds=y['inclusive_seconds'],difference_seconds=x['inclusive_seconds']-y['inclusive_seconds']))
  for label in a['notes']:
   if label.startswith(name+'.') or label.startswith(name+'.L'):
    if label in b['notes']:
     equal=clean(a['notes'][label])==clean(b['notes'][label]);checks.append(dict(label=label,shape_dtype_stride_equal=equal));assert equal,label
 common.append(dict(operation='global_block_residual_and_layerscale',calls=24,ours_seconds=a['profile']['common_global_residual_self_seconds'],saf3r_seconds=b['profile']['common_global_residual_self_seconds'],difference_seconds=a['profile']['common_global_residual_self_seconds']-b['profile']['common_global_residual_self_seconds']))
 totals=[];memory=[]
 alg_names=['ours.value_update','ours.pooling','ours.m5c_coarse_observation','ours.sampled_probe_grid_summary','ours.m5c_evidence_and_host_copy','ours.history_summary','ours.coarse_observation','ours.residual_fine_probe','ours.reuse_and_budget']
 for method,d in data.items():
  rows=d['profile']['events'];gg=g[method];root=rows[0]['seconds'];algorithm=sum(gg[name]['inclusive_seconds'] for name in alg_names) if method=='ours256' else 0.
  attention=gg['ours.execution_and_evidence' if method=='ours256' else 'saf3r.attention_and_routing']['inclusive_seconds'];commons=d['profile']['common_module_inclusive_seconds']+d['profile']['common_global_residual_self_seconds']
  allocator=0.;extra_cache=0.;empty=[r for r in rows if r['name']=='shared.empty_cache']
  for event,note in zip(empty,d['notes']['empty_cache_calls']):
   parent=rows[event['parent']]['name'];is_common=parent=='model'
   if is_common:allocator+=event['seconds']
   else:extra_cache+=event['seconds']
   memory.append(dict(method=method,parent=parent,common_aggregator_call=is_common,gpu_timeline_seconds=event['seconds'],cpu_seconds=note['cpu_seconds'],released_reserved_bytes=note['reserved_before']-note['reserved_after']))
  remaining=root-algorithm-attention
  totals.append(dict(method=method,warmed_median_seconds=d['timing']['median_seconds'],profile_wall_seconds=d['profile']['wall_seconds'],profile_gpu_span_seconds=root,common_model_seconds=commons,excluded_policy_seconds=algorithm,excluded_attention_and_routing_seconds=attention,common_allocator_gap_seconds=allocator,other_pipeline_seconds=remaining-commons-allocator,remaining_after_exclusions_seconds=remaining,extra_graph_allocator_seconds_nested_in_policy=extra_cache,dense_speedup=None))
 common_total=sum(r['ours_seconds'] for r in common);saf_total=sum(r['saf3r_seconds'] for r in common)
 extra=[]
 for label,our,saf in [
  ('qkv_prepare_views_casts_and_contiguous_extra',g['ours256']['ours.qkv_prepare_total']['self_seconds'],g['saf3r']['saf3r.qkv_prepare_total']['self_seconds']),
  ('output_assembly_and_layout',g['ours256']['ours.window_output_copy']['inclusive_seconds']+g['ours256']['ours.output_layout']['inclusive_seconds'],g['saf3r']['saf3r.output_head_concat']['inclusive_seconds']+g['saf3r']['saf3r.output_layout']['inclusive_seconds']),
  ('post_attention_checks_and_stats',g['ours256']['ours.post_attention_checks_stats']['inclusive_seconds'],0.),
  ('other_wrapper_and_model_dispatch',g['ours256']['global_wrapper']['self_seconds']+g['ours256']['model']['self_seconds'],g['saf3r']['global_wrapper']['self_seconds']+g['saf3r']['model']['self_seconds'])]:extra.append(dict(operation=label,ours_seconds=our,saf3r_seconds=saf,difference_seconds=our-saf))
 for i,t in enumerate(totals):assert abs(sum(r['ours_seconds' if i==0 else 'saf3r_seconds'] for r in extra)-t['other_pipeline_seconds'])<2e-5
 monitors=[json.loads((OUT/f'monitors/{method}.json').read_text()) for method in data]
 assert all(m['returncode']==0 and m['max_simultaneous_gpu_applications']==1 and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] for m in monitors);assert monitors[0]['gpu_uuid']==monitors[1]['gpu_uuid'] and monitors[0]['gpu']==7
 assert all(x['topk_calls'].get('compute_qk_topk_indices_fused',0)==1 for x in b['timing']['samples']);assert b['cached_routing_diagnostic']['topk_calls']['compute_qk_topk_indices_fused']==0
 historical=json.loads((ROOT/'outputs/sampled_history_quality_20261104/references/saf3r/7scenes/heads.json').read_text());driver='scripts/evaluate_saf3r_matched.py';assert historical['source_sha256'][driver]==hashlib.sha256((ROOT/driver).read_bytes()).hexdigest();assert historical['method']=='saf3r_lazy4'
 # Per-mode times are reconstructed in actual head order;they include each
 # pattern's preparation/gathers/SDPA or Triton work,not just mathematical QK.
 cfg=json.loads((ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json').read_text());head_map={}
 for layer,patterns in cfg.items():
  heads=['full']*16
  for pattern in patterns:
   for h in pattern['head_ids']:heads[h]=pattern['mode']
  head_map[int(layer)]=heads
 per_mode={};counter={}
 for row in b['profile']['events']:
  if row['name']!='saf3r.attention_and_routing':continue
  layer=row['layer'];h=counter.get(layer,0);counter[layer]=h+1;mode=head_map[layer][h];m=per_mode.setdefault(mode,dict(mode=mode,head_calls=0,pipeline_seconds=0.));m['head_calls']+=1;m['pipeline_seconds']+=row['seconds']
 assert len(counter)==24 and all(v==16 for v in counter.values())
 write_csv(OUT/'common_modules.csv',common);write_csv(OUT/'other_pipeline.csv',extra);write_csv(OUT/'totals.csv',totals);write_csv(OUT/'common_shape_checks.csv',checks);write_csv(OUT/'allocator_calls.csv',memory);write_csv(OUT/'saf3r_head_modes.csv',list(per_mode.values()))
 for method,d in data.items():write_csv(OUT/f'{method}_scopes.csv',d['profile']['groups']);write_csv(OUT/f'{method}_events.csv',d['profile']['events'])
 from scripts.quality_metrics import quality_record,quality_markdown
 quality=quality_record({},'Runtime/code audit only;new diagnostic runs do not evaluate geometry;complete historical quality is unchanged and reused')
 (OUT/'quality.md').write_text(quality_markdown(quality))
 summary=dict(complete=True,base_commit='c85543db',methods=['ours256','saf3r_lazy4'],input_matched=True,hardware_matched=True,model_flags_matched=True,shape_dtype_stride_checks=len(checks),shape_dtype_stride_all_equal=True,common=common,other_pipeline=extra,totals=totals,common_difference_seconds=common_total-saf_total,common_difference_percent=100*(common_total/saf_total-1),saf3r_head_modes=list(per_mode.values()),fresh_topk_one_per_forward_verified=True,historical_saf3r_driver=driver,historical_saf3r_driver_sha256=historical['source_sha256'][driver],historical_saf3r_reset_each_forward=True,quality=quality,dense_speedup=None,dense_reason='No new Dense run;this is a two-method diagnostic comparison')
 save(OUT/'summary.json',summary)
 sources=['scripts/audit_common_path.py','scripts/report_common_path_audit.py','scripts/evaluate_saf3r_matched.py','scripts/evaluate_reuse_exploration.py','saf3r/patch/reuse_residual_exploration.py','saf3r/patch/region_history_attention.py','saf3r/patch/reuse_sampled_history.py','saf3r/patch/patch_vggt.py','saf3r/patch/wrapper.py','saf3r/patch/sparse_attention.py','saf3r/triton/reuse_direct_sparse_attention.py','saf3r/triton/reuse_direct_sampled_probability.py','saf3r/utils/model_utils.py','saf3r/models/official_vggt/models/vggt.py','saf3r/models/official_vggt/models/aggregator.py','saf3r/models/official_vggt/layers/block.py','configs/evaluation/eval_saf3r_vggt.yaml','configs/evaluation/reuse_sampled_history_256.json','configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']
 hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources};save(OUT/'source_sha256.json',hashes)
 with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
  for name in sources:archive.add(ROOT/name,arcname=name,recursive=False)
 text=['# VGGT common-path audit:ours256 versus SAF3R lazy Top4','',
 'The main hypothesis is supported:shared mathematical model work has the same operators,shapes,dtypes,strides and call counts,with practically identical measured time. This does not mean every engineering operation is identical or every timing sample must be exactly equal.','',
 'Same heads128 inputs/checkpoint/preprocessing/GPU7 RTX5090 D,oneCUDA program per physical GPU. No algorithm,mask,retention or checkpoint edits. One cold+three warmed uninstrumented full forwards per method,then one separately instrumented forward. Common shape/type/stride checks exclude pointer addresses. Both save_intermediates=False and both execute camera,depth and point heads;no extra model head or FFN exists only in ours.','',
 '## Exact common-module comparison','',table(common,['operation','calls','ours_seconds','saf3r_seconds','difference_seconds']),'',f'Common totals including Block residual/layerscale:ours={common_total:.9f}s,SAF3R={saf_total:.9f}s,difference={common_total-saf_total:.9f}s({100*(common_total/saf_total-1):.4f}%). These are single-profile CUDA-event spans,not repeated-sample statistical evidence of a small speed difference.','',
 '## Disjoint accounting and exclusions','',table(totals,['method','profile_gpu_span_seconds','excluded_policy_seconds','excluded_attention_and_routing_seconds','common_model_seconds','common_allocator_gap_seconds','other_pipeline_seconds','remaining_after_exclusions_seconds']),'',
 'Ours policy bucket includes clustering,pooling,Grid/history production,history consumption,state loss,budget,packing and transfer;it already includes nested history_prior and state_loss_budget_pack_transfer. SAF3R attention/routing bucket includes all384 head paths,TopK selection,per-pattern gathers/conversions and SDPA/Triton. Thus this comparison excludes the COMPLETE attention/routing pipelines rather than pretending equal retained-edge arithmetic or isolating every per-mode gather. It establishes common-path equivalence;it is not an equal-sparsity kernel-efficiency benchmark.','',
 'After these explicit exclusions,both remaining totals are about5.14s. Their close totals contain offsetting engineering differences;they do not prove each substep identical. The previous5.30s residual was never a pure irreducible shared-model constant.','',
 '## Non-identical engineering work','',table(extra,['operation','ours_seconds','saf3r_seconds','difference_seconds']),'',
 'QKV projections,norms and RoPE are common. At wrapper exit,ours converts FP32 Q/K to BF16 and materializes strided V as contiguous[H,N,D];SAF3R keeps FP32 Q/K and strided BF16 V at that boundary. Its downstream SDPA/autocast and custom kernels can perform their own conversions;do not claim it has no casts or infer a quality effect from this runtime-only audit. The0.034s is the extra preparation interval around common module calls,not all attention-pipeline conversion costs.','',
 'Ours writes1584 Query-window outputs into a full buffer,and then changes Head-major to token-major layout. SAF3R concatenates16 per-Head outputs24times,and performs the same final layout change. Both assemble the same output dimensions,but use different dispatch/copy strategies. Our post-Attention validation and statistics add a24-layer0.039s device-timeline interval;their0.355s summed CPU latency includes waiting for earlier GPU work and MUST NOT be added to GPU spans.','',
 'Both aggregators call empty_cache24times with the same source. Runtime allocator state differs;observed ordinary allocator gaps are listed above. Ours also captures4 new selector CUDA Graphs in this profiled forward,which call empty_cache;their time is nested inside the policy bucket and is not added again. Individual timings depend on cache/allocator context and normal measurement variation despite identical arithmetic.','',
 '## Attention patterns,kept separate from common work','',table(list(per_mode.values()),['mode','head_calls','pipeline_seconds']),'',
 'Active profile:212 stride heads,85 all_to_first,75 q_probe_topk,7 dino_topk and5 global_to_frame,384 Head/layer instances total. No broadcast_first head is active,so that optional attention-map-sharing mode does not explain this comparison. Regular slicing/SDPA and irregular shared-support gathers are different execution strategies;remaining Attention timing differences cannot be attributed to retained fraction alone without equal-support experiments.','',
 '## Fresh routing and historical protocol check','',
 f"Fresh full-forward medians:ours256={a['timing']['median_seconds']:.9f}s,SAF3R={b['timing']['median_seconds']:.9f}s. Both include their current policy/routing work. SAF3R fused TopK executes once in EVERY cold/warmed/profiled forward. TopK scope in the profile={g['saf3r']['saf3r.mask_topk']['inclusive_seconds']:.9f}s. The separate ONE-forward cached-routing diagnostic has zero TopK calls;it is not a complete independent-inference timing baseline.",'',
 'Historical6.505093521s SAF3R was traced by its recorded source hash to evaluate_saf3r_matched.py,which resets routing inside every forward loop. Its stored driver hash matches the archived source. The initially suspected stale-route issue belongs to other generic runners which reset only outside their loops;it does NOT invalidate the historical6.51s matched run. These historical times still belong to a different batch.','',
 '## Scope,validation and reproduction','',quality_markdown(quality),'',
 'No fresh quality campaign or Dense rerun was requested/needed for this code/runtime inspection. All six fields are N/A for these diagnostic runs;complete historical quality is reused separately. No default source forward was edited:ours diagnostic forward is dynamically generated from the original with scope wrappers;SAF3R diagnostic forward preserves arithmetic with explicit scope wrappers. Uninstrumented medians were measured BEFORE installing module hooks/dynamic forwards/Tensor scalar instrumentation. Instrumented forward text,raw event trees,CPU latencies,tensor layouts,source snapshot and GPU monitors are retained. CPU waits overlap preceding GPU work;do not sum CPU/GPU measurements.','',
 f"Independent checks:equal input/hardware/model flags;{len(checks)} common Tensor metadata comparisons;equal module call counts;two GPU7 exclusive monitors;historical matched driver hash;one fresh TopK call per forward. Reproduce the two audit_common_path.py invocations under run_gpu_exclusive.py,then python -m scripts.report_common_path_audit."]
 (OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(common_ours=common_total,common_saf3r=saf_total,checks=len(checks),totals=totals,other=extra),indent=2))
if __name__=='__main__':main()
