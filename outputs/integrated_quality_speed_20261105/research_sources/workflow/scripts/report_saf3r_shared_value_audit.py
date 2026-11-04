"""Reconcile support construction vs selected-support execution on the same input."""
from pathlib import Path
import json
import math
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/saf3r_shared_value_audit_20261105'
SCENES=('heads','chess','stairs')
METHODS=('saf3r','shared_value_q128')

def mean(values):
    values=list(values);return sum(values)/len(values)

def decompose(d):
    ps=d['event_profiles']
    names=set().union(*(p['intervals'] for p in ps))
    ev={n:mean(p['intervals'].get(n,{}).get('cuda_interval_seconds',0) for p in ps) for n in names}
    get=lambda n:ev.get(n,0.)
    total=mean(p['cuda_interval_seconds'] for p in ps)
    projection=get('projection/qkv')+get('projection/output')
    heads=sum(get('model/'+n) for n in ('camera_head','depth_head','point_head'))
    attention=get('global/attention_wrapper')
    if d['method']=='saf3r':
        selection=sum(v for n,v in ev.items() if n.startswith('sf_select/'))
        pure_av=sum(v for n,v in ev.items() if n.startswith('sf_av/'))
        pack=sum(v for n,v in ev.items() if n.startswith('sf_pack/'))
        dispatch=sum(v for n,v in ev.items() if n.startswith('sf_dispatch/'))
        dispatch_select=get('sf_select/q_probe')+get('sf_select/stride_indices')
        support_execution=dispatch-dispatch_select
        components=dict(support_construction=selection,selected_support_execution=support_execution,
            history_summary_completion=0.,qkv_output_projection=projection)
        components['attention_other']=attention-sum(components.values())
        details=dict(pure_qk_softmax_av=pure_av,explicit_kv_gather=pack,
            dispatch_other=support_execution-pure_av-pack,
            lazy_topk=get('sf_select/lazy_topk'),query_probe_selection=get('sf_select/q_probe'),
            stride_selection=get('sf_select/stride_indices'),history_probability_export=0.)
        for p in ps:assert p['intervals']['sf_select/lazy_topk']['calls']==1
        pairs=mean(sum(v['qk_pairs'] for v in p['workload'].values()) for p in ps)
    else:
        support_names=('value_update','pooling','m5c_coarse_observation','coarse_observation','residual_fine_probe','reuse_and_budget','cluster_seed_prepare')
        selection=sum(get('shared_stage/'+n) for n in support_names)
        execution=get('shared_stage/execution_and_evidence')
        history=sum(get('shared_stage/'+n) for n in ('sampled_probe_grid_summary','m5c_evidence_and_host_copy','history_summary'))
        components=dict(support_construction=selection,selected_support_execution=execution,
            history_summary_completion=history,qkv_output_projection=projection)
        components['attention_other']=attention-sum(components.values())
        details=dict(pure_qk_softmax_av=get('shared_execution/sampled_direct_av'),
            history_probability_export=get('shared_execution/sampled_probability_export'),
            value_clustering=get('shared_stage/value_update'),pooling_geometry=get('shared_stage/pooling'),
            coarse_observation=get('shared_stage/m5c_coarse_observation')+get('shared_stage/coarse_observation'),
            residual_candidates=get('shared_stage/residual_fine_probe'),prior_loss_budget_pack=get('shared_stage/reuse_and_budget'),
            seed=get('shared_stage/cluster_seed_prepare'))
        details['execution_call_other']=execution-details['pure_qk_softmax_av']-details['history_probability_export']
        pairs=mean(p['workload']['final_qk_pairs'] for p in ps)
    common=dict(image_encoder=get('model/image_encoder'),frame_blocks=get('model/frame_blocks'),
        global_mlp=get('global/mlp'),global_block_other=get('model/global_blocks')-attention-get('global/mlp'),
        output_heads=heads)
    common['outside_modules']=total-attention-sum(common.values())
    assert abs(sum(components.values())-attention)<1e-7
    assert abs(sum(components.values())+sum(common.values())-total)<1e-7
    frames=d['input']['shape'][0];h,w=d['input']['shape'][-2:]
    tokens=frames*((h//14)*(w//14)+5)
    return dict(profile_cuda_interval_seconds=total,profile_wall_seconds=mean(p['wall_seconds'] for p in ps),
        attention_components=components,details=details,common=common,all_scopes=ev,
        final_qk_pairs=pairs,final_qk_density=pairs/(24*16*tokens*tokens))

def main():
    runs={};per_scene={};all_runs=[];monitors={};summary={}
    for scene in SCENES:
        runs[scene]={};per_scene[scene]={}
        for method in METHODS:
            p=OUT/scene/method/'result.json'
            if not p.exists():raise RuntimeError(f'Missing run: {p}')
            d=json.loads(p.read_text());m=json.loads((OUT/f'{scene}_{method}_monitor.json').read_text())
            assert m['returncode']==0 and not m['same_gpu_overlap_detected'] and not m['foreign_gpu_pids']
            assert d['validation']['passed'] and d['validation']['cases']==8
            assert len(d['normal_samples'])==4 and len(d['event_profiles'])==3
            assert all(v is None for v in d['quality']['values'].values())
            runs[scene][method]=d;per_scene[scene][method]=decompose(d);monitors[f'{scene}/{method}']=m
            all_runs.append(d)
        a,b=(runs[scene][m] for m in METHODS)
        assert a['input']==b['input'] and a['hardware']==b['hardware']
        assert monitors[f'{scene}/saf3r']['gpu_uuid']==monitors[f'{scene}/shared_value_q128']['gpu_uuid']
        ha=json.loads((OUT/scene/'saf3r/source_sha256.json').read_text());hb=json.loads((OUT/scene/'shared_value_q128/source_sha256.json').read_text())
        assert ha==hb
    for method in METHODS:
        ds=[per_scene[s][method] for s in SCENES]
        summary[method]=dict(normal_mean_scene_median_seconds=mean(runs[s][method]['normal_warmed_median_seconds'] for s in SCENES),
            profile_cuda_interval_seconds=mean(d['profile_cuda_interval_seconds'] for d in ds),
            profile_wall_seconds=mean(d['profile_wall_seconds'] for d in ds),
            attention_components={k:mean(d['attention_components'][k] for d in ds) for k in ds[0]['attention_components']},
            common={k:mean(d['common'][k] for d in ds) for k in ds[0]['common']},
            details={k:mean(d['details'][k] for d in ds) for k in ds[0]['details']},
            final_qk_pairs=mean(d['final_qk_pairs'] for d in ds),final_qk_density=mean(d['final_qk_density'] for d in ds),
            peak_allocated_gib=max(p['peak_allocated_gib'] for s in SCENES for p in runs[s][method]['normal_samples'][1:]))
    sf,sv=(summary[m] for m in METHODS)
    normal_gap=sv['normal_mean_scene_median_seconds']-sf['normal_mean_scene_median_seconds']
    gap=sv['profile_cuda_interval_seconds']-sf['profile_cuda_interval_seconds']
    comp_gap={k:sv['attention_components'][k]-sf['attention_components'][k] for k in sv['attention_components']}
    qk_ratio=sv['final_qk_pairs']/sf['final_qk_pairs']
    pure_ratio=sv['details']['pure_qk_softmax_av']/sf['details']['pure_qk_softmax_av']
    attribution=dict(normal_gap_seconds=normal_gap,profile_gap_seconds=gap,attention_component_gap=comp_gap,
        final_qk_pair_ratio=qk_ratio,pure_av_interval_ratio=pure_ratio,
        pure_av_interval_ratio_divided_by_logical_work_ratio=pure_ratio/qk_ratio,
        caveat='Logical pair counts are not padded GPU FLOPs. Event intervals include queue/launch gaps and indexed memory access. Different supports and kernels; normalized ratio does not isolate a same-mask kernel-efficiency effect.')
    lines=['# Does shared Value only spend extra time constructing the sparse mask?','',
        'Matched heads/chess/stairs,128 identical frames each; same physical RTX4090 for each method pair, strict GPU leases and zero foreign overlap.',
        'Shared Value:group128,20% max selected-Key budget, nominal20% clusters, sampled history interval256; frozen d7f535ca method plus group-size patch.',
        'SAF3R:repository lazy first-correspondence-head Top4, profile configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json; not paper DINO Top-K.',
        'One warmup + three UNINSTRUMENTED full forwards, then three separate CUDA Event passes. Top-K recomputed inside every SAF3R forward.',
        'All reset/clustering/routing/history/reordering/output costs count; no algorithm, retention or gates changed.',
        'Probe/stride functions were split into measurement scopes AFTER normal timing and passed8 exact-path numerical checks against originals.',
        'Stages below are CUDA Event intervals, including CPU launch/queue gaps, not pure kernel self time; no Kineto used.',
        'Three-scene equal averages, NOT a full7Scenes evaluation.','',
        '## Normal complete-forward timings','',
        '| Scene | SAF3R s | shared Value group128 s | shared/SAF3R time ratio |','|---|---:|---:|---:|']
    for scene in SCENES:
        a,b=(runs[scene][m]['normal_warmed_median_seconds'] for m in METHODS)
        lines.append(f'| {scene} | {a:.6f} | {b:.6f} | {b/a:.4f}x |')
    lines.append(f"| Three-scene mean | {sf['normal_mean_scene_median_seconds']:.6f} | {sv['normal_mean_scene_median_seconds']:.6f} | {sv['normal_mean_scene_median_seconds']/sf['normal_mean_scene_median_seconds']:.4f}x |")
    lines+=['','## Additive top-level stage comparison','',
        'Support construction is defined BROADLY: Value clustering, shared geometry/pooling, coarse probes, residual candidates, prior, loss, budget solve and packing. This gives the hypothesis its strongest interpretation.',
        'Selected-support execution includes K/V preparation, actual attention and current-method evidence export. It excludes separately timed SAF3R selection.',
        '| Stage | SAF3R s | shared Value s | extra shared seconds |','|---|---:|---:|---:|']
    for k in sv['attention_components']:
        a,b=sf['attention_components'][k],sv['attention_components'][k]
        lines.append(f'| {k} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    for k in sv['common']:
        a,b=sf['common'][k],sv['common'][k]
        lines.append(f'| {k} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines.append(f"| Total profiled CUDA interval | {sf['profile_cuda_interval_seconds']:.6f} | {sv['profile_cuda_interval_seconds']:.6f} | {gap:+.6f} |")
    lines+=['','## Inside selected-support execution (included above)','',
        '| Component | SAF3R s | shared Value s |','|---|---:|---:|',
        f"| Actual QK/softmax/AV call intervals | {sf['details']['pure_qk_softmax_av']:.6f} | {sv['details']['pure_qk_softmax_av']:.6f} |",
        f"| Explicit K/V gather (SAF3R) | {sf['details']['explicit_kv_gather']:.6f} | Indexed inside AV |",
        f"| Per-selected-Key history probability export | Not executed | {sv['details']['history_probability_export']:.6f} |",
        f"| Dispatch/setup/other within execution | {sf['details']['dispatch_other']:.6f} | {sv['details']['execution_call_other']:.6f} |",'',
        'These are child scopes; do not add them to the preceding parent table. Shared AV gathers indexed K/V inside its Triton kernel and includes inline sampled history-probe writes; SAF3R materializes compact K/V banks for many SDPA calls. The separately measured export interval excludes those inline writes; evidence-enabled versus evidence-disabled kernels were not isolated here.',
        '', '## Support construction details (included above)','',
        '| Shared Value component | Seconds |','|---|---:|']
    for k in ('value_clustering','pooling_geometry','coarse_observation','residual_candidates','prior_loss_budget_pack','seed'):
        lines.append(f"| {k} | {sv['details'][k]:.6f} |")
    lines+=['','| SAF3R component | Seconds |','|---|---:|']
    for k in ('lazy_topk','query_probe_selection','stride_selection'):lines.append(f"| {k} | {sf['details'][k]:.6f} |")
    lines+=['','## Actual logical QK workload','',
        '| Quantity | SAF3R | Shared Value |','|---|---:|---:|',
        f"| Final attention QK pairs, all24 layers/all16 heads | {sf['final_qk_pairs']:.0f} | {sv['final_qk_pairs']:.0f} |",
        f"| Density relative to full global QK | {sf['final_qk_density']*100:.6f}% | {sv['final_qk_density']*100:.6f}% |",'',
        f'Work ratio(shared/SAF3R):{qk_ratio:.4f}x. QK/softmax/AV interval ratio:{pure_ratio:.4f}x.',
        f'Ratio after dividing by logical work ratio:{pure_ratio/qk_ratio:.4f}x. This is an effective path comparison, NOT a same-mask kernel microbenchmark or actual GPU FLOP-efficiency claim.',
        '', 'SAF3R profile has384 layer-heads:212 stride(157stride4,55stride128),85 first-frame,75query-probe(73keep~25%,2keep~1/128),5within-frame,7correspondence.',
        'Measured Key lengths include all configured first-frame/special unions: stride2704–34569; q-probe1041–33312; first/within-frame1041; correspondence512 patch keys plus special-query work.',
        'Shared Value uses up to26649 total keys/query, across every head. Actual shared supports vary by128-query group. SAF3R often shares the compact bank across ALL queries of a head.',
        '', '## Conclusion','',
        f"Even with broad support construction, extra support-building time is {comp_gap['support_construction']:.6f}s while extra selected-support execution is {comp_gap['selected_support_execution']:.6f}s; extra history bookkeeping is {comp_gap['history_summary_completion']:.6f}s.",
        'The hypothesis that ONLY mask construction is slower is false. Actual attention execution and additional historical evidence maintenance also differ.',
        'The attention difference combines more logical edges and a different execution path (indexed per-group banks versus many compact-bank SDPA calls); this experiment does not isolate each hardware effect.',
        '', '## Required quality fields for these diagnostics','',
        '| Scene | Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | vs Dense |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for scene in SCENES:
        for method in METHODS:lines.append(f'| {scene} | {method} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
    lines+=['','All six quality fields are N/A: explicit performance diagnostics; pose/recon_unposed not evaluated here. Complete prior quality stays in outputs/dense_saf3r_query128_20261104, a separate batch.',
        'No Dense rerun; Dense speedup N/A with reason. Reconstruction protocol if evaluated:0.05m F1 threshold,0.0078125m downsample,unsquared CD=(ACC+Comp)/2.',
        'Per-scene profiles, per-layer rows, workload counters, normal timing/memory samples, source archives and strict GPU monitors are archived beside this report.','']
    data=dict(summary=summary,attribution=attribution,per_scene=per_scene,monitors=monitors,complete=True,
        quality_evaluated=False,quality={k:None for k in ('auc03','auc30','acc','comp','cd','f1')},
        speedup_vs_dense=None,speedup_unavailable='No contemporaneous paired Dense; unchanged control not rerun',
        scope='Three128-frame sequences; performance diagnostic only')
    dump=lambda p,d:p.write_text(json.dumps(d,indent=2)+'\n')
    dump(OUT/'summary.json',data);(OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2));print(json.dumps(attribution,indent=2))

if __name__=='__main__':main()
