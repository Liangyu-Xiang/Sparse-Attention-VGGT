"""Reconcile non-overlapping CUDA-interval levels; keep warmed timing separate."""
from pathlib import Path
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/query_group_breakdown_20261104'
SCENES=('heads','chess','stairs')

def avg(v):
    v=list(v)
    return sum(v)/len(v)

def decompose(result):
    profiles=result['event_profiles']
    ev={name:avg(p['intervals'][name]['cuda_interval_seconds'] for p in profiles)
        for name in profiles[0]['intervals']}
    cpu={name:avg(p['intervals'][name]['cpu_submission_seconds'] for p in profiles)
        for name in profiles[0]['intervals']}
    get=lambda n:ev.get(n,0.)
    total=avg(p['cuda_forward_interval_seconds'] for p in profiles)
    wrapper=get('global/attention_wrapper')
    stage_total=sum(v for n,v in ev.items() if n.startswith('stage/'))
    projection=get('projection/qkv')+get('projection/output')
    model=dict(image_encoder=get('model/image_encoder'),frame_blocks=get('model/frame_blocks'),
        global_attention=wrapper,global_mlp=get('global/mlp'),
        global_block_other=get('model/global_blocks')-wrapper-get('global/mlp'),
        camera_head=get('model/camera_head'),depth_head=get('model/depth_head'),point_head=get('model/point_head'))
    model['outside_modules']=total-sum(model.values())
    attention=dict(value_clustering=get('stage/value_update'),pooling_geometry=get('stage/pooling'),
        coarse_observation=get('stage/m5c_coarse_observation')+get('stage/coarse_observation'),
        residual_candidates=get('stage/residual_fine_probe'),prior_loss_budget_pack=get('stage/reuse_and_budget'),
        sparse_av_export=get('stage/execution_and_evidence'),
        history_summary_completion=get('stage/sampled_probe_grid_summary')+get('stage/m5c_evidence_and_host_copy')+get('stage/history_summary'),
        seed=get('stage/cluster_seed_prepare'),qkv_output_projection=projection,
        attention_other=wrapper-stage_total-projection)
    assert abs(sum(model.values())-total)<1e-7
    assert abs(sum(attention.values())-wrapper)<1e-7
    return dict(cuda_interval_seconds=total,profile_wall_seconds=avg(p['wall_seconds'] for p in profiles),
        model=model,attention=attention,all_intervals=ev,cpu_submission=cpu,
        stage_calls=profiles[0]['intervals'])

def main():
    runs={};monitors={};decomposed={};summary={}
    for scene in SCENES:
        runs[scene]={};decomposed[scene]={}
        for size in (32,128):
            path=OUT/scene/f'q{size}/result.json'
            if not path.exists():raise RuntimeError(f'Missing completed run: {path}')
            d=json.loads(path.read_text())
            m=json.loads((OUT/f'{scene}_q{size}_monitor.json').read_text())
            assert not m['same_gpu_overlap_detected'] and not m['foreign_gpu_pids']
            assert m['returncode']==0 or (m['returncode']==-15 and d.get('trace_unavailable') and len(d['event_profiles'])==3)
            assert d['config']['query_group_size']==size
            assert all(l['size']==size for l in d['layers'])
            assert len(d['normal_samples'])==4 and len(d['event_profiles'])==3
            runs[scene][size]=d;monitors[f'{scene}_{size}']=m
            decomposed[scene][size]=decompose(d)
        a,b=runs[scene][32],runs[scene][128]
        assert a['input']==b['input']
        assert {k:v for k,v in a['config'].items() if k!='query_group_size'}=={k:v for k,v in b['config'].items() if k!='query_group_size'}
        assert monitors[f'{scene}_32']['gpu_uuid']==monitors[f'{scene}_128']['gpu_uuid']
        ha=json.loads((OUT/scene/'q32/source_sha256.json').read_text())
        hb=json.loads((OUT/scene/'q128/source_sha256.json').read_text())
        assert ha==hb
    for size in (32,128):
        ds=[decomposed[s][size] for s in SCENES]
        summary[size]=dict(normal_warmed_mean_scene_median_seconds=avg(runs[s][size]['normal_warmed_median_seconds'] for s in SCENES),
            mean_profile_cuda_interval_seconds=avg(d['cuda_interval_seconds'] for d in ds),
            mean_profile_wall_seconds=avg(d['profile_wall_seconds'] for d in ds),
            model={k:avg(d['model'][k] for d in ds) for k in ds[0]['model']},
            attention={k:avg(d['attention'][k] for d in ds) for k in ds[0]['attention']},
            all_intervals={k:avg(d['all_intervals'][k] for d in ds) for k in ds[0]['all_intervals']},
            cpu_submission={k:avg(d['cpu_submission'][k] for d in ds) for k in ds[0]['cpu_submission']},
            peak_allocated_gib=max(p['peak_allocated_gib'] for s in SCENES for p in runs[s][size]['normal_samples'][1:]),
            peak_reserved_gib=max(p['peak_reserved_gib'] for s in SCENES for p in runs[s][size]['normal_samples'][1:]))
    speedup=summary[32]['normal_warmed_mean_scene_median_seconds']/summary[128]['normal_warmed_mean_scene_median_seconds']
    lines=['# Query-group32 vs128: matched forward and breakdown','',
        'Three sequences only:7Scenes heads/chess/stairs,128 identical frames each. Not a dataset quality evaluation.',
        'Same checkpoint, preprocessing, BF16, TF32 off,20% selected-Key budget, nominal20% Value clusters, history interval256.',
        'Each scene pair on the SAME exclusive RTX4090. All six GPU monitor files show zero foreign-context overlap.',
        'Only query_group_size changes. Four coarse probes/group, evenly spaced within the actual group.',
        'Source: d7f535ca plus prior group-size patch. No algorithm/retention/quality-gate retuning.',
        'Dense not rerun per AGENTS.md; speedup vs Dense=N/A (no contemporaneous paired timing).',
        '', '## Normal complete-forward timing','',
        'One warmup, three uninstrumented full forwards. Table uses warmed median. Every initialization/clustering/routing/history/graph/output cost counts.',
        '| Scene | group32 s | group128 s | 128 vs32 speedup |', '|---|---:|---:|---:|']
    for scene in SCENES:
        a=runs[scene][32]['normal_warmed_median_seconds'];b=runs[scene][128]['normal_warmed_median_seconds']
        lines.append(f'| {scene} | {a:.6f} | {b:.6f} | {a/b:.4f}x |')
    lines.append(f"| Equal-scene mean | {summary[32]['normal_warmed_mean_scene_median_seconds']:.6f} | {summary[128]['normal_warmed_mean_scene_median_seconds']:.6f} | {speedup:.4f}x |")
    lines+=['','## CUDA Event interval hierarchy','',
        'Three ADDITIONAL instrumented forwards per run; mean across repeats, then equal-scene mean.',
        'These intervals include CPU launch/synchronization/queue gaps. They are NOT kernel self time and are NOT the normal benchmark median.',
        'Each table below is internally additive; the attention table EXPANDS a row of the model table. Do not add the two tables together.','',
        '| Model component | group32 s | group128 s | saved s | 32/128 |','|---|---:|---:|---:|---:|']
    def row(name,a,b):return f'| {name} | {a:.6f} | {b:.6f} | {a-b:+.6f} | {a/b:.3f}x |' if b else f'| {name} | {a:.6f} | {b:.6f} | {a-b:+.6f} | N/A |'
    for k in summary[32]['model']:lines.append(row(k,summary[32]['model'][k],summary[128]['model'][k]))
    lines.append(row('TOTAL profile CUDA interval',summary[32]['mean_profile_cuda_interval_seconds'],summary[128]['mean_profile_cuda_interval_seconds']))
    lines+=['','| Inside global attention wrapper | group32 s | group128 s | saved s | 32/128 |','|---|---:|---:|---:|---:|']
    for k in summary[32]['attention']:lines.append(row(k,summary[32]['attention'][k],summary[128]['attention'][k]))
    lines+=['','## Details inside prior/loss/budget/packing (included above)','',
        '| Scope | GPU interval32 s | GPU interval128 s | CPU submission32 s | CPU submission128 s |','|---|---:|---:|---:|---:|']
    for k in ('selection_detail/prior_completion','selection_detail/selector_total','selector_detail/geometry_copies','selector_detail/input_copy_graph_replay'):
        lines.append(f"| {k} | {summary[32]['all_intervals'].get(k,0):.6f} | {summary[128]['all_intervals'].get(k,0):.6f} | {summary[32]['cpu_submission'].get(k,0):.6f} | {summary[128]['cpu_submission'].get(k,0):.6f} |")
    lines+=['','## Logical workload and calls (heads)','',
        '| Item | group32 | group128 |','|---|---:|---:|']
    workload={}
    for size in (32,128):
        d=runs['heads'][size];ls=d['layers'];pr=d['event_profiles'][0]
        workload[size]=dict(tokens=d['input']['shape'][0]*(28*37+5),groups=ls[0]['groups'],
            history_sampled_groups=ls[0]['history_sampled_groups'],clusters=ls[0]['fine_clusters'],
            attention_query_tiles_per_layer=(133248+31)//32,
            coarse_window_calls_all24=pr['intervals']['stage/m5c_coarse_observation']['calls'],
            selector_window_calls_all23=pr['intervals']['stage/reuse_and_budget']['calls'],
            main_QK_per_head_all24=sum(l['main_qk_per_head'] for l in ls),
            coarse_QK_per_head_all24=sum(l['coarse_qk_per_head'] for l in ls),
            mean_actual_density=avg(l['actual_density'] for l in ls),
            graph_cache=d['graph_cache_after_timing'])
    for key in workload[32]:lines.append(f'| {key} | {workload[32][key]} | {workload[128][key]} |')
    traces={}
    for size in (32,128):
        path=OUT/'heads'/f'q{size}/kernel_summary.json'
        if path.exists():traces[size]=json.loads(path.read_text())
    if len(traces)==2:
        names=set(traces[32]['kernels'])|set(traces[128]['kernels'])
        top=sorted(names,key=lambda n:-(traces[32]['kernels'].get(n,{}).get('seconds',0)+traces[128]['kernels'].get(n,{}).get('seconds',0)))[:30]
        lines+=['','## Heads: separate Kineto GPU kernel durations','',
            'One extra instrumented pass per variant, independent of the event passes and warmed benchmark. Kernel sums may overlap across streams.',
            '| Kernel name | group32 s | group128 s | calls32 | calls128 |','|---|---:|---:|---:|---:|']
        for name in top:
            a=traces[32]['kernels'].get(name,dict(seconds=0,calls=0));b=traces[128]['kernels'].get(name,dict(seconds=0,calls=0))
            lines.append(f"| {name.replace('|','/')} | {a['seconds']:.6f} | {b['seconds']:.6f} | {a['calls']} | {b['calls']} |")
        lines+=['',f"Kernel-duration sums32/128: {traces[32]['total_kernel_seconds']:.6f}s / {traces[128]['total_kernel_seconds']:.6f}s.",
            f"Device-busy union32/128: {traces[32]['device_busy_union_seconds']:.6f}s / {traces[128]['device_busy_union_seconds']:.6f}s."]
    if len(traces)!=2:
        lines+=['','## Optional kernel trace unavailable','',
            'Heads/group32 Kineto tracing stalled with an idle GPU after all normal/event measurements had completed, and was terminated. Saved normal timings and three Event passes are intact; the raw monitor retains returncode=-15 for that optional phase. Group128 ran without Kineto. No kernel-self-time claim is made.']
    lines+=['','## Required quality fields (performance diagnostics only)','',
        '| Scene | Query group | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | vs Dense |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for scene in SCENES:
        for size in (32,128):lines.append(f'| {scene} | {size} | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
    lines+=['','All six quality metrics N/A: these runs explicitly profile performance and do not execute pose/recon_unposed evaluation.',
        'Existing128-group geometric scores remain in the previous experiment; they are not new measurements here.',
        'F1 threshold0.05m, downsample0.0078125m if quality were evaluated; CD=(ACC+Comp)/2 unsquared metres.',
        'Raw per-run normal timing samples, memory, event samples/per-layer scopes, source archives, monitors and the optional trace-failure record are beside this report; no usable Kineto trace was produced.','']
    result=dict(scenes=SCENES,frames=128,groups=[32,128],summary=summary,decomposed_per_scene=decomposed,
        normal_speedup_128_over_32=speedup,heads_workload=workload,monitors=monitors,
        quality_evaluated=False,quality_values={k:None for k in ('auc03','auc30','acc','comp','cd','f1')},
        quality_unavailable='Explicit performance diagnostic; pose/recon_unposed not evaluated',
        speedup_vs_dense=None,speedup_unavailable='No contemporaneous paired Dense; unchanged control not rerun per AGENTS.md')
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2));print('SPEEDUP',speedup)

if __name__=='__main__':main()
