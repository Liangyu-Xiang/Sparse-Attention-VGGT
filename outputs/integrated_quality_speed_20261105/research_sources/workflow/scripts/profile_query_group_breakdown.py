"""Matched actual Query-group32/128 diagnostic with event hierarchy and CUDA trace.

Times normal forwards before installing hooks. Three separate event-profiled forwards
measure intervals including CPU/queue gaps. A further optional Kineto pass measures
kernel durations without CUDA Event instrumentation. Quality is explicitly N/A.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from functools import wraps
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.quality_metrics import quality_record,quality_markdown


def save(path,value):
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')


def run(a):
    import numpy as np
    import torch
    import yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
    import saf3r.patch.reuse_residual_exploration as core
    from saf3r.patch import reuse_sampled_history as history, reuse_window_graph as selector
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    out=a.output_dir
    out.mkdir(parents=True,exist_ok=False)
    q=quality_record({},'Explicit performance diagnostic: pose/recon_unposed not evaluated; no quality-evaluation claim')
    q['speedup_unavailable']='No contemporaneous paired Dense full-forward; unchanged Dense not rerun per AGENTS.md'
    save(out/'quality.json',q);(out/'quality.md').write_text(quality_markdown(q))
    save(out/'status.json',dict(stage='loading',scene=a.scene,query_group_size=a.group_size))
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=MV_REGISTRY.get('7scenes')()
    indices,scene=sample_frames(dataset.get_data(a.scene),a.scene,128,sample_mode='uniform',seed=0)
    cfg=json.loads((ROOT/'configs/evaluation/reuse_shared_value_loss_256.json').read_text())
    cfg['query_group_size']=a.group_size
    config=ReuseExploreConfig(**cfg)
    model_cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    model_cfg.pop('patch_module');model_cfg.pop('sparse_config_path')
    model_cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(model_cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    state=install_reuse_residual_exploration(model,config)
    canonical=dict(scene=a.scene,frame_indices=indices,image_paths=scene.image_files,shape=list(images.shape),checkpoint=model_cfg.ckpt_path)
    save(out/'input.json',canonical);save(out/'config.json',asdict(config))
    # Archive all method/model/evaluation source, including this standalone instrumentation.
    files=list((ROOT/'saf3r').rglob('*.py'))+[Path(__file__),ROOT/'configs/evaluation/reuse_shared_value_loss_256.json',ROOT/'scripts/run_gpu_lease.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(out/'source_sha256.json',hashes)
    with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
        for p in files:archive.add(p,arcname=str(p.relative_to(ROOT)))
    save(out/'protocol.json',dict(hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        warmup=1,repeats=3,event_repeats=3,frames=128,quality_tasks_evaluated=[],
        quality_protocol_if_evaluated=dict(tasks=['pose','recon_unposed'],f1_threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample),
        sparse_profile='Shared Value loss; no SAF3R head profile or correspondence Top-K',
        scope='Complete forward: includes reset, clustering, pooling, selector, history, graph setup/copy, routing and output heads',
        quality=q))
    def forward():
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            prediction=model(images)
        return prediction
    samples=[]
    for repeat in range(4):
        save(out/'status.json',dict(stage='forward',iteration=repeat))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();beg=time.perf_counter()
        prediction=forward();torch.cuda.synchronize()
        sample=dict(seconds=time.perf_counter()-beg,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        samples.append(sample);del prediction
        print('FORWARD',a.scene,a.group_size,repeat,sample,flush=True)
    def layer_record():
        return [dict(layer=x['layer'],groups=x['query_groups'],size=x['query_group_size'],
            history_sampled_groups=x['history_sampled_groups'],fine_clusters=x['fine_clusters'],
            actual_density=float(x['actual_unique_density']),main_qk_per_head=float(x['direct_logical_main_qk_per_head']),
            coarse_qk_per_head=x['extra_coarse_probe_qk_per_head'],geometry_width=x['geometry_width'],
            unused_patch_budget=float(x['unused_budget_mean']),stored_history_bytes=x['stored_history_bytes']) for x in state.records]
    layers=layer_record();save(out/'layers.json',layers)
    result=dict(input=canonical,config=asdict(config),quality=q,normal_samples=samples,
        normal_warmed_median_seconds=statistics.median(x['seconds'] for x in samples[1:]),
        graph_cache_after_timing=dict(hits=state.graph_cache_hits,misses=state.graph_cache_misses,
            evictions=state.graph_cache_evictions,constant_updates=state.graph_constant_updates),
        layers=layers)
    save(out/'timing.json',result)
    # Each namespace below is a separate hierarchy. Never add parent and child scopes.
    active=dict(mode='events');intervals=[];handles=[];module_stacks=defaultdict(list)
    @contextmanager
    def interval(name,layer=None):
        if active['mode']=='trace':
            with torch.profiler.record_function('qgroup/'+name):yield
            return
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();wall=time.perf_counter()
        try:yield
        finally:
            elapsed=time.perf_counter()-wall;end.record()
            intervals.append((name,layer,start,end,elapsed))
    @contextmanager
    def staged(s,name):
        with interval('stage/'+name,s.previous_layer+1):yield
    core.stage=staged
    def hook(module,name,layer=None):
        def before(m,args):
            cm=interval(name,layer);module_stacks[id(m)].append(cm);cm.__enter__()
        def after(m,args,output):module_stacks[id(m)].pop().__exit__(None,None,None)
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    hook(model.aggregator.patch_embed,'model/image_encoder')
    for idx,block in enumerate(model.aggregator.frame_blocks):hook(block,'model/frame_blocks',idx)
    for idx,block in enumerate(model.aggregator.global_blocks):
        hook(block,'model/global_blocks',idx);hook(block.attn,'global/attention_wrapper',idx)
        hook(block.mlp,'global/mlp',idx)
        hook(block.attn.original.qkv,'projection/qkv',idx)
        hook(block.attn.original.proj,'projection/output',idx)
    for name in ('camera_head','depth_head','point_head'):
        mod=getattr(model,name,None)
        if mod is not None:hook(mod,'model/'+name)
    def wrap(owner,name,label):
        original=getattr(owner,name)
        @wraps(original)
        def wrapped(*args,**kwargs):
            with interval(label):return original(*args,**kwargs)
        setattr(owner,name,wrapped)
    wrap(history,'build_prior','selection_detail/prior_completion')
    wrap(selector,'select','selection_detail/selector_total')
    wrap(selector.Window,'update_constants','selector_detail/geometry_copies')
    wrap(selector.Window,'__call__','selector_detail/input_copy_graph_replay')
    state.direct_scope=lambda name:interval('execution_detail/'+name)
    profiles=[]
    for repeat in range(3):
        save(out/'status.json',dict(stage='event_profile',iteration=repeat))
        intervals.clear();torch.cuda.synchronize()
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
        first.record();wall=time.perf_counter();prediction=forward();last.record();torch.cuda.synchronize()
        wall=time.perf_counter()-wall
        byname=defaultdict(lambda:dict(cuda_interval_seconds=0.,cpu_submission_seconds=0.,calls=0))
        perlayer=defaultdict(lambda:defaultdict(float))
        for name,layer,start,end,cpu in intervals:
            seconds=start.elapsed_time(end)/1000
            byname[name]['cuda_interval_seconds']+=seconds;byname[name]['cpu_submission_seconds']+=cpu;byname[name]['calls']+=1
            if layer is not None:perlayer[layer][name]+=seconds
        profile=dict(wall_seconds=wall,cuda_forward_interval_seconds=first.elapsed_time(last)/1000,
            intervals=dict(byname),per_layer=dict(perlayer))
        profiles.append(profile);del prediction
        save(out/f'event_profile_{repeat}.json',profile)
        print('EVENT_PROFILE',a.scene,a.group_size,repeat,wall,flush=True)
    result['event_profiles']=profiles
    result['event_median']={name:{key:statistics.median(p['intervals'][name][key] for p in profiles)
        for key in ('cuda_interval_seconds','cpu_submission_seconds','calls')} for name in profiles[0]['intervals']}
    result['event_note']='CUDA Event intervals include CPU launch/queue gaps; CPU submission spans include GPU synchronization in algorithm. Hierarchies overlap. Never sum parent and child scopes or label intervals as kernel self time.'
    if a.trace:
        save(out/'status.json',dict(stage='kernel_trace'))
        intervals.clear();active['mode']='trace';torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,with_stack=False,profile_memory=False) as prof:
            prediction=forward();torch.cuda.synchronize()
        trace_wall=time.perf_counter()-start;del prediction
        prof.export_chrome_trace(str(out/'kernel_trace.json'))
        gpu=defaultdict(lambda:dict(seconds=0.,calls=0));cpu=[];gpu_intervals=[]
        for event in prof.events():
            if event.device_type==torch.autograd.DeviceType.CUDA:
                duration=event.time_range.elapsed_us()/1e6
                gpu[event.name]['seconds']+=duration;gpu[event.name]['calls']+=1
                gpu_intervals.append((event.time_range.start,event.time_range.end))
        for event in prof.key_averages():
            if event.key.startswith('qgroup/'):
                cpu.append(dict(name=event.key,calls=event.count,cpu_total_seconds=event.cpu_time_total/1e6,
                    cpu_self_seconds=event.self_cpu_time_total/1e6,attributed_device_seconds=event.device_time_total/1e6))
        merged=[]
        for start,end in sorted(gpu_intervals):
            if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],end)
            else:merged.append([start,end])
        trace=dict(wall_seconds=trace_wall,kernels=dict(sorted(gpu.items(),key=lambda x:-x[1]['seconds'])),
            total_kernel_seconds=sum(x['seconds'] for x in gpu.values()),
            device_busy_union_seconds=sum(end-start for start,end in merged)/1e6,scope_attribution=cpu,
            note='Separate instrumented Kineto pass; not warmed benchmark time. Kernel durations can overlap across streams; scope attribution is hierarchical and may omit CUDA Graph kernel attribution.')
        save(out/'kernel_summary.json',trace);result['kernel_trace_summary']=trace
    save(out/'result.json',result);save(out/'status.json',dict(stage='diagnostic_complete'))
    print('COMPLETE',a.scene,a.group_size,result['normal_warmed_median_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=['heads','chess','stairs'],required=True)
    p.add_argument('--group-size',type=int,choices=[32,128],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--trace',action='store_true')
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        if a.output_dir.exists():
            reason=f'Performance diagnostic failed: {type(exc).__name__}: {exc}; no pose/recon evaluation'
            q=quality_record({},reason);save(a.output_dir/'quality.json',q)
            (a.output_dir/'quality.md').write_text(quality_markdown(q))
            save(a.output_dir/'status.json',dict(stage='failed',reason=reason,traceback=traceback.format_exc()))
        raise
