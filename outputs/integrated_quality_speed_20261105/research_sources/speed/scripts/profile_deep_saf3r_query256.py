"""Matched support-construction vs execution audit; no Kineto, no algorithm retuning."""
import argparse
from collections import defaultdict
from contextlib import contextmanager,nullcontext
from dataclasses import asdict
from functools import wraps
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]

def dump(p,d):
    p.write_text(json.dumps(d,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n')


def instrumented_q_probe(q,k,v,topk,q_sample_ratio=1,indices=None,scope=lambda n:nullcontext()):
    import torch
    import torch.nn.functional as F
    assert q.shape[0]==1 and q.shape[1]==1
    assert k.shape[0]==1 and k.shape[1]==1
    assert v.shape[0]==1 and v.shape[1]==1
    _,_,n,d=q.shape
    with scope('sf_select/q_probe'):
        q_sel=q
        if q_sample_ratio<1:
            count=max(1,int(q.size(2)*q_sample_ratio))
            pos=torch.linspace(0,q.size(2)-1,steps=count).long()
            q_sel=q[:,:,pos,:]
        q_bar=q_sel.mean(dim=2,keepdim=True)
        scores=(torch.matmul(q_bar.to(torch.float32),k.transpose(-1,-2).to(torch.float32))/math.sqrt(d)).squeeze()
        ids=torch.topk(scores,k=topk,dim=-1,sorted=False).indices
        if indices is not None:
            indices=indices.to(device=ids.device,dtype=torch.long)
            ids=torch.unique(torch.cat([ids,indices]))
        else:ids=torch.sort(ids).values
    with scope('sf_pack/q_probe_kv'):
        kt=k[:,:,ids,:];vt=v[:,:,ids,:]
    return F.scaled_dot_product_attention(q,kt,vt)


def instrumented_stride(q,k,v,stride,start_pos=0,indices=None,scope=lambda n:nullcontext()):
    import torch
    import torch.nn.functional as F
    if indices is None:
        with scope('sf_pack/stride_kv'):
            kt=k[:,:,start_pos::stride,:];vt=v[:,:,start_pos::stride,:]
        return F.scaled_dot_product_attention(q,kt,vt)
    with scope('sf_select/stride_indices'):
        ids=torch.arange(start_pos,k.shape[2],step=stride,device=k.device,dtype=torch.long)
        indices=indices.to(device=k.device,dtype=torch.long)
        ids=torch.unique(torch.cat([ids,indices]))
    with scope('sf_pack/stride_kv'):
        kt=k[:,:,ids,:];vt=v[:,:,ids,:]
    return F.scaled_dot_product_attention(q,kt,vt)


def validate_refactoring(device):
    import torch
    sp=importlib.import_module('saf3r.patch.sparse_attention')
    torch.manual_seed(761)
    q=torch.randn(1,1,61,32,device=device);k=torch.randn_like(q);v=torch.randn_like(q)
    for fraction in (1,.5):
        for ids in (None,torch.tensor([0,1,5],device=device)):
            a=sp.q_probe_topk_attention(q,k,v,13,fraction,ids)
            b=instrumented_q_probe(q,k,v,13,fraction,ids)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    for offset in (0,2):
        for ids in (None,torch.tensor([0,1,5],device=device)):
            a=sp.stride_attention(q,k,v,4,offset,ids)
            b=instrumented_stride(q,k,v,4,offset,ids)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    return dict(device=device,cases=8,passed=True)


def run(a):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.patch import update_args_per_forward
    import saf3r.patch.reuse_residual_exploration as core
    from scripts.quality_metrics import quality_record,quality_markdown
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    quality=quality_record({},'Explicit support/execution performance diagnostic; pose and recon_unposed not evaluated')
    quality['speedup_unavailable']='No new matched Dense timing; unchanged control not rerun per AGENTS.md'
    dump(out/'quality.json',quality);(out/'quality.md').write_text(quality_markdown(quality))
    dump(out/'status.json',dict(stage='loading'))
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    validation=validate_refactoring('cpu')
    dataset=MV_REGISTRY.get('7scenes')()
    positions,scene=sample_frames(dataset.get_data(a.scene),a.scene,a.frames,sample_mode='uniform',seed=0)
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    cfg.sparse_config_path=str(ROOT/cfg.sparse_config_path)
    if a.method!='saf3r':cfg.pop('patch_module');cfg.pop('sparse_config_path')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    state=None
    shared_cfg=json.loads(a.config.read_text())
    if a.method=='shared_value_q256':
        state=core.install_reuse_residual_exploration(model,core.ReuseExploreConfig(**shared_cfg))
    else:update_args_per_forward(model,'vggt',*images.shape[-2:])
    canonical=dict(scene=a.scene,frame_indices=positions,image_paths=scene.image_files,shape=list(images.shape),checkpoint=cfg.ckpt_path)
    dump(out/'input.json',canonical)
    dump(out/'config.json',dict(method=a.method,model=dict(cfg),shared_config=shared_cfg if state else None,
        saf3r_profile='configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json',saf3r_topk=4,saf3r_topk_source='lazy first-correspondence head, NOT paper DINO'))
    paths=list((ROOT/'saf3r').rglob('*.py'))+[Path(__file__),ROOT/'scripts/deep_operation_timing.py',ROOT/'configs/evaluation/reuse_shared_value_loss_q256.json',ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']
    paths=list(dict.fromkeys(paths+[a.config]))
    dump(out/'source_sha256.json',{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'scripts/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
        for p in paths:archive.add(p,arcname=(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'scripts/'+p.name))
    samples=[]
    def forward():
        if a.method=='saf3r':update_args_per_forward(model,'vggt',*images.shape[-2:])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):return model(images)
    for i in range(4):
        dump(out/'status.json',dict(stage='forward',iteration=i))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        pred=forward();torch.cuda.synchronize()
        row=dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        samples.append(row);del pred
        print('NORMAL',a.scene,a.method,i,row,flush=True)
    result=dict(method=a.method,input=canonical,quality=quality,normal_samples=samples,
        normal_warmed_median_seconds=statistics.median(x['seconds'] for x in samples[1:]),
        hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        validation=validation,scope='Complete forward;1warmup+3normal repeats,3separate CUDA Event passes; no Kineto',
        threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample)
    dump(out/'timing.json',result)
    intervals=[];stacks=defaultdict(list);handles=[];context=dict(head=None)
    workload=defaultdict(lambda:dict(calls=0,qk_pairs=0,max_keys=0,min_keys=None))
    @contextmanager
    def scope(name,layer=None):
        beg=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        beg.record();wall=time.perf_counter()
        try:yield
        finally:
            cpu=time.perf_counter()-wall;end.record();intervals.append((name,layer,beg,end,cpu))
    def hook(module,name,layer=None):
        def pre(m,args):
            cm=scope(name,layer);stacks[id(m)].append(cm);cm.__enter__()
        def post(m,args,out):stacks[id(m)].pop().__exit__(None,None,None)
        handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
    hook(model.aggregator.patch_embed,'model/image_encoder')
    for i,b in enumerate(model.aggregator.frame_blocks):hook(b,'model/frame_blocks',i)
    for i,b in enumerate(model.aggregator.global_blocks):
        hook(b,'model/global_blocks',i);hook(b.attn,'global/attention_wrapper',i);hook(b.mlp,'global/mlp',i)
        original=b.attn.orig_module if a.method=='saf3r' else b.attn.original
        hook(original.qkv,'projection/qkv',i);hook(original.proj,'projection/output',i)
        b.attn._audit_layer=i
    for name in ('camera_head','depth_head','point_head'):
        m=getattr(model,name,None)
        if m is not None:hook(m,'model/'+name)
    if state is not None:
        @contextmanager
        def stage(s,name):
            with scope('shared_stage/'+name,s.previous_layer+1):yield
        core.stage=stage
        state.direct_scope=lambda name:scope('shared_execution/'+name)
    else:
        wrapper=importlib.import_module('saf3r.patch.wrapper')
        sp=importlib.import_module('saf3r.patch.sparse_attention')
        old_head=wrapper.BaseHeadwiseSparseAttentionWrapper.sparse_attention
        @wraps(old_head)
        def head(self,q,k,v,cfg,head_idx=0):
            previous=context['head'];context['head']=(self._audit_layer,head_idx,cfg['mode'])
            try:
                with scope('sf_head/'+cfg['mode'],self._audit_layer):return old_head(self,q,k,v,cfg,head_idx)
            finally:context['head']=previous
        wrapper.BaseHeadwiseSparseAttentionWrapper.sparse_attention=head
        old_dispatch=wrapper.sparse_attention
        @wraps(old_dispatch)
        def dispatch(*args,**kwargs):
            mode=args[4]['mode']
            with scope('sf_dispatch/'+mode,context['head'][0]):return old_dispatch(*args,**kwargs)
        wrapper.sparse_attention=dispatch
        sp.q_probe_topk_attention=lambda q,k,v,topk,q_sample_ratio=1,indices=None:instrumented_q_probe(q,k,v,topk,q_sample_ratio,indices,scope)
        sp.stride_attention=lambda q,k,v,stride,start_pos=0,indices=None:instrumented_stride(q,k,v,stride,start_pos,indices,scope)
        def count(name,q,k):
            row=workload[name];keys=k.shape[-2]
            batches=math.prod(torch.broadcast_shapes(q.shape[:-2],k.shape[:-2]))
            row['calls']+=1;row['qk_pairs']+=batches*q.shape[-2]*keys
            row['max_keys']=max(row['max_keys'],keys)
            row['min_keys']=keys if row['min_keys'] is None else min(row['min_keys'],keys)
        old_sdpa=F.scaled_dot_product_attention
        @wraps(old_sdpa)
        def sdpa(q,k,v,*args,**kwargs):
            if context['head'] is None:return old_sdpa(q,k,v,*args,**kwargs)
            mode=context['head'][2];count(mode,q,k)
            with scope('sf_av/sdpa/'+mode,context['head'][0]):return old_sdpa(q,k,v,*args,**kwargs)
        F.scaled_dot_product_attention=sdpa
        for name in ('fused_sparse_topk_attention','fused_sparse_topk_attention_2'):
            old=getattr(sp,name)
            def make_fused(fn):
                @wraps(fn)
                def fused(q,k,v,ids,*args,**kwargs):
                    row=workload['dino_topk_fused'];row['calls']+=1;row['qk_pairs']+=q.shape[0]*ids.shape[-1]
                    row['max_keys']=max(row['max_keys'],ids.shape[-1]);row['min_keys']=ids.shape[-1]
                    with scope('sf_av/correspondence_fused',context['head'][0]):return fn(q,k,v,ids,*args,**kwargs)
                return fused
            setattr(sp,name,make_fused(old))
        # First normal forwards have already selected the compatible Top-K backend.
        backend=sys.modules.get('saf3r.triton.fused_qk_topk')
        owner=backend if backend is not None else importlib.import_module('saf3r.patch.utils')
        name='compute_qk_topk_indices_fused' if backend is not None else 'compute_qk_topk_indices_batched'
        old_topk=getattr(owner,name)
        @wraps(old_topk)
        def topk(*args,**kwargs):
            with scope('sf_select/lazy_topk'):return old_topk(*args,**kwargs)
        setattr(owner,name,topk);result['saf3r_topk_backend']=name
        if backend is not None:
            original_kernel=backend._fused_perframe_qk_topk_kernel
            class MeasuredKernel:
                def __getitem__(self,grid):
                    launch=original_kernel[grid]
                    def measured(*args,**kwargs):
                        with scope('sf_detail/fused_correspondence_qk_topk'):
                            return launch(*args,**kwargs)
                    return measured
            backend._fused_perframe_qk_topk_kernel=MeasuredKernel()
    recorder=None
    if state is not None:
        from scripts.deep_operation_timing import install_deep
        recorder,deep_validation,deep_handles=install_deep(model,state,scope)
        result['deep_validation']=deep_validation
        result['device_clock_calibration']=recorder.calibration
        # Recapture a marked graph cache in an explicitly untimed warm forward.
        dump(out/'status.json',dict(stage='instrumentation_warmup'))
        pred=forward();torch.cuda.synchronize();del pred
        recorder.reset();intervals.clear()
    profiles=[]
    for i in range(3):
        dump(out/'status.json',dict(stage='event_profile',iteration=i))
        intervals.clear();workload.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        if recorder is not None:recorder.reset()
        beg=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();beg.record();pred=forward();end.record();torch.cuda.synchronize()
        wall=time.perf_counter()-start;del pred
        by=defaultdict(lambda:dict(cuda_interval_seconds=0.,cpu_submission_seconds=0.,calls=0))
        per_layer=defaultdict(lambda:defaultdict(float))
        for name,layer,first,last,cpu in intervals:
            seconds=first.elapsed_time(last)/1000
            by[name]['cuda_interval_seconds']+=seconds;by[name]['cpu_submission_seconds']+=cpu;by[name]['calls']+=1
            if layer is not None:per_layer[layer][name]+=seconds
        work=dict(workload)
        if state is not None:
            heads=model.aggregator.global_blocks[0].attn.original.num_heads
            work=dict(final_qk_pairs=sum(float(x['direct_logical_main_qk_per_head'])*heads for x in state.records),
                mean_density=sum(float(x['actual_unique_density']) for x in state.records)/24,
                groups=state.groups,patch_budget=state.budget,specials=len(state.specials),
                history_sampled_groups=state.records[0]['history_sampled_groups'])
        profile=dict(wall_seconds=wall,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,cuda_interval_seconds=beg.elapsed_time(end)/1000,
            intervals=dict(by),per_layer=dict(per_layer),workload=work)
        profile['host_boundary_and_clock_residual_seconds']=wall-profile['cuda_interval_seconds']
        if recorder is not None:
            profile['graph_detail']=recorder.collect()
            expected=math.ceil(state.groups/state.config.window_groups)*23
            assert profile['graph_detail']['captured_rows']==expected,(profile['graph_detail']['captured_rows'],expected)
            profile['graph_cache']=dict(hits=state.graph_cache_hits,misses=state.graph_cache_misses,evictions=state.graph_cache_evictions,constant_updates=state.graph_constant_updates,creations=getattr(state,'graph_cache_creations',state.graph_cache_misses),live_tensor_bytes=getattr(state,'graph_cache_bytes',None))
        dump(out/f'event_profile_{i}.json',profile);profiles.append(profile)
        print('PROFILE',a.scene,a.method,i,wall,flush=True)
    result['event_profiles']=profiles
    result['event_note']='CUDA Event intervals include queue and CPU launch gaps; not kernel self time. Parent/child namespaces overlap. Refactored SAF3R q-probe/stride only split scopes, numerically checked against original; other code unchanged.'
    dump(out/'result.json',result);dump(out/'status.json',dict(stage='diagnostic_complete'))
    print('COMPLETE',a.scene,a.method,result['normal_warmed_median_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=int,choices=[2,128],default=128)
    p.add_argument('--scene',choices=['heads','chess','stairs'],required=True)
    p.add_argument('--method',choices=['saf3r','shared_value_q256'],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/reuse_shared_value_loss_q256.json')
    p.add_argument('--source-root',type=Path,default=ROOT)
    a=p.parse_args()
    ROOT=a.source_root.resolve();sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
    try:run(a)
    except Exception as exc:
        if a.output_dir.exists():dump(a.output_dir/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
