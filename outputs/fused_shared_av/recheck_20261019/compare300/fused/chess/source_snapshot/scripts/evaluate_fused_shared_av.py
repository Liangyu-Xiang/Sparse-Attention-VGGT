"""Opt-in fused A/V evaluation; primary warmed timing has NO layer sync hooks."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')


def run(args):
    import torch
    import open3d as o3d
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from saf3r.patch.fused_shared_av import FusedSharedAVConfig,install_fused_shared_av,DEFAULT_ALPHA
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from evaluate_da3bench import load_scene_data,evaluate
    if args.warmup<0 or args.repeats<1:raise ValueError('Invalid timing repetitions')
    out=args.output_dir/args.mode/args.scene
    if (out/'result.json').exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0)
    dataset=MV_REGISTRY.get(args.dataset)()
    original=dataset.get_data(args.scene)
    positions,scene=sample_frames(original,args.scene,args.frames,sample_mode='uniform',seed=0)
    model_cfg=Dict(name='VGGT',dpt_only=True,ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'))
    if args.mode=='saf3r':
        model_cfg.sparse_config_path=str(ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json')
        model_cfg.patch_module=dict(type='headsparse',topk_mode='token',lazy_dino_topk=True,topk=args.saf3r_topk)
    model=build_model(model_cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    ratios=DEFAULT_ALPHA
    if args.mode=='value':ratios=(0.,)*24
    if args.mode=='attention':ratios=(0.,)+(1.,)*23
    settings=dict(attention_ratios=ratios,retentions=(args.retention,)*24,window_groups=args.window_groups,capture=args.capture)
    if args.schedule:settings.update(json.loads(args.schedule.read_text()))
    # Ablations must not silently inherit the mixed alpha array from a schedule.
    if args.mode=='value':settings['attention_ratios']=(0.,)*24
    if args.mode=='attention':settings['attention_ratios']=(0.,)+(1.,)*23
    config=FusedSharedAVConfig(**settings)
    state=None if args.mode in ('dense','saf3r') else install_fused_shared_av(model,config)
    sources=['saf3r/patch/fused_shared_av.py','saf3r/triton/fused_shared_av.py',
        'saf3r/triton/shared_av_terminal.py','saf3r/triton/shared_av_routing.py',
        'saf3r/patch/value_clustering_gpu.py','scripts/evaluate_fused_shared_av.py']
    if args.mode=='saf3r':
        sources+=['saf3r/patch/patch_vggt.py','saf3r/patch/wrapper.py','saf3r/patch/utils.py',
            'saf3r/triton/fused_qk_topk.py','configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']
    hashes={}
    for name in sources:
        data=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    save(out/'config.json',dict(config=asdict(config),model_config=dict(model_cfg),args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    save(out/'input.json',dict(image_paths=scene.image_files,frame_indices=positions,shape=list(images.shape),checkpoint=model_cfg.ckpt_path))
    def reset_saf3r():
        if args.mode!='saf3r':return
        from saf3r.patch import update_args_per_forward
        if args.saf3r_cache_reset=='release':
            # Warmup indices are not valid evidence for a fresh input. Release
            # ALL wrapper references before recomputing, not just the first head.
            for block in model.aggregator.global_blocks:
                if hasattr(block.attn,'est_topk_idx'):block.attn.est_topk_idx=torch.empty(0)
        update_args_per_forward(model,'vggt',*images.shape[-2:])
    samples=[];cold_samples=[];fingerprints=[];raw=None
    for iteration in range(args.warmup+args.repeats):
        del raw;raw=None
        save(out/'status.json',dict(stage='forward',iteration=iteration))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        before_gib=torch.cuda.memory_allocated()/2**30
        start=time.perf_counter()
        reset_saf3r()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        sample=dict(seconds=seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,before_reset_allocated_gib=before_gib,
            after_forward_allocated_gib=torch.cuda.memory_allocated()/2**30)
        pose=raw['pose_enc'].float().cpu().numpy()
        if not np.isfinite(pose).all():raise RuntimeError('Nonfinite pose')
        if iteration>=args.warmup:
            samples.append(sample);fingerprints.append(hashlib.sha256(pose.tobytes()).hexdigest())
        else:cold_samples.append(sample)
        print('FORWARD',iteration,json.dumps(sample),flush=True)
    # Convert exactly as the official SAF3R VGGT inference wrapper does.
    ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
    pred=Dict(extrinsics=ext.squeeze().float().cpu().numpy(),intrinsics=intr.squeeze().float().cpu().numpy(),
        depth=raw['depth'].squeeze().float().cpu().numpy(),conf=raw['depth_conf'].squeeze().float().cpu().numpy(),
        processed_images=raw['images'].squeeze().permute(0,2,3,1).cpu().numpy())
    del raw
    save(out/'status.json',dict(stage='metrics'))
    if args.tasks==['pose']:
        gt=Dict(extrinsics=scene.extrinsics)
    else:
        _,gt=load_scene_data(dataset,args.dataset,args.scene,args.frames,'uniform',seed=0)
    metrics=evaluate(dataset,gt,pred,args.tasks)
    np.savez_compressed(out/'poses.npz',extrinsics=pred.extrinsics,intrinsics=pred.intrinsics)
    if args.capture and state is not None:
        for layer,data in state.captured.items():
            common=data['common'].cpu();attention=data['attention'].cpu()
            for row in attention:
                ids=torch.cat((common,row.long()))
                if ids.min()<0 or ids.max()>=state.n or len(ids.unique())!=len(ids):raise RuntimeError('Invalid support')
            labels=data['value_labels']
            if labels is not None and torch.unique(labels).numel()!=len(data['value_representatives']):raise RuntimeError('Incomplete Value coverage')
    profile=[];profile_totals={}
    if args.profile:
        if state is not None:state.profiling=True
        events=[];handles=[];starts={}
        def before(module,inputs,layer):
            a=torch.cuda.Event(enable_timing=True);a.record();starts[layer]=a
        def after(module,inputs,output,layer):
            b=torch.cuda.Event(enable_timing=True);b.record();events.append((layer,'global_module',starts[layer],b))
        for layer,block in enumerate(model.aggregator.global_blocks):
            handles.append(block.attn.register_forward_pre_hook(lambda m,i,l=layer:before(m,i,l)))
            handles.append(block.attn.register_forward_hook(lambda m,i,o,l=layer:after(m,i,o,l)))
        torch.cuda.synchronize();t0=time.perf_counter()
        reset_saf3r()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):profile_output=model(images)
        torch.cuda.synchronize();profile_seconds=time.perf_counter()-t0;del profile_output
        for handle in handles:handle.remove()
        grouped={}
        for layer,name,a,b in events+([] if state is None else state.profile_events):grouped[(layer,name)]=grouped.get((layer,name),0)+a.elapsed_time(b)/1000
        profile=[dict(layer=layer,stage=name,seconds=value) for (layer,name),value in grouped.items()]
        global_seconds=sum(r['seconds'] for r in profile if r['stage']=='global_module')
        profile_totals=dict(full_seconds=profile_seconds,global_seconds=global_seconds,other_seconds=profile_seconds-global_seconds)
    result=dict(dataset=args.dataset,scene=args.scene,mode=args.mode,frames=args.frames,metrics=metrics,
        median_seconds=float(np.median([s['seconds'] for s in samples])),median_peak_gib=float(np.median([s['peak_gib'] for s in samples])),
        samples=samples,cold_samples=cold_samples,pose_fingerprints=fingerprints,repeated_pose_identical=len(set(fingerprints))==1 if len(fingerprints)>1 else None,
        input=dict(image_paths=scene.image_files,frame_indices=positions,shape=list(images.shape),checkpoint=model_cfg.ckpt_path),
        layers=[] if state is None else state.records,config=asdict(config),source_sha256=hashes,profile=profile,profile_totals=profile_totals,
        protocol=dict(warmup=args.warmup,repeats=args.repeats,capture=args.capture,tasks=args.tasks,
            saf3r_topk=args.saf3r_topk if args.mode=='saf3r' else None,
            saf3r_cache_reset=args.saf3r_cache_reset if args.mode=='saf3r' else None),
        timing='One warmup/three samples by default; no primary layer hooks or sync; includes reset/selection/state/kernel',
        hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda))
    save(out/'result.json',result);save(out/'status.json',dict(stage='complete'))
    print('RESULT',json.dumps({k:v for k,v in result.items() if k in ['metrics','median_seconds','median_peak_gib']},default=lambda x:x.item()),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--mode',choices=['dense','fused','value','attention','saf3r'],default='fused')
    p.add_argument('--dataset',choices=['7scenes','eth3d'],default='7scenes');p.add_argument('--scene',default='chess')
    p.add_argument('--frames',type=int,default=128);p.add_argument('--retention',type=float,default=.2)
    p.add_argument('--window-groups',type=int,default=128);p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--schedule',type=Path);p.add_argument('--capture',action='store_true');p.add_argument('--profile',action='store_true')
    p.add_argument('--tasks',nargs='+',choices=['pose','depth','recon_unposed'],default=['pose'])
    p.add_argument('--saf3r-topk',type=int,default=128,help='Per-frame Top-K; previous comparison used 128, official default is 4')
    p.add_argument('--saf3r-cache-reset',choices=['release','retain'],default='release',help='Release obsolete indices before fresh selection; retain reproduces old harness')
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        import traceback
        import torch
        out=args.output_dir/args.mode/args.scene
        failure=dict(stage='oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'failed',
            error=str(error),traceback=traceback.format_exc())
        if torch.cuda.is_initialized():
            failure.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        save(out/'status.json',failure)
        raise
