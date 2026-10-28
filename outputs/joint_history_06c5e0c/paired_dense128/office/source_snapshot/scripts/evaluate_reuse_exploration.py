"""Matched Dense/SAF3R/History80/Reuse+Exploration quality and timing runner."""
import argparse
from dataclasses import asdict,replace
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.evaluate_full_global_output_comparison import history80_config
from scripts.quality_metrics import quality_record


def run(args):
    import torch
    import yaml
    import open3d as o3d
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from evaluate_da3bench import load_scene_data,evaluate
    out=args.output_dir
    if out.exists():raise ValueError('Fresh output directory required')
    out.mkdir(parents=True)
    write_quality(out,quality_record({},'Run in progress'))
    save(out/'status.json',dict(stage='loading',method=args.method,scene=args.scene))
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=MV_REGISTRY.get('7scenes')()
    positions,scene=sample_frames(dataset.get_data(args.scene),args.scene,args.frames,sample_mode='uniform',seed=0)
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.ckpt_path=str(getattr(args,'checkpoint',None) or ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    cfg.sparse_config_path=str(ROOT/cfg.sparse_config_path)
    saf3r_settings=dict(profile=cfg.sparse_config_path,topk=cfg.patch_module.topk,
        topk_source='lazy first-correspondence head, NOT paper DINO feature Top-K',
        lazy_dino_topk=cfg.patch_module.lazy_dino_topk)
    if args.method!='saf3r':cfg.pop('patch_module');cfg.pop('sparse_config_path')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    state=None;config=None
    if args.method=='reuse_explore':
        from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
        config=ReuseExploreConfig(**json.loads(args.config.read_text()))
        config=replace(config,capture=args.capture)
        state=install_reuse_residual_exploration(model,config)
    elif args.method=='history80':
        from saf3r.patch.value_region_av import install_value_region_av
        config=replace(history80_config(),population_compensation=True,capture=args.capture)
        state=install_value_region_av(model,config)
    elif args.method=='saf3r':
        from saf3r.patch import update_args_per_forward
        if saf3r_settings['topk']!=4 or not saf3r_settings['lazy_dino_topk']:raise ValueError('Frozen SAF3R configuration changed')
        update_args_per_forward(model,'vggt',*images.shape[-2:])
    canonical=dict(image_paths=scene.image_files,frame_indices=positions,shape=list(images.shape),checkpoint=cfg.ckpt_path)
    if getattr(args,'expected_input',None):
        expected=json.loads(args.expected_input.read_text())
        if json.loads(json.dumps(canonical,default=lambda x:x.tolist()))!=expected:
            raise ValueError('Exact historical checkpoint/frame/preprocessing input mismatch')
    save(out/'input.json',canonical)
    save(out/'config.json',dict(method=args.method,config=None if config is None else asdict(config),
        model=dict(cfg),saf3r=saf3r_settings if args.method=='saf3r' else None,seed=0))
    sources=['saf3r/patch/reuse_residual_exploration.py','saf3r/triton/reuse_explore_av.py',
        'saf3r/patch/fused_shared_av.py','saf3r/patch/value_clustering_gpu.py',
        'saf3r/patch/region_history_attention.py','saf3r/patch/value_evolving_attention.py',
        'saf3r/patch/value_region_av.py','saf3r/patch/patch_vggt.py','saf3r/triton/fused_shared_av.py',
        'saf3r/triton/value_region_selection.py','scripts/evaluate_reuse_exploration.py',
        'scripts/quality_metrics.py','scripts/unified_diagnostic_runtime.py',
        'scripts/evaluate_all_edge_history.py','configs/evaluation/reuse_residual_exploration.json',
        'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']
    hashes={}
    for name in sources:
        data=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    save(out/'source_sha256.json',hashes)
    samples=[];cold=[];raw=None
    for iteration in range(args.warmup+args.repeats):
        del raw;raw=None
        save(out/'status.json',dict(stage='forward',iteration=iteration,method=args.method))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize()
        sample=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (cold if iteration<args.warmup else samples).append(sample)
        print('FORWARD',args.scene,args.method,iteration,sample,flush=True)
    if not torch.isfinite(raw['pose_enc']).all() or not torch.isfinite(raw['depth']).all():raise ValueError('Nonfinite model output')
    if state is not None:
        if len(state.records)!=24:raise ValueError('Missing sparse layers')
        save(out/'layers.json',state.records)
        if args.capture:
            def cpu(item):
                if torch.is_tensor(item):return item.cpu()
                if isinstance(item,dict):return {k:cpu(v) for k,v in item.items()}
                if isinstance(item,list):return [cpu(v) for v in item]
                return item
            torch.save(cpu(state.captured),out/'supports.pt')
    timing=dict(median_seconds=float(np.median([s['seconds'] for s in samples])),samples=samples,cold_samples=cold,
        warmup=args.warmup,repeats=args.repeats,capture=args.capture,
        scope='Complete forward, includes reset/clustering/pooling/selection/history and 4/32-query probe evidence; no per-layer sync')
    common=dict(method=args.method,scene=args.scene,dataset='7scenes',frames=len(images),input=canonical,
        config=None if config is None else asdict(config),model_config=dict(cfg),saf3r=saf3r_settings if args.method=='saf3r' else None,
        source_sha256=hashes,timing=timing,
        hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
            device_uuid=os.environ.get('CUDA_VISIBLE_DEVICES')),
        quality_protocol=dict(tasks=['pose','recon_unposed'],sampling='uniform',seed=0,
            f1_threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample,
            cd_definition='(ACC+Comp)/2 unsquared metres',reconstruction='unposed'))
    save(out/'forward.json',common)
    if args.diagnostic:
        q=quality_record({},'Forward-only diagnostic/timing run: pose and recon_unposed not evaluated')
        write_quality(out,q);save(out/'diagnostic.json',dict(common,quality=q))
        save(out/'status.json',dict(stage='diagnostic_complete'));return
    ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
    pred=Dict(extrinsics=ext.squeeze().float().cpu().numpy(),intrinsics=intr.squeeze().float().cpu().numpy(),
        depth=raw['depth'].squeeze().float().cpu().numpy(),conf=raw['depth_conf'].squeeze().float().cpu().numpy(),
        processed_images=raw['images'].squeeze().permute(0,2,3,1).cpu().numpy())
    del raw,images,model,state;gc.collect();torch.cuda.empty_cache()
    # This signal is emitted only after freeing model/GPU, allowing next paired
    # method to overlap with CPU reconstruction evaluation on this same GPU.
    save(out/'status.json',dict(stage='quality',method=args.method))
    metrics={};errors={}
    try:
        _,gt=load_scene_data(dataset,'7scenes',args.scene,args.frames,'uniform',seed=0)
        for task in ('pose','recon_unposed'):
            try:
                np.random.seed(0);o3d.utility.random.seed(0)
                metrics.update(evaluate(dataset,gt,pred,[task]))
            except Exception as exc:errors[task]=f'{type(exc).__name__}: {exc}'
    except Exception as exc:errors['ground_truth']=f'{type(exc).__name__}: {exc}'
    q=quality_record(metrics,'; '.join(f'{k}: {v}' for k,v in errors.items()) or 'Task unavailable')
    if args.method=='dense':q.update(speedup_vs_vggt=1.,speedup_unavailable=None)
    elif args.baseline:
        baseline=json.loads(args.baseline.read_text())
        for key in ('input','hardware','quality_protocol'):
            if baseline[key]!=common[key]:raise ValueError('Paired Dense mismatch: '+key)
        bt=baseline['timing']
        if any(bt[key]!=timing[key] for key in ('warmup','repeats','capture')) or baseline['method']!='dense':
            raise ValueError('Paired Dense timing mismatch')
        q.update(speedup_vs_vggt=bt['median_seconds']/timing['median_seconds'],speedup_unavailable=None)
    result=dict(common,quality=q,metrics=metrics,quality_errors=errors)
    save(out/'result.json',result);write_quality(out,q)
    save(out/'status.json',dict(stage='complete' if q['complete'] else 'quality_incomplete'))
    print('QUALITY',args.scene,args.method,q,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['dense','saf3r','history80','reuse_explore'],required=True)
    p.add_argument('--scene',required=True);p.add_argument('--frames',type=int,default=128)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--baseline',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--expected-input',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/reuse_residual_exploration.json')
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--capture',action='store_true');p.add_argument('--diagnostic',action='store_true')
    a=p.parse_args()
    if a.warmup<0 or a.repeats<1:raise ValueError('Invalid timing protocol')
    try:run(a)
    except Exception as exc:
        if a.output_dir.exists() and not (a.output_dir/'result.json').exists():
            write_quality(a.output_dir,quality_record({},f'{type(exc).__name__}: {exc}'))
            save(a.output_dir/'status.json',dict(stage='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        raise
