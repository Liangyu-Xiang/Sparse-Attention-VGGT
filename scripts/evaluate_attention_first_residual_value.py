"""Independent Attention-first / residual-Value evaluation, or paired VGGT Dense."""
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
from scripts.quality_metrics import quality_record, quality_cells, LABELS


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')


def matched_speedup(result,baseline):
    for key in ('dataset','scene','frames','input','hardware','protocol'):
        if result[key]!=baseline[key]:raise ValueError(f'Dense comparison mismatch: {key}')
    if baseline['mode']!='dense':raise ValueError('Baseline must be original VGGT Dense')
    return baseline['median_seconds']/result['median_seconds']


def write_quality(out,quality):
    save(out/'quality.json',quality)
    speed=quality.get('speedup_vs_vggt')
    text='| '+' | '.join((*LABELS,'Speedup vs VGGT ↑'))+' |\n'
    text+='| '+' | '.join(['---:']*7)+' |\n'
    text+='| '+' | '.join(quality_cells(quality)+[f'{speed:.3f}x' if speed is not None else 'N/A'])+' |\n'
    if quality['unavailable']:text+='\nUnavailable: '+json.dumps(quality['unavailable'])+'\n'
    if speed is None:text+='\nSpeed unavailable: '+quality['speedup_unavailable']+'\n'
    (out/'quality.md').write_text(text)


def run(args):
    import torch
    import open3d as o3d
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from saf3r.patch.attention_first_residual_value import AttentionFirstConfig,install_attention_first_residual_value
    from saf3r.patch.value_region_av import ValueRegionAVConfig,install_value_region_av
    from saf3r.patch.av_stage_ablation import apply_stage_ablation,STAGES
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from evaluate_da3bench import load_scene_data,evaluate
    if args.warmup<0 or args.repeats<1:raise ValueError('Invalid timing protocol')
    out=args.output_dir/args.mode/args.scene
    if (out/'result.json').exists():raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0)
    dataset=MV_REGISTRY.get(args.dataset)()
    positions,scene=sample_frames(dataset.get_data(args.scene),args.scene,args.frames,sample_mode='uniform',seed=0)
    model_cfg=Dict(name='VGGT',dpt_only=True,ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'))
    model=build_model(model_cfg,'cuda')
    images=load_images(model,'vggt',scene.image_files).cuda()
    settings=json.loads(args.config.read_text()) if args.config else {}
    settings['capture']=args.capture
    if args.mode=='old':
        cfg=ValueRegionAVConfig(**(dict(parent_only=True,history_fraction=.8,capture=args.capture)|settings))
        state=install_value_region_av(model,cfg)
    else:
        cfg=AttentionFirstConfig(**settings)
        state=install_attention_first_residual_value(model,cfg) if args.mode=='attention_first' else None
    if args.active_stage!='all':
        if state is None:raise ValueError('Stage ablation requires a sparse scheme')
        apply_stage_ablation(model,args.active_stage)
    input_info=dict(image_paths=scene.image_files,frame_indices=positions,shape=list(images.shape),checkpoint=model_cfg.ckpt_path)
    hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    protocol=dict(warmup=args.warmup,repeats=args.repeats,capture=args.capture,tasks=args.tasks)
    save(out/'input.json',input_info);save(out/'config.json',asdict(cfg))
    sources=['scripts/evaluate_attention_first_residual_value.py','scripts/quality_metrics.py',
        'saf3r/patch/attention_first_residual_value.py','saf3r/triton/attention_first_selection.py',
        'saf3r/patch/fused_shared_av.py','saf3r/patch/value_clustering_gpu.py',
        'saf3r/triton/fused_shared_av.py','saf3r/triton/shared_av_routing.py',
        'saf3r/patch/av_stage_ablation.py','saf3r/patch/region_history_attention.py',
        'saf3r/patch/value_evolving_attention.py','saf3r/patch/value_region_av.py',
        'saf3r/patch/parent_region_selection.py','saf3r/triton/value_region_selection.py']
    hashes={}
    for source in sources:
        data=(ROOT/source).read_bytes();hashes[source]=hashlib.sha256(data).hexdigest()
        dest=out/'source_snapshot'/source;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    samples=[];cold=[];raw=None;fingerprints=[]
    for iteration in range(args.warmup+args.repeats):
        del raw;raw=None
        save(out/'status.json',dict(stage='forward',iteration=iteration))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize()
        sample=dict(seconds=time.perf_counter()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        pose=raw['pose_enc'].float().cpu().numpy()
        if not np.isfinite(pose).all():raise RuntimeError('Nonfinite output pose')
        (samples if iteration>=args.warmup else cold).append(sample)
        if iteration>=args.warmup:fingerprints.append(hashlib.sha256(pose.tobytes()).hexdigest())
        print('FORWARD',iteration,json.dumps(sample),flush=True)
    layers=[] if state is None else state.records
    # Capture is opt-in for small diagnostic inputs, not performance comparisons.
    if args.capture and state is not None:
        def cpu(value):
            if torch.is_tensor(value):return value.cpu()
            if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu(v) for v in value]
            return value
        torch.save(cpu(state.captured),out/'supports.pt')
    ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
    pred=Dict(extrinsics=ext.squeeze().float().cpu().numpy(),intrinsics=intr.squeeze().float().cpu().numpy(),
        depth=raw['depth'].squeeze().float().cpu().numpy(),conf=raw['depth_conf'].squeeze().float().cpu().numpy(),
        processed_images=raw['images'].squeeze().permute(0,2,3,1).cpu().numpy())
    del raw
    save(out/'status.json',dict(stage='metrics'))
    if args.tasks==['pose']:gt=Dict(extrinsics=scene.extrinsics)
    else:_,gt=load_scene_data(dataset,args.dataset,args.scene,args.frames,'uniform',seed=0)
    metrics=evaluate(dataset,gt,pred,args.tasks)
    result=dict(mode=args.mode,dataset=args.dataset,scene=args.scene,frames=args.frames,input=input_info,
        hardware=hardware,protocol=protocol,config=asdict(cfg),metrics=metrics,
        median_seconds=float(np.median([r['seconds'] for r in samples])),
        median_peak_gib=float(np.median([r['peak_gib'] for r in samples])),samples=samples,cold_samples=cold,
        layers=layers,source_sha256=hashes,pose_fingerprints=fingerprints,
        active_stage=args.active_stage,active_sparse_layers=list(STAGES[args.active_stage]),
        ablation_state='All layers maintain sparse state; inactive layers return original Dense output; auxiliary routing and duplicate QKV are included in timing' if args.active_stage!='all' else None,
        quality_protocol=dict(reconstruction='recon_unposed',f1_threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample),
        timing='Complete model forward including reset, gate synchronization, clustering, selection, history, final attention; no per-layer timing synchronization')
    quality=quality_record(metrics,missing_reason='Task explicitly omitted from this diagnostic run')
    if args.mode=='dense':quality.update(speedup_vs_vggt=1.,speedup_unavailable=None)
    elif args.baseline_result:
        speed=matched_speedup(result,json.loads(args.baseline_result.read_text()))
        quality.update(speedup_vs_vggt=speed,speedup_unavailable=None)
    result['quality']=quality
    np.savez_compressed(out/'poses.npz',extrinsics=pred.extrinsics,intrinsics=pred.intrinsics)
    write_quality(out,quality);save(out/'result.json',result);save(out/'status.json',dict(stage='complete'))
    print('RESULT',json.dumps(dict(quality=quality,seconds=result['median_seconds'],peak_gib=result['median_peak_gib'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--mode',choices=['dense','old','attention_first'],default='attention_first')
    p.add_argument('--active-stage',choices=['all','shallow','middle','deep'],default='all')
    p.add_argument('--dataset',choices=['7scenes','eth3d'],default='7scenes');p.add_argument('--scene',default='chess')
    p.add_argument('--frames',type=int,default=128);p.add_argument('--config',type=Path)
    p.add_argument('--baseline-result',type=Path);p.add_argument('--capture',action='store_true')
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--tasks',nargs='+',choices=['pose','depth','recon_unposed'],default=['pose','depth','recon_unposed'])
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        out=args.output_dir/args.mode/args.scene
        # Do not overwrite an earlier successful run on an accidental rerun.
        if not (out/'result.json').exists():
            out.mkdir(parents=True,exist_ok=True)
            write_quality(out,quality_record({},f'{type(exc).__name__}: {exc}'))
            save(out/'status.json',dict(stage='failed',error=f'{type(exc).__name__}: {exc}'))
        raise
