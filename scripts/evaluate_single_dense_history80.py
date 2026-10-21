"""Quality-only single-global-layer Dense restoration of ORIGINAL population History80."""
import argparse
from dataclasses import asdict,replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))
from scripts.evaluate_full_global_output_comparison import save,snapshot,history80_config
from scripts.quality_metrics import quality_record
from scripts.evaluate_attention_first_residual_value import write_quality


def run_name(layer):return 'sparse_baseline' if layer==-1 else f'dense_L{layer:02d}'


def run(args):
    import numpy as np
    import torch
    import yaml
    import open3d as o3d
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from saf3r.patch.value_region_av import install_value_region_av
    from saf3r.patch.single_dense_av_ablation import apply_single_dense_ablation
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from evaluate_da3bench import load_scene_data,evaluate
    out=args.output_dir/args.scene/run_name(args.dense_layer)
    if out.exists():raise ValueError('Fresh run required; no prior result overwritten')
    out.mkdir(parents=True);hashes=snapshot(out)
    for name in ('scripts/evaluate_single_dense_history80.py','saf3r/patch/single_dense_av_ablation.py',
        'configs/evaluation/parent_region_history80_population.json'):
        b=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(b).hexdigest()
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    save(out/'source_sha256.json',hashes)
    save(out/'status.json',dict(stage='loading',dense_layer=args.dense_layer))
    write_quality(out,quality_record({},'Single-layer quality ablation in progress'))
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=MV_REGISTRY.get('7scenes')()
    indices,scene=sample_frames(dataset.get_data(args.scene),args.scene,args.frames,sample_mode='uniform',seed=0)
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.pop('patch_module',None);cfg.pop('sparse_config_path',None)
    cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    config=replace(history80_config(),population_compensation=True)
    assert config.attention_ratios==(0.,)+(.2,)*9+(.8,)*7+(.2,)*7
    assert config.history_fraction==.8 and config.incoming_history_fractions is None
    state=install_value_region_av(model,config)
    wrapper=apply_single_dense_ablation(model,args.dense_layer) if args.dense_layer>=0 else None
    canonical=dict(image_paths=scene.image_files,frame_indices=indices,shape=list(images.shape),checkpoint=cfg.ckpt_path)
    baseline_input=json.loads((args.baseline/'7scenes'/args.scene/'history80/input.json').read_text())
    if canonical!=baseline_input:raise ValueError('Exact baseline input mismatch')
    save(out/'input.json',canonical)
    save(out/'config.json',dict(history80=asdict(config),dense_output_layer=args.dense_layer,
        shadow_sparse_state_preserved=True,history_source='shadow sparse compensated probabilities,NOT Dense',
        future_supports='natural recursive regeneration from actual upstream inputs;not frozen masks',
        global_layers=list(range(24)),frame_local_unchanged=True,seed=0,frames=len(images),
        warmup=0,repeats=1,speed_benchmark=False,model_config=dict(cfg)))
    save(out/'status.json',dict(stage='forward',dense_layer=args.dense_layer))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    peak=torch.cuda.max_memory_allocated()/2**30;reserved=torch.cuda.max_memory_reserved()/2**30
    if len(state.records)!=24 or state.previous_layer!=23:raise ValueError('Sparse causal state did not pass all24 layers')
    if wrapper is not None and wrapper.calls!=1:raise ValueError('Expected exactly one Dense restoration')
    if not all(bool(torch.isfinite(v).all()) for v in raw.values() if torch.is_tensor(v)):
        raise ValueError('Nonfinite final model output')
    save(out/'history80_runtime.json',state.records)
    save(out/'intervention.json',dict(dense_layer=args.dense_layer,calls=0 if wrapper is None else wrapper.calls,
        sparse_vs_dense_same_input_postprojection=None if wrapper is None else wrapper.comparison,
        state_records=24,shadow_history_preserved=True))
    common=dict(dataset='7scenes',scene=args.scene,method=run_name(args.dense_layer),frames=len(images),input=canonical,
        dense_layer=args.dense_layer,config=asdict(config),source_sha256=hashes,
        hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        diagnostic_seconds=seconds,peak_allocated_gib=peak,peak_reserved_gib=reserved,
        timing='Cold forward,shadow sparse+Dense intervention+local comparison;NOT paired warmed speed benchmark',
        quality_protocol=dict(reconstruction='recon_unposed',f1_threshold_m=dataset.eval_threshold,
            downsample_m=dataset.down_sample,cd_definition='(ACC+Comp)/2, unsquared',seed=0))
    if args.smoke:
        q=quality_record({},'Two-frame single-layer implementation diagnostic;no quality or paired warmed timing')
        save(out/'smoke.json',dict(common,quality=q,finite_outputs=True))
        write_quality(out,q);save(out/'status.json',dict(stage='smoke_complete'));return
    ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
    pred=Dict(extrinsics=ext.squeeze().float().cpu().numpy(),intrinsics=intr.squeeze().float().cpu().numpy(),
        depth=raw['depth'].squeeze().float().cpu().numpy(),conf=raw['depth_conf'].squeeze().float().cpu().numpy(),
        processed_images=raw['images'].squeeze().permute(0,2,3,1).cpu().numpy())
    pose_hash=hashlib.sha256(raw['pose_enc'].float().cpu().numpy().tobytes()).hexdigest()
    del raw,images,model,state,wrapper;gc.collect();torch.cuda.empty_cache()
    # This stage is deliberately announced only AFTER GPU allocations are freed.
    # Queue can overlap another GPU forward with this CPU-only quality evaluation.
    save(out/'status.json',dict(stage='quality',dense_layer=args.dense_layer))
    metrics={};errors={}
    try:
        o3d.utility.random.seed(0);np.random.seed(0)
        _,gt=load_scene_data(dataset,'7scenes',args.scene,args.frames,'uniform',seed=0)
        for task in ('pose','recon_unposed'):
            try:metrics.update(evaluate(dataset,gt,pred,[task]))
            except Exception as exc:errors[task]=f'{type(exc).__name__}: {exc}'
    except Exception as exc:errors['ground_truth']=f'{type(exc).__name__}: {exc}'
    q=quality_record(metrics,'; '.join(f'{k}: {v}' for k,v in errors.items()) or 'Task unavailable')
    q['speedup_unavailable']='Quality-first single-layer diagnostic;no matched warmed uninstrumented Dense timings'
    save(out/'result.json',dict(common,quality=q,metrics=metrics,quality_errors=errors,pose_hash=pose_hash))
    write_quality(out,q);save(out/'status.json',dict(stage='complete' if q['complete'] else 'quality_incomplete'))
    if not q['complete']:raise RuntimeError('Quality metrics incomplete: '+str(errors))
    print('COMPLETE',args.scene,run_name(args.dense_layer),q['values'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--scene',choices=['heads','stairs','chess'],required=True)
    p.add_argument('--dense-layer',type=int,choices=range(-1,24),required=True)
    p.add_argument('--frames',type=int,default=128);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    try:run(a)
    except Exception as exc:
        out=a.output_dir/a.scene/run_name(a.dense_layer)
        if not (out/'result.json').exists() and not (out/'smoke.json').exists():
            out.mkdir(parents=True,exist_ok=True);reason=f'{type(exc).__name__}: {exc}'
            write_quality(out,quality_record({},reason))
            save(out/'status.json',dict(stage='failed',error=reason,traceback=traceback.format_exc()))
        raise
