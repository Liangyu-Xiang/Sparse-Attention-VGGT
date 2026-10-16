"""Paired 7Scenes128 evaluation of fixed/adaptive head-aware block attention."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import csv
import hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SCENES=['chess','fire','heads','office','pumpkin','redkitchen','stairs']


def source_manifest():
    sources=['saf3r/patch/adaptive_block_value.py','saf3r/triton/adaptive_block_attention.py',
        'saf3r/patch/coarse_sensing.py','saf3r/patch/hierarchical_value_attention.py',
        'saf3r/patch/value_clustering_gpu.py','saf3r/patch/patch_vggt.py','saf3r/patch/wrapper.py',
        'scripts/benchmark_adaptive_block_value.py','configs/evaluation/eval_adaptive_block_value_vggt.yaml',
        'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}


def worker(args):
    import torch
    from addict import Dict
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.patch.adaptive_block_value import install_adaptive_block_value
    from saf3r.patch.patch_vggt import patch_model,update_args_per_forward
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sys.path.insert(0,str(ROOT/'evaluation'))
    from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous
    directory=args.output_dir/args.scene;directory.mkdir(parents=True,exist_ok=True)
    manifest=source_manifest()
    (directory/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    status=dict(state='initializing',pid=os.getpid(),scene=args.scene)
    def save():
        (directory/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        reader=SevenScenes();reader.data_root=str(ROOT/'workspace/benchmark_dataset/7scenes')
        data=reader.get_data(args.scene)
        positions=np.linspace(0,len(data.image_files)-1,args.frames,dtype=int)
        paths=[data.image_files[i] for i in positions]
        images=load_and_process_images_vggt(paths).cuda()
        model=build_model(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt','cuda')
        originals=[b.attn for b in model.aggregator.global_blocks]
        original_forward=model.aggregator.forward
        methods={}
        diagnostics={}
        for name in args.variants:
            state=None
            if name in {'fixed10','adaptive'}:
                state=install_adaptive_block_value(model,audit_guard=name=='adaptive',
                    audit_nmse=args.audit_nmse,audit_p90=args.audit_p90,audit_queries=args.audit_queries,
                    query_block=args.query_block,key_block=args.key_block,
                    validation_queries=args.validation_queries,profile_stages=args.profile_stages)
            elif name=='saf3r':
                patch_model(model,Dict(sparse_config_path=str(ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json'),
                    patch_module=dict(type='headsparse',topk_mode='token',lazy_dino_topk=True,topk=128)))
                update_args_per_forward(model,*images.shape[-2:])
            events=[]
            def before(module,inputs):
                module._start_event=torch.cuda.Event(enable_timing=True);module._start_event.record()
            def after(module,inputs,output):
                end=torch.cuda.Event(enable_timing=True);end.record();events.append((module._start_event,end))
            handles=[hook for b in model.aggregator.global_blocks for hook in (
                b.attn.register_forward_pre_hook(before),b.attn.register_forward_hook(after))]
            samples=[]
            pose_fingerprints=[]
            for iteration in range(args.warmup+args.repeats):
                status.update(state='inference',variant=name,iteration=iteration);save()
                events.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(images)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-started
                global_seconds=sum(a.elapsed_time(b) for a,b in events)/1000
                peak=torch.cuda.max_memory_allocated()/2**30
                print(args.scene,name,iteration,'seconds',elapsed,'global',global_seconds,'GiB',peak,flush=True)
                if iteration>=args.warmup:
                    samples.append(dict(seconds=elapsed,global_seconds=global_seconds,peak_gib=peak))
                pose=pred['pose_enc'].float().cpu();del pred
                if iteration>=args.warmup:
                    pose_fingerprints.append(hashlib.sha256(pose.numpy().tobytes()).hexdigest())
                if not torch.isfinite(pose).all():
                    raise ValueError('Nonfinite pose')
            ext,intr=pose_encoding_to_extri_intri(pose,images.shape[-2:])
            metrics=compute_pose(torch.from_numpy(as_homogeneous(ext[0].numpy())),torch.from_numpy(np.asarray(data.extrinsics)[positions]))
            result=dict(samples=samples,pose={k:float(v) for k,v in metrics.items()})
            result['measured_pose_fingerprints']=pose_fingerprints
            result['measured_pose_identical']=len(set(pose_fingerprints))==1
            if not result['measured_pose_identical']:
                raise ValueError(f'{name} predictions are not repeatable across measured forwards')
            result.update({'median_'+k:float(np.median([r[k] for r in samples])) for k in samples[0]})
            if state is not None:
                result['layers']=state.records
                result['mean_connection_density']=float(np.mean([r['actual_density'] for r in state.records]))
                result['dense_fallback_head_fraction']=sum(len(r['dense_fallback_heads']) for r in state.records)/(24*16)
                assert len(state.records)==24
                (directory/f'{name}_layers.json').write_text(json.dumps(state.records,indent=2)+'\n')
                if args.diagnostics_after_timing:
                    # Separate extra forward. Its validation/profiling cost is
                    # NOT mixed into formal samples, unlike the online budget
                    # probes, which remain part of every timed forward.
                    state.validation_queries=64;state.profile_stages=True
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        diagnostic_pred=model(images)
                    diagnostic_pose=diagnostic_pred['pose_enc'].float().cpu();del diagnostic_pred
                    difference=float((diagnostic_pose-pose).abs().max())
                    if difference>1e-4:
                        raise ValueError(f'Diagnostics changed pose: {difference}')
                    diagnostics[name]=dict(layers=state.records,max_absolute_pose_difference=difference,
                        note='Separate untimed diagnostic forward; independent 64-query checks and stage profiling enabled.')
                    np.savez_compressed(directory/f'{name}_diagnostic_poses.npz',pose_enc=diagnostic_pose.numpy())
                    (directory/f'{name}_diagnostics.json').write_text(json.dumps(diagnostics[name],indent=2)+'\n')
                state.reset_hook_handle.remove();delattr(model,'adaptive_block_value_state')
            methods[name]=result
            np.savez_compressed(directory/f'{name}_poses.npz',pose_enc=pose.numpy(),extrinsics=ext.numpy(),intrinsics=intr.numpy())
            (directory/f'{name}_result.json').write_text(json.dumps(result,indent=2)+'\n')
            print('METRICS',args.scene,name,result['pose'],result.get('mean_connection_density'),flush=True)
            for handle in handles:handle.remove()
            for b,original in zip(model.aggregator.global_blocks,originals):b.attn=original
            model.aggregator.forward=original_forward
            del state;torch.cuda.empty_cache()
        metadata=dict(scene=args.scene,frames=args.frames,image_paths=paths,frame_indices=positions.tolist(),
            gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda,
            checkpoint=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'),source_sha256=manifest,
            image_shape=list(images.shape),global_layers_zero_based=list(range(24)),local_attention='unchanged dense',
            query_block=args.query_block,key_block=args.key_block,audit_queries=args.audit_queries,
            audit_nmse=args.audit_nmse,audit_p90=args.audit_p90,base_retention=.10,
            independent_validation_queries=args.validation_queries,profile_stages=args.profile_stages,
            diagnostics_after_timing=args.diagnostics_after_timing,
            candidates=[.10,.20,.35,.50,.75,1.],warmup=args.warmup,repeats=args.repeats,
            note='ALL initialization, sensing, probe dense QK/output checks, head matching, selection, validation and fallback INCLUDED in timings. Block-quantized actual density reported; no fixed-10% claim for adaptive method.')
        (directory/'result.json').write_text(json.dumps(dict(input=metadata,methods=methods),indent=2)+'\n')
        if diagnostics:
            (directory/'diagnostics_result.json').write_text(json.dumps(dict(methods=diagnostics),indent=2)+'\n')
        status['state']='complete'
    except Exception as exc:
        status.update(state='failed',error=repr(exc));raise
    finally:save()


def summarize(args):
    results=[]
    for scene in SCENES:
        p=args.output_dir/scene/'result.json'
        if p.exists():results.append((scene,json.loads(p.read_text())))
    rows=[]
    for scene,result in results:
        for method,r in result['methods'].items():
            rows.append(dict(scene=scene,method=method,seconds=r['median_seconds'],global_ms=1000*r['median_global_seconds'],
                peak_gib=r['median_peak_gib'],actual_density=r.get('mean_connection_density'),
                dense_fallback_head_fraction=r.get('dense_fallback_head_fraction'),
                **{k:100*v for k,v in r['pose'].items()}))
    with (args.output_dir/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    aggregate={}
    for method in args.variants:
        rr=[r for r in rows if r['method']==method]
        aggregate[method]={k:float(np.mean([r[k] for r in rr if r[k] is not None])) if any(r[k] is not None for r in rr) else None
            for k in list(rows[0])[2:]}
    (args.output_dir/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2),flush=True)


def batch(args):
    pending=list(args.scenes);active={};failed=[];args.output_dir.mkdir(parents=True,exist_ok=True)
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:continue
            scene=pending.pop(0);directory=args.output_dir/scene;directory.mkdir(exist_ok=True)
            if (directory/'result.json').exists():
                existing=json.loads((directory/'result.json').read_text())['input']
                expected=dict(frames=args.frames,warmup=args.warmup,repeats=args.repeats,
                    audit_nmse=args.audit_nmse,audit_p90=args.audit_p90,audit_queries=args.audit_queries,
                    query_block=args.query_block,key_block=args.key_block,
                    independent_validation_queries=args.validation_queries,profile_stages=args.profile_stages)
                expected['diagnostics_after_timing']=args.diagnostics_after_timing
                if existing.get('source_sha256')!=source_manifest() or any(existing.get(k)!=v for k,v in expected.items()):
                    raise ValueError(f'Refusing stale/mismatched results at {directory}; choose a fresh output directory')
                continue
            command=[sys.executable,str(Path(__file__).resolve()),'--scene',scene,'--output-dir',str(args.output_dir),
                '--frames',str(args.frames),'--warmup',str(args.warmup),'--repeats',str(args.repeats),
                '--audit-nmse',str(args.audit_nmse),'--audit-p90',str(args.audit_p90),'--audit-queries',str(args.audit_queries),
                '--query-block',str(args.query_block),'--key-block',str(args.key_block),'--variants',*args.variants]
            command+=['--validation-queries',str(args.validation_queries)]
            if args.profile_stages:command+=['--profile-stages']
            if args.diagnostics_after_timing:command+=['--diagnostics-after-timing']
            log=(directory/'run.log').open('a')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1',
                'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=scene,p,log;print('START',scene,gpu,flush=True)
        for gpu,(scene,p,log) in list(active.items()):
            if p.poll() is None:continue
            log.close();print('END',scene,p.returncode,flush=True)
            if p.returncode:failed.append(scene)
            del active[gpu]
        if active:time.sleep(5)
    if failed:raise RuntimeError(f'Workers failed: {failed}')
    summarize(args)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=SCENES);p.add_argument('--scenes',nargs='+',default=SCENES)
    p.add_argument('--gpus',nargs='+',default=list(range(7)))
    p.add_argument('--frames',type=int,default=128);p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--audit-nmse',type=float,default=.02);p.add_argument('--audit-p90',type=float,default=.10)
    p.add_argument('--audit-queries',type=int,default=32);p.add_argument('--query-block',type=int,default=64);p.add_argument('--key-block',type=int,default=64)
    p.add_argument('--validation-queries',type=int,default=0,help='Independent diagnostic probes; zero for formal timing')
    p.add_argument('--profile-stages',action='store_true',help='Diagnostic stage events; disable for formal timing')
    p.add_argument('--diagnostics-after-timing',action='store_true',help='Extra untimed forward for independent checks and stage profiling')
    p.add_argument('--variants',nargs='+',choices=['dense','saf3r','fixed10','adaptive'],default=['dense','saf3r','fixed10','adaptive'])
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/adaptive_block_value/7scenes_n128_audit02')
    args=p.parse_args()
    worker(args) if args.scene else batch(args)
