"""Matched Dense/SAF3R/History80/Reuse+Exploration quality and timing runner."""
import argparse
from dataclasses import asdict,replace,is_dataclass
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import io,tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.evaluate_full_global_output_comparison import history80_config
from scripts.quality_metrics import quality_record,quality_markdown

def config_record(config):
    return asdict(config) if is_dataclass(config) else config


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
    dataset_name=args.dataset
    dataset=MV_REGISTRY.get(dataset_name)()
    positions,scene=sample_frames(dataset.get_data(args.scene),args.scene,args.frames,sample_mode='uniform',seed=0)
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    cfg.sparse_config_path=str(ROOT/cfg.sparse_config_path)
    saf3r_settings=dict(profile=cfg.sparse_config_path,topk=cfg.patch_module.topk,
        topk_source='lazy first-correspondence head, NOT paper DINO feature Top-K',
        lazy_dino_topk=cfg.patch_module.lazy_dino_topk)
    native_methods=('saf3r','saf3r_grouped')
    if args.method not in native_methods:cfg.pop('patch_module');cfg.pop('sparse_config_path')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    if args.selector_execution=='eager':
        import saf3r.patch.reuse_window_graph as window_graph
        def eager_select(state,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=None):
            geometry['p0_loss_pack']=getattr(state.config,'p0_loss_pack',False)
            return window_graph.eager(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,
                state.budget,state.config.dual_iterations,state.config.drop_multiplier,uniform=uniform)
        window_graph.select=eager_select
    state=None;config=None
    if args.method=='zdrop_history_states':
        from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
        config=ReuseExploreConfig(**json.loads(args.config.read_text()))
        config=replace(config,capture=args.capture)
        state=install_reuse_residual_exploration(model,config)
    elif args.method=='history80':
        from saf3r.patch.value_region_av import install_value_region_av
        config=replace(history80_config(),population_compensation=True,capture=args.capture)
        state=install_value_region_av(model,config)
    elif args.method in native_methods:
        from saf3r.patch import update_args_per_forward
        if saf3r_settings['topk']!=4 or not saf3r_settings['lazy_dino_topk']:raise ValueError('Frozen SAF3R configuration changed')
        if args.method=='saf3r_grouped':
            from saf3r.patch.grouped_saf3r import install_grouped_saf3r
            config=json.loads(args.config.read_text())
            if config.get('engine')!='same-pattern-head-batching' or config.get('group_q_probe_topk') is not False:raise ValueError('Explicit frozen grouped SAF3R configuration required')
            install_grouped_saf3r(model,config['geometry_cache_capacity'],config['release_cuda_cache_each_block'])
            saf3r_settings['execution']='Batch identical static head patterns; original scalar query-probe and lazy correspondence heads'
        update_args_per_forward(model,'vggt',*images.shape[-2:])
    canonical=dict(image_paths=scene.image_files,frame_indices=positions,shape=list(images.shape),checkpoint=str(Path(cfg.ckpt_path).resolve()),checkpoint_sha256=hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest())
    save(out/'input.json',canonical)
    save(out/'config.json',dict(method=args.method,config=config_record(config),
        model=dict(cfg),saf3r=saf3r_settings if args.method in native_methods else None,seed=0))
    sources=json.loads((ROOT/'baseline_source_hashes.json').read_text())
    names=set(sources)|{'saf3r/patch/reuse_inference_ops.py','saf3r/triton/reuse_inference_ops.py','saf3r/patch/reuse_window_graph.py','saf3r/triton/reuse_fused_states.py','saf3r/patch/reuse_grid_fused_history.py','saf3r/triton/reuse_grid_fused_normalizer.py','saf3r/triton/reuse_split_fused_history.py','saf3r/triton/reuse_split_grid_attention.py','saf3r/patch/reuse_probe_cache.py','configs/evaluation/reuse_efficiency_v6.json','saf3r/patch/reuse_grid_local_history.py','saf3r/triton/reuse_grid_local_history.py','saf3r/patch/reuse_headmean_history.py','saf3r/triton/reuse_headmean_history.py','saf3r/triton/reuse_single_qk_grid.py','configs/evaluation/reuse_grid_local_history.json','saf3r/patch/reuse_headmean_history.py','saf3r/triton/reuse_headmean_history.py','saf3r/triton/reuse_single_qk_grid.py','saf3r/patch/reuse_ready_history.py','saf3r/triton/reuse_ready_history.py','saf3r/triton/reuse_explore_av_ready.py','saf3r/patch/zdrop_history_cleanup.py','saf3r/triton/zdrop_history_cleanup.py','saf3r/patch/zdrop_history_states.py','saf3r/triton/zdrop_history_states.py','scripts/evaluate_zdrop_history_states.py','configs/evaluation/reuse_zdrop_history_states.json'}
    names.update({'saf3r/patch/source_local_unanchored.py','saf3r/patch/adaptive_value_clustering.py','saf3r/patch/reuse_sampled_history.py','saf3r/triton/reuse_sampled_history.py','saf3r/triton/reuse_direct_sampled_probability.py','saf3r/patch/reuse_probability_history.py','saf3r/triton/reuse_probability_history.py','saf3r/patch/reuse_direct_coarse_observation.py','saf3r/triton/reuse_direct_sparse_attention.py','saf3r/triton/reuse_direct_legacy_observation.py','saf3r/patch/reuse_direct_history.py','saf3r/triton/reuse_torch_flash.py','saf3r/triton/reuse_p0_states.py','saf3r/triton/reuse_p0_pack.py','saf3r/triton/reuse_shared_value_states.py','scripts/validate_shared_value_loss.py',str(args.config.resolve().relative_to(ROOT.resolve()))})
    names={str(p.relative_to(ROOT)) for folder in ('saf3r','evaluation') for p in (ROOT/folder).rglob('*.py')}
    names.update({'scripts/evaluate_zdrop_history_states.py','scripts/quality_metrics.py','scripts/calibrate_integrated_rgb.py',str(args.config.resolve().relative_to(ROOT.resolve())),str(Path(cfg.get('sparse_config_path',ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json')).relative_to(ROOT))})
    hashes={};buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for name in sorted(names):
            data=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    (out/'source_snapshot.tar').write_bytes(buffer.getvalue())
    save(out/'source_sha256.json',hashes)
    samples=[];cold=[];raw=None
    for iteration in range(args.warmup+args.repeats):
        del raw;raw=None
        save(out/'status.json',dict(stage='forward',iteration=iteration,method=args.method))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        if args.method in native_methods:
            # Reset INSIDE every complete-forward timer. Warmup must not cache
            # correspondence Top-K indices for the following timed inputs.
            update_args_per_forward(model,'vggt',*images.shape[-2:])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize()
        sample=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        if state is not None:
            sample['graph_counters']={k:getattr(state,k,None) for k in ('graph_cache_creations','graph_cache_hits','graph_cache_misses','graph_cache_evictions','graph_constant_updates')}
        (cold if iteration<args.warmup else samples).append(sample)
        print('FORWARD',args.scene,args.method,iteration,sample,flush=True)
    if not torch.isfinite(raw['pose_enc']).all() or not torch.isfinite(raw['depth']).all():raise ValueError('Nonfinite model output')
    if args.method=='saf3r_grouped':
        save(out/'grouped_layers.json',[dict(layer=i,**block.attn.grouped_execution_record) for i,block in enumerate(model.aggregator.global_blocks)])
    statistics_diagnostic=None
    if state is not None and hasattr(config,'collect_statistics') and not config.collect_statistics:
        # Dense and SAF3R do not assemble this large per-layer research report
        # during production inference either. Keep every numerical validation
        # in the timed forwards, and collect the report in a separate forward.
        def prediction_hashes():
            return {key:hashlib.sha256(raw[key].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                    for key in ('pose_enc','depth','depth_conf')}
        prediction_before=prediction_hashes()
        save(out/'status.json',dict(stage='statistics_diagnostic',method=args.method,scene=args.scene))
        state.config=replace(config,collect_statistics=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();statistics_start=time.perf_counter()
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):statistics_raw=model(images)
            torch.cuda.synchronize()
            statistics_diagnostic=dict(seconds=time.perf_counter()-statistics_start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                scope='Separate report-only forward, not included in production timing or quality prediction; all timed-forward numerical/support/mass validations remain enabled',
                timed_prediction_sha256=prediction_before,
                timed_prediction_unchanged=prediction_hashes()==prediction_before,
                quality=quality_record({},'Report-only forward; geometric metrics evaluated from the retained last timed production prediction instead'))
            assert statistics_diagnostic['timed_prediction_unchanged']
            save(out/'statistics_diagnostic.json',statistics_diagnostic)
            (out/'statistics_quality.md').write_text(quality_markdown(statistics_diagnostic['quality']))
            del statistics_raw
        finally:state.config=config
    attention_profiles=[]
    if getattr(args,'attention_timing',0):
        from contextlib import contextmanager
        normal_records=state.records
        events=[]
        @contextmanager
        def attention_scope(name):
            first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
            first.record()
            try:yield
            finally:last.record();events.append((name,first,last))
        state.direct_scope=attention_scope
        for profile_index in range(args.attention_timing):
            events.clear();torch.cuda.synchronize();begin=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):profile_output=model(images)
            torch.cuda.synchronize();wall=time.perf_counter()-begin
            stages={}
            for name,first,last in events:stages[name]=stages.get(name,0.)+first.elapsed_time(last)/1000
            attention_profiles.append(dict(wall_seconds=wall,stages_seconds=stages,
                logical_qk_per_head=sum(r['direct_logical_main_qk_per_head'] for r in state.records),
                mean_key_density=sum(float(r['actual_unique_density']) for r in state.records)/len(state.records),
                graph_creations=state.graph_cache_creations,
                compiler_resources=state.sampled_compiler_resources))
            del profile_output
        del state.direct_scope
        state.records=normal_records
        save(out/'attention_profile.json',dict(scope='Separate lightly-instrumented complete forwards; normal full-forward timings and final normal-output quality remain separate',profiles=attention_profiles))
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
        warmup=args.warmup,repeats=args.repeats,capture=args.capture,selector_execution=args.selector_execution,
        saf3r_topk_reset_each_forward=args.method in native_methods,
        scope='Complete forward, includes all reset/clustering/pooling/selection/history/repeated evidence QK; no per-layer sync')
    common=dict(method=args.method,scene=args.scene,dataset=dataset_name,frames=len(images),input=canonical,
        config=config_record(config),model_config=dict(cfg),saf3r=saf3r_settings if args.method in native_methods else None,
        source_sha256=hashes,timing=timing,attention_profiles=attention_profiles,statistics_diagnostic=statistics_diagnostic,
        execution_policy=dict(release_cuda_cache_each_block=getattr(model.aggregator,'release_cuda_cache_each_block',True),same_pattern_head_batching=args.method=='saf3r_grouped'),
        hardware=dict(gpu=torch.cuda.get_device_name(),gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),torch=torch.__version__,cuda=torch.version.cuda,cpu_affinity=sorted(__import__('os').sched_getaffinity(0))),
        quality_protocol=dict(tasks=['pose','recon_unposed'],sampling='uniform',seed=0,
            f1_threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample,
            cd_definition='(ACC+Comp)/2 unsquared metres',reconstruction='unposed'))
    common['quality_protocol'].update(voxel_length_m=dataset.voxel_length,sdf_trunc_m=dataset.sdf_trunc,mesh_sample_points=dataset.sampling_number,max_depth=dataset.max_depth,alignment='Repository unposed Umeyama/RANSAC scale+pose alignment; random_state42',depth_resize='Nearest neighbour to original RGB resolution; intrinsics scaled consistently',masking='Existing dataset _prep_unposed and GT-validity masks; no new filtering or threshold tuning')
    common['environment']=dict(python=sys.version,open3d=o3d.__version__)
    save(out/'forward.json',common)
    if args.diagnostic or args.timing_only:
        reason='Explicit warmed full-forward timing-only run; geometric quality not rerun' if args.timing_only else 'Explicit implementation diagnostic; pose/recon_unposed not evaluated'
        q=quality_record({},reason)
        if args.method=='dense':q.update(speedup_vs_vggt=1.,speedup_unavailable=None)
        write_quality(out,q);save(out/('timing_result.json' if args.timing_only else 'diagnostic.json'),dict(common,quality=q))
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
        _,gt=load_scene_data(dataset,dataset_name,args.scene,args.frames,'uniform',seed=0)
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
    elif dataset_name!='7scenes':
        q.update(speedup_unavailable='Speed evaluation requested only for7Scenes; no paired Dense ETH3D timing')
    result=dict(common,quality=q,metrics=metrics,quality_errors=errors)
    save(out/'result.json',result);write_quality(out,q)
    save(out/'status.json',dict(stage='complete' if q['complete'] else 'quality_incomplete'))
    print('QUALITY',args.scene,args.method,q,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['eth3d','7scenes'],default='eth3d')
    p.add_argument('--method',choices=['dense','saf3r','saf3r_grouped','zdrop_history_states'],required=True)
    p.add_argument('--scene',required=True);p.add_argument('--frames',type=int,default=128)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--baseline',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/reuse_residual_exploration.json')
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--attention-timing',type=int,default=0)
    p.add_argument('--selector-execution',choices=['graph','eager'],default='graph')
    p.add_argument('--capture',action='store_true');p.add_argument('--diagnostic',action='store_true')
    p.add_argument('--timing-only',action='store_true')
    a=p.parse_args()
    if a.warmup<0 or a.repeats<1:raise ValueError('Invalid timing protocol')
    try:run(a)
    except Exception as exc:
        if a.output_dir.exists() and not (a.output_dir/'result.json').exists():
            write_quality(a.output_dir,quality_record({},f'{type(exc).__name__}: {exc}'))
            save(a.output_dir/'status.json',dict(stage='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        raise
