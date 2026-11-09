"""Authorized timing-only measurement; no dataset truth or quality evaluation."""
import argparse,hashlib,json,os,statistics,sys,time,traceback,types
from pathlib import Path

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def run(a):
    sys.path.insert(0,str(a.repo))
    import saf3r
    package=types.ModuleType('saf3r.patch');package.__path__=[str(a.repo/'saf3r/patch')]
    sys.modules['saf3r.patch']=package;saf3r.patch=package
    import numpy as np,torch
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    a.output.mkdir(parents=True,exist_ok=False)
    expected=json.loads(a.expected.read_text());reference=json.loads(a.reference.read_text())
    assert expected==reference['input']
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Same CuDNN search policy for all methods; retained throughout measured calls.
    torch.backends.cudnn.benchmark=True
    cfg=Dict(name='VGGT',dpt_only=True,ckpt_path=expected['checkpoint'])
    assert hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest()==expected['checkpoint_sha256']
    model=build_model(cfg,'cuda').eval();state=None
    if a.method!='dense':
        from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
        if a.method=='candidate':
            from saf3r.patch import beta_unified_importance as importance
            from saf3r.patch.beta_fast_subgroup_forward import install
            from saf3r.patch.beta_head_batching import install as batch
            importance.ALPHA=.5;importance.TAU=.5;install();batch(model,32,validate=False)
        config=reference['config']
        state=install_reuse_residual_exploration(model,ReuseExploreConfig(**config))
    images=load_images(model,'vggt',expected['image_paths']).cuda()
    assert list(images.shape)==expected['shape'] and len(images)==128
    pixels=hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest()
    expected_pixels=reference.get('processed_image_sha256')
    if expected_pixels is None:
        # The older quick quality record did not serialize a pixel fingerprint.
        # Compare to the frozen current-candidate pixels with identical input IDs.
        pixel_record=a.pixel_reference or a.reference.parent/'processed_image.json'
        expected_pixels=json.loads(pixel_record.read_text())['sha256']
    assert pixels==expected_pixels
    save(a.output/'status.json',dict(stage='full_forward'))
    samples=[];prediction=None
    for iteration in range(4):
        del prediction;prediction=None
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(images)
        torch.cuda.synchronize()
        samples.append(dict(seconds=time.perf_counter()-start,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
        print('FORWARD',a.method,a.scene,iteration,samples[-1],flush=True)
    assert all(k in prediction for k in ['pose_enc','depth','depth_conf','world_points','world_points_conf'])
    assert torch.isfinite(prediction['pose_enc']).all() and torch.isfinite(prediction['depth']).all()
    if state is not None:
        assert len(state.records)==24
        if a.method=='quick':
            assert all(l['current_probability_probes']==4 and l['history_evidence_backend']=='none:no sparse-logit export or grid cache' for l in state.records)
        else:
            assert all(l['history_source_layer']==l['layer']-1 and l['current_fine_patch_qk_before_mask']==0 for l in state.records)
    reason='Explicitly authorized timing-only run;pose and recon_unposed not rerun; existing quality stays a separately identified reused result'
    keys=['auc03','auc30','acc','comp','cd','f1']
    quality=dict(values={k:None for k in keys},complete=False,unavailable={k:reason for k in keys},
                 units=dict(auc03='fraction',auc30='fraction',acc='m',comp='m',cd='m',f1='fraction'),
                 cd_definition='(ACC+Comp)/2 unsquared distances')
    used={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if path.is_relative_to(a.repo.resolve()) and path.is_file():used[str(path.relative_to(a.repo.resolve()))]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(method=a.method,scene=a.scene,frames=128,input=expected,processed_image_sha256=pixels,
        reference_result=str(a.reference),model_config=dict(cfg),config=None if a.method=='dense' else reference['config'],
        source_root=str(a.repo),loaded_source_sha256=used,quality=quality,quality_protocol=dict(
            tasks='Timing only;no new quality',f1_threshold_m=.05,downsample_m=.0078125,voxel_length_m=.0078125,
            sdf_trunc_m=.04,SAF3R_TopK_profile='N/A: none of these runs is SAF3R'),
        timing=dict(warmup=1,repeats=3,median_seconds=statistics.median(x['seconds'] for x in samples[1:]),
            cold_samples=samples[:1],samples=samples[1:],scope='All model heads/full forward including per-forward initialization/reset,clustering,routing,history,reordering; excludes checkpoint/model/image loading and CPU quality'),
        hardware=dict(gpu=torch.cuda.get_device_name(),physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
            torch=torch.__version__,cuda=torch.version.cuda),
        execution=dict(precision='FP32 parameters/prediction heads,BF16 backbone autocast',TF32=False,cudnn_benchmark=True,
            head_frames_chunk=32 if a.method=='candidate' else 8,all_prediction_heads_retained=True),
        speedup_vs_dense=None,speedup_reason='Filled only after matched same-scene same-GPU Dense timing completes')
    save(a.output/'result.json',result);save(a.output/'quality.json',quality)
    (a.output/'quality.txt').write_text('AUC@3=N/A; AUC@30=N/A; ACC=N/A; Comp=N/A; CD=N/A; F1=N/A\n'+reason+'\n')
    save(a.output/'status.json',dict(stage='timing_complete',complete_timing=True,complete_quality=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['dense','quick','candidate'],required=True)
    for name in ['repo','expected','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pixel-reference',type=Path)
    p.add_argument('--scene',required=True);a=p.parse_args()
    try:run(a)
    except BaseException as exc:
        reason=f'Timing run failed: {type(exc).__name__}: {exc}'
        save(a.output/'failure.json',dict(reason=reason,traceback=traceback.format_exc()))
        save(a.output/'quality.json',dict(values={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},complete=False,reason=reason))
        raise
