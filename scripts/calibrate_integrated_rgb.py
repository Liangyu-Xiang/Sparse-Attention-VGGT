"""RGB-only independent-trajectory fidelity/timing diagnostic; never reads GT.

The JSON input is a frozen list of RGB files disjoint from the final evaluation
frame list. Dense predictions are an internal teacher, not geometric truth.
All six geometric quality fields remain unavailable for this diagnostic.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]


def save(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--method',choices=['dense','candidate'],required=True)
    parser.add_argument('--teacher',type=Path)
    parser.add_argument('--local-fidelity',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    a=parser.parse_args()
    import numpy as np
    import torch
    import yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from scripts.quality_metrics import quality_record,quality_markdown

    a.output_dir.mkdir(parents=True,exist_ok=False)
    q=quality_record({},'RGB-only numerical calibration and timing diagnostic; no pose/recon_unposed truth or geometry metrics read')
    save(a.output_dir/'quality.json',q)
    (a.output_dir/'quality.md').write_text(quality_markdown(q))
    save(a.output_dir/'status.json',dict(stage='loading'))
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=json.loads(a.input.read_text())
    paths=inputs['image_paths']
    assert not set(paths)&set(inputs['excluded_final_evaluation_image_paths'])
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    cfg.pop('patch_module');cfg.pop('sparse_config_path')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',paths).cuda()
    state=None
    if a.method=='candidate':
        from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
        state=install_reuse_residual_exploration(model,ReuseExploreConfig(**json.loads(a.config.read_text())))
    canonical=dict(image_paths=paths,shape=list(images.shape),checkpoint=str(Path(cfg.ckpt_path).resolve()),
                   checkpoint_sha256=hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest())
    save(a.output_dir/'input.json',canonical)
    samples=[];cold=[];raw=None
    for i in range(a.warmup+a.repeats):
        del raw;raw=None
        save(a.output_dir/'status.json',dict(stage='forward',iteration=i))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize()
        sample=dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        (cold if i<a.warmup else samples).append(sample)
        print('FORWARD',a.method,i,sample,flush=True)
    pose=raw['pose_enc'].float().cpu().numpy()
    # Fixed sparse spatial sampling, not a fitted alignment or quality metric.
    depth=raw['depth'].float().squeeze(0)[...,0][:,::16,::16].clamp_min(1e-8).log().cpu().numpy()
    np.savez_compressed(a.output_dir/'teacher_features.npz',pose=pose,log_depth=depth)
    result=dict(method=a.method,config=None if a.config is None else json.loads(a.config.read_text()),
                input=canonical,quality=q,timing=dict(median_seconds=float(np.median([s['seconds'] for s in samples])),
                samples=samples,cold_samples=cold,warmup=a.warmup,repeats=a.repeats,
                scope='Complete forward including reset/clustering/selection/history/all output heads'),
                fidelity=None,selection_uses_geometric_truth=False)
    if state is not None:
        def cpu(value):
            if torch.is_tensor(value):return value.detach().cpu().tolist()
            if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
            if isinstance(value,list):return [cpu(v) for v in value]
            return value
        save(a.output_dir/'layers.json',cpu(state.records))
    if a.local_fidelity:
        if state is None:raise ValueError('Local fidelity requires the candidate method')
        # An additional instrumented sparse trajectory. Dense QK is computed
        # only AFTER each mask and sparse output exist, never for selection.
        # This is distinct from a complete unchanged Dense control rerun.
        import saf3r.patch.reuse_residual_exploration as exploration
        original_prepare=exploration.prepare_qkv;current={};local=[];handles=[]
        def capture(original,x,pos):
            qq,kk,vv=original_prepare(original,x,pos)
            current[id(original)]=(qq.to(vv.dtype).contiguous(),kk.to(vv.dtype).contiguous(),vv)
            return qq,kk,vv
        exploration.prepare_qkv=capture
        for layer,block in enumerate(model.aggregator.global_blocks):
            original=block.attn.original
            def measure(module,args,layer=layer,original=original):
                qq,kk,vv=current.pop(id(original));n=qq.shape[1]
                camera=state.specials.reshape(-1,5)[:,0]
                camera=camera[torch.arange(min(16,len(camera)),device=qq.device)*len(camera)//min(16,len(camera))]
                patch=state.patches[torch.arange(min(64,len(state.patches)),device=qq.device)*len(state.patches)//min(64,len(state.patches))]
                ids=torch.cat((camera,patch))
                reference=torch.nn.functional.scaled_dot_product_attention(qq[:,ids].unsqueeze(0),kk.unsqueeze(0),vv.unsqueeze(0),dropout_p=0.,scale=original.scale)[0]
                dense=reference.transpose(0,1).reshape(len(ids),-1)
                sparse=args[0][0,ids]
                # Wo makes head errors comparable in the actual output space.
                dense=torch.nn.functional.linear(dense,original.proj.weight)
                sparse=torch.nn.functional.linear(sparse,original.proj.weight)
                errors=(dense.float()-sparse.float()).square();energy=dense.float().square()
                row=dict(layer=layer,error_sum_sq=float(errors.sum()),dense_sum_sq=float(energy.sum()),queries=len(ids))
                for name,sl in [('camera',slice(0,len(camera))),('patch',slice(len(camera),None))]:
                    row[name+'_relative_mse']=float(errors[sl].sum()/energy[sl].sum().clamp_min(1e-12))
                local.append(row)
            handles.append(original.proj.register_forward_pre_hook(measure))
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):diagnostic_output=model(images)
            torch.cuda.synchronize();del diagnostic_output
        finally:
            exploration.prepare_qkv=original_prepare
            for handle in handles:handle.remove()
        assert len(local)==24
        score=sum(r['error_sum_sq'] for r in local)/max(1e-12,sum(r['dense_sum_sq'] for r in local))
        result['fidelity']=dict(selection_score=score,layers=local,
            definition='Fixed 16 camera + 64 patch queries, Wo projected, pooled local same-QKV Dense error on separate sparse trajectory; no geometric truth, not independent Dense trajectory or quality guarantee')
    del raw,images,model,state;gc.collect();torch.cuda.empty_cache()
    save(a.output_dir/'status.json',dict(stage='timing_complete'))
    if a.teacher:
        teacher=json.loads((a.teacher/'result.json').read_text())
        if teacher['input']!=canonical:raise ValueError('RGB teacher input/checkpoint mismatch')
        reference=np.load(a.teacher/'teacher_features.npz')
        pose_error=float(np.mean((pose-reference['pose'])**2)/max(1e-12,float(np.mean(reference['pose']**2))))
        depth_error=float(np.mean((depth-reference['log_depth'])**2))
        result['fidelity']=dict(pose_relative_mse=pose_error,log_depth_mse=depth_error,
                                selection_score=pose_error+depth_error,
                                definition='unweighted sum fixed before any final geometric quality; diagnostic proxy, not a quality guarantee')
    save(a.output_dir/'result.json',result)
    print('COMPLETE',a.output_dir,flush=True)


if __name__=='__main__':main()
