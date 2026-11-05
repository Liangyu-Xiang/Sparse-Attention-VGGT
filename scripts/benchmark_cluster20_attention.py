"""Tune equivalent launch parameters on all 24 current-layer first windows, no GT."""
import argparse,json,sys,time,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
from saf3r.triton import reuse_direct_sampled_probability as kernel
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
model=build_model(cfg,'cuda');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();state=install_reuse_residual_exploration(model,ReuseExploreConfig(**json.loads(a.config.read_text())))
original=kernel.direct_sparse_attention_sampled;seen=set();rows=[]
cases=[(64,128,4,1),(64,128,4,2),(64,128,4,3),(64,128,8,2),(128,128,8,2),(64,256,4,2),(128,64,8,2)]
def call(*args,**kw):
    ref,e=original(*args,**kw);layer=state.previous_layer+1
    if layer in seen:return ref,e
    seen.add(layer)
    for bm,bk,w,st in cases:
        opts=dict(kw,block_queries=bm,block_keys=bk,num_warps=w,num_stages=st)
        row=dict(layer=layer,bm=bm,bk=bk,warps=w,stages=st)
        try:
            out,ev=original(*args,**opts);d=out.float()-ref.float();row['relative_l2']=float(d.norm()/ref.float().norm().clamp_min(1e-30));row['output_equal']=torch.equal(out,ref)
            l=ev['debiased_log_patch_probability'];r=e['debiased_log_patch_probability'];valid=torch.isfinite(l)&torch.isfinite(r)
            row['history_log_max_abs']=float((l[valid]-r[valid]).abs().max()) if bool(valid.any()) else 0.;row['history_equal']=torch.equal(l,r)
            row['same_finite_mask']=torch.equal(torch.isfinite(l),torch.isfinite(r));samples=[]
            for _ in range(3):
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();value=original(*args,**opts);end.record();end.synchronize();samples.append(begin.elapsed_time(end)/1000);del value
            row['seconds']=statistics.median(samples);row['valid']=row['relative_l2']<.005 and row['history_log_max_abs']<.05 and row['same_finite_mask']
        except Exception as exc:row.update(error=str(exc),valid=False)
        rows.append(row)
    print('LAYER',layer,flush=True)
    return ref,e
kernel.direct_sparse_attention_sampled=call
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
result=dict(rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Fixed-input Attention launch tuning; alternate outputs never feed model; no geometry',speedup_vs_dense=None,speedup_reason='Kernel diagnostic only; no Dense full model rerun')
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('DONE',flush=True)
