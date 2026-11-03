"""Changed method vs its immediate reference; Dense explicitly not rerun."""
import argparse,json,statistics,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();c=ReuseExploreConfig(**json.loads(a.config.read_text()));s=install_reuse_residual_exploration(model,c)
    import saf3r.triton.reuse_split_fused_history as backend
    from saf3r.triton.reuse_split_late_v import weighted_attention as alternative
    normal=backend.weighted_attention;rows=[];seen=set()
    def hooked(q,k,v,ids,bias,special,scale,**kw):
        layer=s.previous_layer+1
        initial=normal(q,k,v,ids,bias,special,scale,**kw)
        if layer in [0,12,23] and layer not in seen:
            seen.add(layer);samples={}
            for name,fn in [('base',normal),('late_v',alternative)]:
                values=[];meta={};values_kw={**kw,'block_keys':64,'stages':1,'warps':4,'evidence_layout':None,'static_probes':False,'kernel_meta':meta}
                for i in range(5):
                    x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True);x.record();out=fn(q,k,v,ids,bias,special,scale,**values_kw);y.record();y.synchronize()
                    if i>=2:values.append(x.elapsed_time(y))
                error=float((initial[0].float()-out[0].float()).norm()/initial[0].float().norm().clamp_min(1e-30));assert error<.005
                samples[name]=dict(median_ms=statistics.median(values),relative_l2=error,output_bitwise_equal=bool(torch.equal(initial[0],out[0])),mass_bitwise_equal=bool(torch.equal(initial[1],out[1])),resource=meta)
                del out
            row=dict(layer=layer,groups=len(ids),samples=samples);rows.append(row);print(json.dumps(row),flush=True)
        return initial
    backend.weighted_attention=hooked
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize()
    result=dict(complete=True,rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Real-window key-tile timing diagnostic; geometry evaluation not performed',dense_speedup=None,dense_reason='User explicitly requested no Dense rerun; historical Dense comparison not contemporaneous')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
