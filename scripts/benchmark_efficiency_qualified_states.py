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
    import saf3r.triton.reuse_fused_states as backend
    from saf3r.triton.reuse_qualified_states import states as qualified
    normal=backend.states;rows=[];seen=set()
    def hooked(prior,z,reps,labels,population,allowed,geometry,context):
        layer=s.previous_layer+1
        initial=normal(prior,z,reps,labels,population,allowed,geometry,context)
        if layer in [1,12,23] and layer not in seen:
            seen.add(layer);samples={}
            for name,fn in [('baseline',normal),('qualified_loads',qualified)]:
                values=[]
                for i in range(7):
                    x=torch.cuda.Event(enable_timing=True);y=torch.cuda.Event(enable_timing=True);x.record();out=fn(prior,z,reps,labels,population,allowed,geometry,context);y.record();y.synchronize()
                    if i>=2:values.append(x.elapsed_time(y))
                assert torch.equal(initial[0],out[0]) and torch.equal(initial[1],out[1])
                samples[name]=dict(median_ms=statistics.median(values),bitwise_equal=True)
            row=dict(layer=layer,groups=len(prior),samples=samples);rows.append(row);print(json.dumps(row),flush=True)
        return initial
    # Benchmark outside graph capture; keep current values and current eligibility.
    import saf3r.patch.reuse_window_graph as window
    def select(state,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry):
        backend.states=hooked
        return window.eager(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,state.budget,state.config.dual_iterations,state.config.drop_multiplier)
    window.select=select
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize()
    result=dict(complete=True,rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Real-window qualified-state load timing diagnostic; geometry evaluation not performed',dense_speedup=None,dense_reason='User explicitly requested no Dense rerun; historical Dense comparison not contemporaneous')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
