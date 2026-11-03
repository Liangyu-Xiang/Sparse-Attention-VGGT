"""Compare execution paths on identical real inputs before cross-layer drift."""
import argparse,json,sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration

def error(a,b):
    a=a.float();b=b.float();finite=torch.isfinite(a)&torch.isfinite(b)
    return dict(same_finite=bool(torch.equal(torch.isfinite(a),torch.isfinite(b))),max_abs=float((a[finite]-b[finite]).abs().max()) if finite.any() else 0.,relative_l2=float((a[finite]-b[finite]).norm()/a[finite].norm().clamp_min(1e-30)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();c=ReuseExploreConfig(**json.loads(a.config.read_text()));s=install_reuse_residual_exploration(model,c)
    import saf3r.triton.reuse_split_fused_history as attention
    from saf3r.triton.reuse_single_qk_grid import weighted_attention as reference_attention
    import saf3r.patch.reuse_window_graph as window
    from saf3r.patch import zdrop_history_states as old_rule
    from saf3r.patch.zdrop_history_cleanup import states,decode,diagnostics
    from saf3r.triton.reuse_drop_moment import drop_norm
    from saf3r.triton.reuse_compact import budget
    rows=[];seen=set();normal_attention=attention.weighted_attention;normal_select=window.select
    def compare_attention(*args,**kw):
        layer=s.previous_layer+1
        if ('attention',layer) in seen:return normal_attention(*args,**kw)
        seen.add(('attention',layer));out=normal_attention(*args,**kw)
        refkw={k:v for k,v in kw.items() if k not in ('stages','warps','block_keys')}
        refkw['evidence_layout']={}
        ref=reference_attention(*args,**refkw)
        rows.append(dict(layer=layer,operation='attention',output=error(ref[0],out[0]),grid_probability=error(ref[1],out[1]),probe=None if out[2] is None else error(ref[2],out[2])))
        print(json.dumps(rows[-1]),flush=True)
        return out
    def compare_select(state,layer,prior,pi,old_ids,allowed,z,reps,labels,pop,geometry):
        result=normal_select(state,layer,prior,pi,old_ids,allowed,z,reps,labels,pop,geometry)
        if ('selection',layer) in seen:return result
        seen.add(('selection',layer));geo=copy.copy(geometry)
        context=old_rule.gate(prior,pi,pop,old_ids,labels,reps);geo['drop_values']=drop_norm(prior,z,geo,len(reps),geo['drop_bins'])
        e,cost=states(pi,z,reps,labels,pop,None,allowed,prior,state.config.drop_multiplier,geo,context)
        selected,_=budget(geo['objective'],cost,e[:,:,0].amax(1)+1e-12,state.budget,state.config.dual_iterations);selected=decode(selected)
        ids,real,_=old_rule.pack(selected,None,reps,labels,pop,state.budget,context)
        weights=old_rule.transfer(selected,labels,reps,None,prior,pi,ids,geo,context)
        candidate_eager=window.eager(prior,pi,old_ids,allowed,z,reps,labels,pop,copy.copy(geometry),state.budget,state.config.dual_iterations,state.config.drop_multiplier)
        def different_set(a,b):
            aa=torch.zeros((len(prior),len(labels)+1),device=prior.device,dtype=torch.bool);bb=torch.zeros_like(aa)
            aa.scatter_(1,(a.long()+1),True);bb.scatter_(1,(b.long()+1),True)
            return int((aa[:,1:]!=bb[:,1:]).sum())
        rows.append(dict(layer=layer,operation='selection',support_different_elements=int((ids!=result[0]).sum()),support_total=ids.numel(),support_set_difference=different_set(ids,result[0]),graph_vs_current_eager_set_difference=different_set(candidate_eager[0],result[0]),graph_vs_current_eager_weight_error=error(candidate_eager[2],result[2]),weights=error(weights,result[2])))
        print(json.dumps(rows[-1]),flush=True)
        return result
    attention.weighted_attention=compare_attention;window.select=compare_select
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
    torch.cuda.synchronize()
    result=dict(complete=True,rows=rows,sampling='First window at each layer on a single real ETH3D sequence; identical local inputs, candidate full-forward trajectory',quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Fixed-input implementation diagnostic, no geometric evaluation',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
