"""Same-input real-window old/new producers; no QK reruns in either producer."""
import argparse,json,sys,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_residual_exploration as impl,reuse_grid_local_history as local,reuse_headmean_history as reference
from saf3r.triton.reuse_ready_history import decode


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');inputs=json.loads(args.input.read_text());images=load_images(model,'vggt',inputs['image_paths']).cuda()
    state=impl.install_reuse_residual_exploration(model,impl.ReuseExploreConfig(**json.loads((ROOT/'configs/evaluation/reuse_grid_local_history.json').read_text())))
    normal=local.save_evidence;rows=[];seen=set()
    def wrapped(s,q,k,start,ids,actual,obs,scale):
        layer=s.previous_layer+1
        if layer in [0,12] and layer not in seen:
            seen.add(layer);samples={n:[] for n in ['reference','local']}
            for i in range(7):
                for name,fn in [('reference',reference.save_evidence),('local',normal)]:
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn(s,q,k,start,ids,actual,obs,scale);b.record();b.synchronize()
                    if i>=2:samples[name].append(a.elapsed_time(b))
            reference.save_evidence(s,q,k,start,ids,actual,obs,scale);old=decode(s.ready_edges[start:start+len(ids)],ids.shape[1]).clone();u=s.ready_uniform[start:start+len(ids)].clone()
            normal(s,q,k,start,ids,actual,obs,scale);new=decode(s.ready_edges[start:start+len(ids)],ids.shape[1])
            row=dict(layer=layer,shape=list(obs['ready_logits'].shape),raw_bytes=obs['ready_logits'].numel()*4,producer_median_ms={n:statistics.median(v) for n,v in samples.items()},samples_ms=samples,cache_max_abs=float((new-old).abs().max()),cache_rel_l2=float((new-old).norm()/old.norm().clamp_min(1e-30)),uniform_max_abs=float((s.ready_uniform[start:start+len(ids)]-u).abs().max()))
            rows.append(row);(out/'partial.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
        return normal(s,q,k,start,ids,actual,obs,scale)
    local.save_evidence=wrapped
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();assert len(state.records)==24 and torch.isfinite(raw['depth']).all()
    result=dict(complete=True,rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Real-window kernel/producer diagnostic, not pose or reconstruction evaluation',dense_speedup=None,dense_reason='Not full-forward Dense comparison')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
