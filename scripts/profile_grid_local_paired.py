"""Paired full-forward timing and disjoint old/new scopes on one GPU."""
import argparse,csv,hashlib,json,statistics,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation'),str(ROOT/'scripts')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_residual_exploration as impl
from grid_local_scopes import Recorder


def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--rollout-only',action='store_true');args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    save(out/'status.json',dict(stage='loading'));torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');inputs=json.loads(args.input.read_text());images=load_images(model,'vggt',inputs['image_paths']).cuda()
    config=impl.ReuseExploreConfig(**json.loads((ROOT/'configs/evaluation/reuse_grid_local_history.json').read_text()))
    state=impl.install_reuse_residual_exploration(model,config)
    methods={'reference':replace(config,history_evidence_backend='reference'), 'local_packed':config,
             'local_fp16':replace(config,history_cache_format='fp16')}
    if args.rollout_only:methods={'reference':methods['reference'],'reference_repeat':methods['reference'],'local_packed':config}
    samples={m:[] for m in methods}
    import saf3r.triton.reuse_single_qk_grid as attention
    normal_attention=attention.weighted_attention;collect=None;captures={}
    def captured_attention(q,k,v,ids,*pos,**kw):
        result=normal_attention(q,k,v,ids,*pos,**kw)
        if collect is not None:
            layer=state.previous_layer+1
            record=captures[collect].setdefault(layer,dict(fingerprint=[],counts=[],output=None))
            token=ids.long()+1
            record['fingerprint'].append(((token*1000003)^(token<<17)^(token*token*9176)).sum(-1))
            record['counts'].append((ids>=0).sum(-1))
            if record['output'] is None:record['output']=result[0][:,:32].clone()
        return result
    attention.weighted_attention=captured_attention
    def run(m):
        state.config=methods[m];state.ready_cache_plain=state.config.history_cache_format=='fp16'
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize();seconds=time.perf_counter()-begin
        assert len(state.records)==24 and torch.isfinite(raw['depth']).all() and torch.isfinite(raw['pose_enc']).all()
        return dict(seconds=seconds,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    save(out/'status.json',dict(stage='forward'))
    for m in methods:
        collect=m;captures[m]={};run(m);collect=None
    comparisons={}
    for m in [n for n in methods if n!='reference']:
        comparisons[m]=[]
        for layer in range(24):
            x=captures['reference'][layer];y=captures[m][layer]
            a=torch.cat(x['fingerprint']);b=torch.cat(y['fingerprint']);ac=torch.cat(x['counts']);bc=torch.cat(y['counts'])
            before=x['output'].float();after=y['output'].float()
            comparisons[m].append(dict(layer=layer,groups=len(a),support_fingerprint_changed_groups=int((a!=b).sum()),count_changed_groups=int((ac!=bc).sum()),sample_attention_relative_l2=float((before-after).norm()/before.norm().clamp_min(1e-30)),max_patch_keys=int(bc.max())))
    save(out/'rollout_comparison.json',dict(comparisons=comparisons,scope='First warmup of each method; single mixed64-bit integer support fingerprint per group; first32 query attention outputs per layer. Fingerprint agreement is not a formal collision-free mask proof. Not timed.'))
    del captures
    if args.rollout_only:
        save(out/'summary.json',dict(complete=True,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Rollout stability diagnostic only',dense_speedup=None,dense_reason='No end-to-end speed benchmark'))
        save(out/'status.json',dict(stage='diagnostic_complete'));return
    for repeat in range(3):
        for m in (list(methods) if repeat%2==0 else list(reversed(methods))):
            samples[m].append(run(m));save(out/'samples.json',samples)
    rec=Recorder(state);rec.install()
    for block in model.aggregator.global_blocks:
        rec.wrap(block.attn.original.proj,'forward','output_projection');rec.wrap(block.attn.original.proj_drop,'forward','output_dropout')
    profiles={}
    for m in methods:
        rec.rows=[];rec.stack=[];rec.ident=0
        measured=run(m);rows,groups,total,unique=rec.finish()
        profiles[m]=dict(wall_seconds=measured['seconds'],global_attention_seconds=total,exclusive_sum_seconds=unique,sum_error_seconds=unique-total,groups=groups,events=rows)
        save(out/(m+'_profile.json'),profiles[m])
    summary=dict(complete=True,samples=samples,median_seconds={m:statistics.median(x['seconds'] for x in v) for m,v in samples.items()},profiles={m:{k:v for k,v in p.items() if k!='events'} for m,p in profiles.items()},quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Paired old/new timing only, no new geometry quality',dense_speedup=None,dense_reason='Dense not rerun; old history method is the paired reference, not Dense')
    save(out/'summary.json',summary);save(out/'status.json',dict(stage='timing_complete'));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
