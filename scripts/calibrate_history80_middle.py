"""Common-Dense-input middle-layer parameter calibration, no task truth."""
import argparse
from dataclasses import asdict,replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))
from scripts.evaluate_full_global_output_comparison import save,snapshot,history80_config
from scripts.quality_metrics import quality_record
from scripts.evaluate_attention_first_residual_value import write_quality


def candidates():
    # One shared pair for all middle layers; no per-sequence/test-truth fitting.
    return [dict(name=f'a{a:g}_h{h:g}',alpha=a,history=h) for a in (0.,.2,.4,.6,.8,1.)
        for h in ((.0,) if a==0 else (.0,.4,.8))]


def config_for(candidate):
    base=history80_config()
    alpha=list(base.attention_ratios);incoming=[.8]*24
    alpha[10:17]=[candidate['alpha']]*7;incoming[10:17]=[candidate['history']]*7
    return replace(base,population_compensation=True,attention_ratios=tuple(alpha),
        incoming_history_fractions=tuple(incoming))


class StopAtMiddleEnd(Exception):pass


class Teacher:
    def __init__(self,model,images,groups):
        import torch
        from saf3r.patch.value_region_av import ValueRegionAVState
        from saf3r.patch.fused_shared_av import patch_budgets
        self.specs=candidates();self.configs=[config_for(p) for p in self.specs]
        self.state=ValueRegionAVState(self.configs[0])
        self.state.reset(len(images),images.shape[-2]//14,images.shape[-1]//14,images.device)
        complete=self.state.n//32
        if groups%2 or groups<2 or groups>complete:raise ValueError('Even sampled group count required')
        self.groups=groups
        gids=torch.linspace(0,complete-1,groups,device=images.device).long()
        self.query_ids=(gids[:,None]*32+torch.arange(32,device=images.device)).flatten()
        self.histories=[None]*len(self.specs);self.layers=[];self.handles=[]
        self.budgets=[patch_budgets(self.state.n,len(self.state.specials),c) for c in self.configs]
        for layer,block in enumerate(model.aggregator.global_blocks):
            def before(module,inputs,kwargs,l=layer):
                if l==17:raise StopAtMiddleEnd()
                self.step(module,inputs[0],kwargs.get('pos',None),l)
            self.handles.append(block.attn.register_forward_pre_hook(before,with_kwargs=True))

    def step(self,o,x,pos,layer):
        import torch
        from torch.nn import functional as F
        from saf3r.patch.region_history_attention import prepare_qkv
        from saf3r.patch.value_region_av import select_value_regions,ragged_history,residual_population_log_weights
        from saf3r.triton.fused_shared_av import shared_av
        s=self.state
        if layer!=len(self.layers):raise ValueError('Layer order')
        q,k,v=prepare_qkv(o,x,pos);q,k=q.to(v.dtype).contiguous(),k.to(v.dtype).contiguous()
        sampled=q[:,self.query_ids].contiguous()
        dense=F.scaled_dot_product_attention(sampled[None],k[None],v[None],scale=o.scale)[0]
        target=F.linear(dense.transpose(0,1).reshape(1,len(self.query_ids),-1),o.proj.weight,o.proj.bias)[0].float()
        with torch.autocast('cuda',enabled=False):
            features=v.mean(0,dtype=torch.float32)[s.patches];s.value.update(features,layer)
        if layer:s.refresh_hierarchy(k,s.value.reps)
        cache={};bank_cache={};results=[]
        for i,(total,kas,kvs) in enumerate(self.budgets):
            kv,ka=kvs[layer],kas[layer]
            outgoing=min(total[layer],math.floor(self.configs[i].incoming_history_fractions[layer+1]*kas[layer+1]))
            signature=(kv,id(self.histories[i]),outgoing)
            if signature in cache:
                metrics,history=cache[signature];self.histories[i]=history;results.append(metrics);continue
            if kv not in bank_cache:
                with torch.autocast('cuda',enabled=False):ids,merged=s.value.representatives(features,kv)
                common=torch.cat((s.patches[ids],s.specials))
                is_value=torch.zeros(len(s.patches),device=x.device,dtype=torch.bool);is_value[ids]=True
                pop=torch.bincount(merged,minlength=kv) if kv else torch.empty(0,device=x.device)
                bank_cache[kv]=(ids,merged,common,is_value,pop,k[:,common].contiguous(),v[:,common].contiguous())
            ids,merged,common,is_value,pop,pk,pv=bank_cache[kv]
            if ka:
                s.value_fine_counts=torch.bincount(s.value.labels[ids],minlength=s.padded_fine)
                attention,counts,_,_=select_value_regions(sampled,s,is_value,self.histories[i],ids,ka,0,self.groups,o.scale)
            else:
                attention=torch.empty((self.groups,0),device=x.device,dtype=torch.int32)
                counts=torch.zeros(self.groups,device=x.device,dtype=torch.long)
            actual=s.patches[attention.long().clamp_min(0)].masked_fill(attention<0,-1).int()
            bias=residual_population_log_weights(pop,merged if kv else s.value.labels,attention,len(s.specials))
            av,e,lse=shared_av(sampled,k,v,pk,pv,actual,kv,scale=o.scale,evidence=bool(outgoing),common_log_weights=bias)
            output=F.linear(av.transpose(0,1).reshape(1,len(self.query_ids),-1),o.proj.weight,o.proj.bias)[0].float()
            metrics={}
            for split,parity in (('train',0),('validation',1)):
                a=output.reshape(self.groups,32,-1)[parity::2].reshape(-1,output.shape[-1])
                b=target.reshape(self.groups,32,-1)[parity::2].reshape_as(a)
                metrics[split]=dict(relative_l2=float((a-b).norm()/b.norm()),
                    cosine=float(F.cosine_similarity(a,b,dim=-1).mean()))
            metrics['density']=float((len(common)+counts.float().mean())/s.n)
            history=None
            if outgoing:
                h=ragged_history(e,lse,common,actual,counts,kv,outgoing).long()
                history=(h-(h//(s.height*s.width+5)+1)*5).int()
            self.histories[i]=history;cache[signature]=(metrics,history);results.append(metrics)
        self.layers.append(dict(layer=layer,candidates=results))
        for name in ('pooled_parent','pooled_child','fine_keys','member_bank'):
            if hasattr(s,name):setattr(s,name,None)
        print('TEACHER',layer,flush=True)


def run(args):
    import numpy as np
    import torch
    import yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    if args.output_dir.exists():raise ValueError('Fresh calibration output required')
    args.output_dir.mkdir(parents=True);snapshot(args.output_dir)
    for name in ('scripts/calibrate_history80_middle.py','configs/profile/profile_saf3r_vggt.yaml'):
        dest=args.output_dir/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    reason='Dense-teacher output calibration only; no pose/reconstruction or paired warmed timing'
    for p in candidates():
        write_quality(args.output_dir/'candidates'/p['name'],quality_record({},reason))
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=MV_REGISTRY.get('eth3d')()
    indices,scene=sample_frames(dataset.get_data(args.scene),args.scene,args.frames,sample_mode='uniform',seed=0)
    cfg=Dict(yaml.safe_load((ROOT/'configs/profile/profile_saf3r_vggt.yaml').read_text())['model'])
    cfg.pop('patch_module',None);cfg.pop('profile_config_path',None);cfg.pop('profile_save_path',None)
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    teacher=Teacher(model,images,args.groups);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model.aggregator(images[None])
    except StopAtMiddleEnd:pass
    if len(teacher.layers)!=17:raise ValueError('Incomplete Dense teacher')
    torch.cuda.synchronize()
    save(args.output_dir/'result.json',dict(scene=args.scene,calibration=True,task_truth_used=False,
        sampling='16 deterministic evenly distributed full32-Query groups; even/odd train/validation',
        not_independent_eth3d_test='Profile/calibration scenes overlap ETH3D evaluation;7Scenes never used for selection',
        image_paths=scene.image_files,frame_indices=indices,shape=list(images.shape),checkpoint=dict(cfg),
        query_ids=teacher.query_ids.cpu().tolist(),candidates=teacher.specs,layers=teacher.layers,
        diagnostic_seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        quality=quality_record({},reason)))


def select(args):
    paths=sorted(args.output_dir.glob('*/result.json'))
    if len(paths)!=3:raise ValueError('Exactly3 predeclared calibration sequences required')
    results=[json.loads(p.read_text()) for p in paths];rows=[]
    for i,c in enumerate(candidates()):
        row=dict(c)
        for split in ('train','validation'):
            for field in ('relative_l2','cosine'):
                values=[l['candidates'][i][split][field] for r in results for l in r['layers'] if 10<=l['layer']<=16]
                row[f'{split}_{field}']=sum(values)/len(values)
        rows.append(row)
    baseline=next(r for r in rows if r['name']=='a0.8_h0.8')
    eligible=[r for r in rows if r['train_relative_l2']<=baseline['train_relative_l2'] and r['validation_cosine']>=baseline['validation_cosine']]
    best=min(eligible,key=lambda r:(r['validation_relative_l2'],-r['validation_cosine'],r['history']))
    save(args.output_dir/'selection.json',dict(best=best,baseline=baseline,
        rule='Minimum validation relative L2 across3 scenes x7 middle layers; train L2 and validation cosine no worse than baseline',
        seven_scenes_truth_used=False,candidates=sorted(rows,key=lambda r:r['validation_relative_l2']),
        calibration_inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
    save(args.output_dir/'selected_config.json',asdict(config_for(best)))
    print(json.dumps(dict(best=best,baseline=baseline),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--scene')
    parser.add_argument('--frames',type=int,default=16);parser.add_argument('--groups',type=int,default=16)
    parser.add_argument('--select',action='store_true')
    args=parser.parse_args();select(args) if args.select else run(args)
