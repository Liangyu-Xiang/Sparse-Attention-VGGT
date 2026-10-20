"""Offline Dense-teacher search of sequence-specific A/V schedules.

Teacher-forced sampled Attention AV error is a search surrogate, not final pose
quality and not a deployable online gate. No test GT is used to rank ratios.
"""
import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))


def policies():
    return [dict(name=f'a{s:g}_{m:g}_{d:g}',stages=[s,m,d],
        attention_ratios=[0.]+[s]*9+[m]*7+[d]*7)
        for s,m,d in product([0.,.2,.5,.8,1.],[.5,.8,1.],[0.,.2,.5,.8,1.])]


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n');temporary.replace(path)


def rank_policies(candidates,layers):
    if len(layers)!=24:raise ValueError('A full24-layer teacher pass is required')
    rows=[]
    for index,policy in enumerate(candidates):
        # Layer0 is identical in every policy, so exclude its constant error.
        train=sum(r['train_nmse'][index] for r in layers[1:])/23
        validation=sum(r['validation_nmse'][index] for r in layers[1:])/23
        rows.append(dict(**policy,train_nmse=train,validation_nmse=validation,
            actual_density=sum(r['density'][index] for r in layers)/24))
    return sorted(rows,key=lambda r:(r['train_nmse'],r['name']))


class DenseRatioTeacher:
    def __init__(self,candidates=None,query_groups=16,capture=False):
        if query_groups<2 or query_groups%2:raise ValueError('Use an even number of Query groups >=2')
        self.candidates=policies() if candidates is None else candidates
        self.groups=query_groups;self.capture=capture;self.layers=[];self.handles=[]

    def reset(self,frames,height,width,device):
        import torch
        from saf3r.patch.value_region_av import ValueRegionAVConfig,ValueRegionAVState
        from saf3r.patch.fused_shared_av import patch_budgets
        self.config=ValueRegionAVConfig(parent_only=True,history_fraction=.8)
        self.state=ValueRegionAVState(self.config);self.state.reset(frames,height,width,device)
        complete_groups=self.state.n//32
        if self.groups>complete_groups:raise ValueError('Not enough complete32-Query groups')
        self.group_ids=torch.linspace(0,complete_groups-1,self.groups,device=device).long()
        self.query_ids=(self.group_ids[:,None]*32+torch.arange(32,device=device)).flatten()
        self.budgets=[patch_budgets(self.state.n,len(self.state.specials),
            ValueRegionAVConfig(parent_only=True,history_fraction=.8,attention_ratios=p['attention_ratios']))
            for p in self.candidates]
        self.histories=[None]*len(self.candidates);self.layers=[];self.last_outputs=[]

    def attach(self,model,images):
        self.reset(images.shape[0],images.shape[-2]//14,images.shape[-1]//14,images.device)
        for layer,block in enumerate(model.aggregator.global_blocks):
            def before(module,inputs,kwargs,layer=layer):
                self.step(module,inputs[0],kwargs.get('pos',inputs[1] if len(inputs)>1 else None),layer)
            self.handles.append(block.attn.register_forward_pre_hook(before,with_kwargs=True))

    def detach(self):
        for h in self.handles:h.remove()
        self.handles=[]

    def step(self,original,x,pos,layer):
        import torch
        from torch.nn import functional as F
        from saf3r.patch.region_history_attention import prepare_qkv
        from saf3r.patch.value_region_av import select_value_regions,ragged_history
        from saf3r.patch.fused_shared_av import reference_attention
        if layer!=len(self.layers):raise ValueError('Invalid teacher layer order')
        s=self.state
        q,k,v=prepare_qkv(original,x,pos)
        q,k=q.to(v.dtype).contiguous(),k.to(v.dtype).contiguous()
        sampled=q[:,self.query_ids].contiguous()
        # Full key bank: exact Dense attention at sampled real Query rows.
        target=F.scaled_dot_product_attention(sampled[None],k[None],v[None],
            dropout_p=0.,scale=original.scale)[0]
        energy=target.float().reshape(len(q),self.groups,32,-1).square().sum((0,2,3))
        with torch.autocast(x.device.type,enabled=False):
            features=v.mean(0,dtype=torch.float32)[s.patches]
            s.value.update(features,layer)
        if layer:s.refresh_hierarchy(k,s.value.reps)
        cache={};errors=[];densities=[];self.last_outputs=[]
        for index,(total,ka_all,kv_all) in enumerate(self.budgets):
            kv,ka=kv_all[layer],ka_all[layer]
            if kv not in cache:
                with torch.autocast(x.device.type,enabled=False):
                    value_ids,_=s.value.representatives(features,kv)
                common=torch.cat((s.patches[value_ids],s.specials))
                is_value=torch.zeros(len(s.patches),device=x.device,dtype=torch.bool)
                is_value[value_ids]=True
                fine_counts=torch.bincount(s.value.labels[value_ids],minlength=s.padded_fine)
                cache[kv]=(value_ids,common,is_value,fine_counts,
                    k[:,common].contiguous(),v[:,common].contiguous())
            value_ids,common,is_value,fine_counts,pk,pv=cache[kv]
            if ka:
                s.value_fine_counts=fine_counts
                attention,counts,_,_=select_value_regions(sampled,s,is_value,self.histories[index],
                    value_ids,ka,0,self.groups,original.scale)
            else:
                attention=torch.empty((self.groups,0),dtype=torch.int32,device=x.device)
                counts=torch.zeros(self.groups,dtype=torch.long,device=x.device)
            actual=s.patches[attention.long().clamp_min(0)].masked_fill(attention<0,-1).int()
            outgoing=min(total[layer],math.floor(.8*ka_all[layer+1])) if layer<23 else 0
            if x.is_cuda:
                from saf3r.triton.fused_shared_av import shared_av
                output,e,lse=shared_av(sampled,k,v,pk,pv,actual,kv,
                    scale=original.scale,evidence=bool(outgoing))
            else:
                output,e,lse=reference_attention(sampled,k,v,common,actual,kv,original.scale)
            error=(output.float()-target.float()).reshape(len(q),self.groups,32,-1).square().sum((0,2,3))
            errors.append(torch.stack((error[::2].sum()/energy[::2].sum().clamp_min(1e-12),
                error[1::2].sum()/energy[1::2].sum().clamp_min(1e-12))))
            densities.append((counts.float().mean()+len(common))/s.n)
            if outgoing:
                ids=ragged_history(e,lse,common,actual,counts,kv,outgoing).long()
                self.histories[index]=(ids-(ids//(s.height*s.width+5)+1)*5).int()
            else:self.histories[index]=None
            if self.capture:self.last_outputs.append(output.clone())
        error=torch.stack(errors).detach().cpu();density=torch.stack(densities).detach().cpu()
        self.layers.append(dict(layer=layer,train_nmse=error[:,0].tolist(),
            validation_nmse=error[:,1].tolist(),density=density.tolist()))
        for name in ('pooled_parent','pooled_child','fine_keys','member_bank'):
            if hasattr(s,name):setattr(s,name,None)
        print('TEACHER_LAYER',layer,'best_sample_nmse',float(error[:,0].min()),flush=True)


def run(args):
    import numpy as np
    import torch
    import open3d as o3d
    from addict import Dict
    from saf3r.patch.value_region_av import ValueRegionAVConfig
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from evaluate_da3bench import load_scene_data,evaluate
    from scripts.quality_metrics import quality_record,quality_markdown
    root=args.output_dir
    if (root/'teacher_result.json').exists():raise ValueError('Never overwrite teacher results')
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0)
    dataset=MV_REGISTRY.get('7scenes')();original=dataset.get_data(args.scene)
    positions,scene=sample_frames(original,args.scene,args.frames,sample_mode='uniform',seed=0)
    checkpoint=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(Dict(name='VGGT',dpt_only=True,ckpt_path=checkpoint),'cuda')
    images=load_images(model,'vggt',scene.image_files).cuda()
    inputs=dict(image_paths=scene.image_files,frame_indices=positions,shape=list(images.shape),checkpoint=checkpoint)
    save(root/'input.json',inputs)
    samples=[];raw=None
    for repeat in range(4):
        del raw;raw=None
        save(root/'status.json',dict(stage='dense_baseline',iteration=repeat))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
        torch.cuda.synchronize()
        sample=dict(seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        if repeat:samples.append(sample)
        print('DENSE_FORWARD',repeat,sample,flush=True)
    ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
    pred=Dict(extrinsics=ext.squeeze().float().cpu().numpy(),intrinsics=intr.squeeze().float().cpu().numpy(),
        depth=raw['depth'].squeeze().float().cpu().numpy(),conf=raw['depth_conf'].squeeze().float().cpu().numpy(),
        processed_images=raw['images'].squeeze().permute(0,2,3,1).cpu().numpy())
    del raw
    _,gt=load_scene_data(dataset,'7scenes',args.scene,args.frames,'uniform',seed=0)
    metrics=evaluate(dataset,gt,pred,['pose','depth','recon_unposed']);del pred,gt
    dense=dict(dataset='7scenes',scene=args.scene,frames=args.frames,mode='dense',input=inputs,metrics=metrics,
        median_seconds=float(np.median([r['seconds'] for r in samples])),
        median_peak_gib=float(np.median([r['peak_gib'] for r in samples])),samples=samples,
        protocol=dict(warmup=1,repeats=3,tasks=['pose','depth','recon_unposed']),
        hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        quality=quality_record(metrics))
    save(root/'dense_result.json',dense);(root/'dense_quality.md').write_text(quality_markdown(dense['quality']))
    teacher=DenseRatioTeacher(query_groups=args.query_groups)
    teacher.attach(model,images)
    save(root/'status.json',dict(stage='teacher_search',candidates=len(teacher.candidates)))
    torch.cuda.synchronize();start=time.perf_counter()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):diagnostic=model(images)
        torch.cuda.synchronize();del diagnostic
    finally:teacher.detach()
    ranked=rank_policies(teacher.candidates,teacher.layers)
    result=dict(scene=args.scene,frames=args.frames,input=inputs,candidates=ranked,layers=teacher.layers,
        query_group_ids=teacher.group_ids.cpu().tolist(),query_groups=args.query_groups,
        criterion='Mean per-layer sampled pre-projection AV relative squared error vs Dense; layers1..23',
        train_groups='even sample slots',validation_groups='odd sample slots; not used for ranking',
        teacher_forced=True,ground_truth_used_for_selection=False,sequence_specific_oracle=True,
        globally_optimal=False,diagnostic_seconds=time.perf_counter()-start,
        deployment_speed_includes_search=False,quality=quality_record({},'Teacher selector diagnostic; see end-to-end candidate evaluations'))
    save(root/'teacher_result.json',result)
    (root/'teacher_quality.md').write_text(quality_markdown(result['quality']))
    selected=[]
    for row in ranked[:args.top]:
        config=asdict(ValueRegionAVConfig(parent_only=True,history_fraction=.8,attention_ratios=row['attention_ratios']))
        schedule=root/'schedules'/f"{row['name']}.json";save(schedule,config)
        selected.append(dict(name=row['name'],schedule=str(schedule),stages=row['stages'],train_nmse=row['train_nmse']))
    baseline=next(r for r in ranked if r['stages']==[.2,.8,.2])
    save(root/'selected.json',dict(selected=selected,manual_baseline=baseline,
        selection='top Dense-teacher train NMSE only; no GT ranking',sequence_specific=True))
    save(root/'status.json',dict(stage='teacher_complete',best=ranked[0]['stages']))
    print('TEACHER_BEST',ranked[0],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--scene',required=True)
    p.add_argument('--frames',type=int,default=128);p.add_argument('--query-groups',type=int,default=16)
    p.add_argument('--top',type=int,default=2);a=p.parse_args()
    try:run(a)
    except Exception as error:
        from scripts.quality_metrics import quality_record,quality_markdown
        save(a.output_dir/'status.json',dict(stage='failed',error=str(error)))
        q=quality_record({},'Teacher search failed: '+str(error));save(a.output_dir/'quality.json',q)
        (a.output_dir/'quality.md').write_text(quality_markdown(q))
        raise
