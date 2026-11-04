"""Frozen-input teacher replay of QK-free loss definitions.

Real QK is used only AFTER a candidate support is selected to measure error.
The executed baseline supplies frozen QKV/clusters/history for all candidates.
No geometry ground truth is read for choosing losses.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch
from scripts import diagnose_shared_value_accuracy as diagnostic
from saf3r.patch.state_loss_reference import output_sketch,losses,uniform_unobserved,transport_losses,group_history_losses
from saf3r.patch import reuse_inference_ops as rule
from saf3r.triton.reuse_compact import budget


class LossAudit(diagnostic.Audit):
    def prepare(self,original,module,x,pos):
        self.sketch=None;self.candidate_ids={};self.loss_rows=[] if not hasattr(self,'loss_rows') else self.loss_rows
        return super().prepare(original,module,x,pos)

    def prior(self,original,state,start,old_ids,obs,return_uniform=False):
        self.observation=obs
        return super().prior(original,state,start,old_ids,obs,return_uniform)

    @torch.autocast('cuda',enabled=False)
    def select(self,original,s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=None):
        result=original(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=uniform)
        target=[(group,group-self.start) for group in self.target_groups if 0<=group-self.start<len(prior)]
        if not target:return result
        if self.sketch is None:self.sketch=output_sketch(self.v,self.module,s.patches)
        indices=torch.tensor([local for _,local in target],device=prior.device)
        p=prior[indices];mass=pi[indices];old=old_ids[indices];permit=allowed[indices]
        context=rule.gate(p,mass,population,old,labels,reps)
        context['remove_rep']=torch.zeros_like(context['remove_rep'])
        obs=self.observation
        sp=(obs['special_logits'][indices,:,:1,:]-obs['coarse_logz'][indices,:,:1,:]).exp().mean((1,2))
        for policy in ['unit','rep_population']:
            definitions,costs,_=losses(p,mass,self.sketch[s.patches],self.sketch[s.specials],sp,labels,reps,population,
                context['qualified'],permit,policy)
            for name,errors in definitions.items():
                selected,_=budget(errors,costs,errors[:,:,0].amax(1)+1e-12,s.budget,16)
                selected=rule.decode(selected)
                ids,_,record=rule.pack(selected,None,reps,labels,population,s.budget,context)
                valid=ids>=0;safe=ids.long().clamp_min(0)
                weights=torch.ones_like(ids,dtype=torch.float32)
                if policy=='rep_population':weights=torch.where(selected.gather(1,labels[safe])==1,population[labels[safe]],1.)
                observed=torch.zeros_like(p,dtype=torch.int32).scatter_add_(1,old.long().clamp_min(0),(old>=0).int())>0
                reused=(observed.gather(1,safe)&valid).sum(1)
                for index,(group,_) in enumerate(target):
                    stats=dict(method=policy+'/'+name,layer=layer,group=group,
                        selected_keys=int(record['counts'][index]),
                        history_key_reuse_fraction=float(reused[index]/record['counts'][index].clamp_min(1)),
                        state_fractions={str(k):float((selected[index]==k).float().mean()) for k in [0,1,5,10]},
                        density=float((record['counts'][index]+len(s.specials))/s.n))
                    self.candidate_ids.setdefault(group,{})[policy+'/'+name]=(ids[index],weights[index],stats)
        sampled=torch.tensor([group%8==0 for group,_ in target],device=p.device)
        up=uniform_unobserved(p,mass,old,labels,population,sampled)
        observed=torch.zeros_like(up,dtype=torch.int32).scatter_add_(1,old.long().clamp_min(0),(old>=0).int())>0
        for gate,qualified in [('native',context['qualified']),('observed',observed)]:
            definitions,costs,other,hmass=transport_losses(up,mass,self.sketch[s.patches],self.sketch[s.specials],sp,
                labels,reps,population,qualified,permit)
            union=other.clone();union[:,reps]=True
            transport_context=dict(context,qualified=union)
            transport_context['eligible_count']=torch.zeros_like(mass,dtype=torch.int32).scatter_add_(1,labels[None].expand_as(up),union.int())
            for name,errors in definitions.items():
                selected,_=budget(errors,costs,errors[:,:,0].amax(1)+1e-12,s.budget,16)
                selected=rule.decode(selected)
                ids,_,record=rule.pack(selected,None,reps,labels,population,s.budget,transport_context)
                good=ids>=0;safe=ids.long().clamp_min(0);cluster=labels[safe]
                is_rep=safe==reps[cluster]
                own=up.gather(1,safe)
                assigned=mass.gather(1,cluster)-torch.where(selected.gather(1,cluster)==5,hmass.gather(1,cluster),0.)
                weights=torch.where(is_rep&(selected.gather(1,cluster)!=10),assigned/own.clamp_min(1e-30),1.)
                reused=(observed.gather(1,safe)&good).sum(1)
                for index,(group,_) in enumerate(target):
                    stats=dict(method='residual_rep_uniform_'+gate+'/'+name,layer=layer,group=group,
                        selected_keys=int(record['counts'][index]),history_key_reuse_fraction=float(reused[index]/record['counts'][index].clamp_min(1)),
                        state_fractions={str(k):float((selected[index]==k).float().mean()) for k in [0,1,5,10]},
                        density=float((record['counts'][index]+len(s.specials))/s.n),history_state='History plus residual Rep; ordinary History weights1')
                    self.candidate_ids.setdefault(group,{})[stats['method']]=(ids[index],weights[index],stats)
        for gate,qualified in [('native',context['qualified']),('observed',observed)]:
            definitions,costs,hmass=group_history_losses(up,mass,self.sketch[s.patches],self.sketch[s.specials],sp,
                labels,reps,population,qualified,permit)
            group_context=dict(context,qualified=qualified)
            group_context['eligible_count']=torch.zeros_like(mass,dtype=torch.int32).scatter_add_(1,labels[None].expand_as(up),qualified.int())
            for name,errors in definitions.items():
                selected,_=budget(errors,costs,errors[:,:,0].amax(1)+1e-12,s.budget,16)
                selected=rule.decode(selected)
                ids,_,record=rule.pack(selected,None,reps,labels,population,s.budget,group_context)
                good=ids>=0;safe=ids.long().clamp_min(0);cluster=labels[safe];state=selected.gather(1,cluster)
                own=up.gather(1,safe)
                weights=torch.where(state==1,mass.gather(1,cluster)/own.clamp_min(1e-30),
                    torch.where(state==5,mass.gather(1,cluster)/hmass.gather(1,cluster).clamp_min(1e-30),1.))
                reused=(observed.gather(1,safe)&good).sum(1)
                for index,(group,_) in enumerate(target):
                    stats=dict(method='group_history_uniform_'+gate+'/'+name,layer=layer,group=group,
                        selected_keys=int(record['counts'][index]),history_key_reuse_fraction=float(reused[index]/record['counts'][index].clamp_min(1)),
                        state_fractions={str(k):float((selected[index]==k).float().mean()) for k in [0,1,5,10]},
                        density=float((record['counts'][index]+len(s.specials))/s.n),history_state='History collection represents cluster; common pi/sum_selected_prior bias')
                    self.candidate_ids.setdefault(group,{})[stats['method']]=(ids[index],weights[index],stats)
        return result

    @torch.autocast('cuda',enabled=False)
    def attention(self,original,q,k,v,ids,specials,scale,bias=None,**kwargs):
        start=self.start
        result=super().attention(original,q,k,v,ids,specials,scale,bias,**kwargs)
        layer=self.state.previous_layer+1
        for group in self.target_groups:
            local=group-start
            if not 0<=local<len(ids):continue
            offsets=self.group_offsets[group]
            positions=torch.tensor([local*32+o for o in offsets],device=q.device)
            logits=(q[:,positions].float()@k.float().transpose(-1,-2))*scale
            dense=logits.softmax(-1)@v.float()
            target=dense.transpose(0,1).reshape(len(positions),-1)
            weight=self.module.proj.weight.float();b=self.module.proj.bias.float() if self.module.proj.bias is not None else None
            target=torch.nn.functional.linear(target,weight,b)
            camera=torch.tensor([group*32+o in self.camera_ids for o in offsets],device=q.device)
            if layer==0:
                proposals={}
                for policy in ['unit','rep_population']:
                    key=ids[local];good=key>=0
                    patch_index=torch.searchsorted(self.state.patches,key[good].long())
                    population=torch.bincount(self.state.fine.labels,minlength=self.state.fine.base_count).float()
                    w=torch.ones_like(key,dtype=torch.float32)
                    if policy=='rep_population':w[good]=population[self.state.fine.labels[patch_index]]
                    for name in ['self_energy','centered_energy','centered_squared_energy','centered_moment','moment_mass01','moment_mass1','conditional_mass']:
                        proposals[policy+'/'+name]=(patch_index.int(),w[good],dict(method=policy+'/'+name,layer=0,group=group,
                            selected_keys=len(patch_index),history_key_reuse_fraction=0.,state_fractions={'0':0.,'1':1.,'5':0.,'10':0.},density=.2))
                for gate in ['native','observed']:
                    for name in ['residual_energy','centered_residual_energy','transport_moment','transport_mass','transport_variance']:
                        method='residual_rep_uniform_'+gate+'/'+name
                        proposals[method]=(patch_index.int(),w[good],dict(method=method,layer=0,group=group,
                            selected_keys=len(patch_index),history_key_reuse_fraction=0.,state_fractions={'0':0.,'1':1.,'5':0.,'10':0.},density=.2))
                    for name in ['group_centroid','group_centered','group_mixture','group_moment','group_moment_variance']:
                        method='group_history_uniform_'+gate+'/'+name
                        proposals[method]=(patch_index.int(),w[good],dict(method=method,layer=0,group=group,
                            selected_keys=len(patch_index),history_key_reuse_fraction=0.,state_fractions={'0':0.,'1':1.,'5':0.,'10':0.},density=.2))
            else:proposals=self.candidate_ids[group]
            for name,(keys,weights,stats) in proposals.items():
                good=keys>=0;patch=keys[good].long()
                actual=self.state.patches[patch] if layer else self.state.patches[patch]
                support=torch.cat((actual,specials.long()))
                logw=torch.cat((weights[good].float().log(),torch.zeros(len(specials),device=q.device)))
                output=(logits[:,:,support]+logw).softmax(-1)@v[:,support].float()
                output=output.transpose(0,1).reshape(len(positions),-1)
                output=torch.nn.functional.linear(output,weight,b)
                row=dict(stats,similarity={})
                for subset,mask in [('all',torch.ones_like(camera)),('camera',camera),('probe_non_camera',~camera)]:
                    if bool(mask.any()):row['similarity'][subset]=diagnostic.vector_similarity(output[mask],target[mask])
                self.loss_rows.append(row)
        if self.start>=self.state.groups:
            diagnostic.save(self.out/'loss_candidates.json',self.loss_rows)
            print('LOSS_LAYER',layer,'candidate_rows',len(self.loss_rows),flush=True)
        self.candidate_ids={}
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene',default='kicker');parser.add_argument('--frames',type=int,default=31)
    args=parser.parse_args()
    diagnostic.Audit=LossAudit
    from saf3r.patch import reuse_window_graph as selector
    original_select=selector.select
    # The baseline diagnostic's Audit object is reached through the bound
    # prepare callback only after installation; capture it at construction.
    audit_instance=[]
    original_init=LossAudit.__init__
    def init(self,*a,**kw):original_init(self,*a,**kw);audit_instance.append(self)
    LossAudit.__init__=init
    selector.select=lambda *a,**kw:audit_instance[0].select(original_select,*a,**kw)
    cfg=json.loads((ROOT/'configs/evaluation/reuse_shared_value_loss_256.json').read_text())
    cfg.update(shared_value_weighting='rep_population',disable_population_compensation=False,shared_value_allow_observed_rep=True)
    config=ROOT/'configs/evaluation/state_loss_teacher_baseline.json'
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    opts=argparse.Namespace(dataset='eth3d',scene=args.scene,frames=args.frames,groups=12,config=config,output=args.output,
        dense_trajectory=False,representative_control=False,camera_probes=True)
    try:diagnostic.run(opts)
    finally:selector.select=original_select


if __name__=='__main__':main()
