"""Diagnostic joint-output loss refinement on frozen features; no extra selection QK.

This CPU-orchestrated coordinate search is explicitly not a production method.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts import search_shared_state_losses as search
from saf3r.patch.state_loss_reference import output_sketch,uniform_unobserved,group_history_losses
from saf3r.patch.joint_state_loss_reference import refine
from saf3r.patch import reuse_inference_ops as rule
from saf3r.triton.reuse_compact import budget
NAMES=['joint/control_separable','joint/gamma0','joint/gamma001','joint/gamma01']

class JointAudit(search.LossAudit):
 @torch.autocast('cuda',enabled=False)
 def select(self,original,s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=None):
  result=original(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,uniform=uniform)
  target=[(group,group-self.start) for group in self.target_groups if 0<=group-self.start<len(prior)]
  if not target:return result
  if self.sketch is None:self.sketch=output_sketch(self.v,self.module,s.patches)
  indices=torch.tensor([i for _,i in target],device=prior.device);p=prior[indices];mass=pi[indices];old=old_ids[indices];permit=allowed[indices]
  sp=(self.observation['special_logits'][indices,:,:1,:]-self.observation['coarse_logz'][indices,:,:1,:]).exp().mean((1,2))
  sampled=torch.tensor([group%8==0 for group,_ in target],device=p.device)
  up=uniform_unobserved(p,mass,old,labels,population,sampled)
  observed=torch.zeros_like(up,dtype=torch.int32).scatter_add_(1,old.long().clamp_min(0),(old>=0).int())>0
  context=rule.gate(p,mass,population,old,labels,reps);context.update(remove_rep=torch.zeros_like(mass,dtype=torch.bool),qualified=observed)
  context['eligible_count']=torch.zeros_like(mass,dtype=torch.int32).scatter_add_(1,labels[None].expand_as(up),observed.int())
  y=self.sketch[s.patches];sy=self.sketch[s.specials]
  definitions,costs,hm=group_history_losses(up,mass,y,sy,sp,labels,reps,population,observed,permit)
  errors=definitions['group_centered'];seed,_=budget(errors-errors[:,:,1:2],costs,errors[:,:,0].amax(1)+1e-12,s.budget,16)
  for name,gamma in zip(NAMES,[None,0.,.01,.1]):
   selected=seed;before=after=None
   if gamma is not None:selected,before,after=refine(seed,costs,errors,up,mass,y,sy,sp,labels,reps,observed,s.budget,gamma=gamma)
   selected=rule.decode(selected);ids,_,record=rule.pack(selected,None,reps,labels,population,s.budget,context)
   good=ids>=0;safe=ids.long().clamp_min(0);cluster=labels[safe];state=selected.gather(1,cluster);own=up.gather(1,safe)
   weights=torch.where(state==1,mass.gather(1,cluster)/own.clamp_min(1e-30),torch.where(state==5,mass.gather(1,cluster)/hm.gather(1,cluster).clamp_min(1e-30),1.))
   reused=(observed.gather(1,safe)&good).sum(1)
   for index,(group,_) in enumerate(target):
    stats=dict(method=name,layer=layer,group=group,selected_keys=int(record['counts'][index]),history_key_reuse_fraction=float(reused[index]/record['counts'][index].clamp_min(1)),
     state_fractions={str(k):float((selected[index]==k).float().mean()) for k in [0,1,5,10]},density=float((record['counts'][index]+len(s.specials))/s.n),
     predicted_objective_before=None if before is None else float(before[index]),predicted_objective_after=None if after is None else float(after[index]),
     offline_coordinate_moves=32,production_ready=False)
    self.candidate_ids.setdefault(group,{})[name]=(ids[index],weights[index],stats)
  return result

if __name__=='__main__':
 search.LossAudit=JointAudit
 search.main()
 out=Path(sys.argv[sys.argv.index('--output')+1]);path=out/'loss_candidates.json';rows=json.loads(path.read_text());seeds=[r for r in rows if r['layer']==0 and r['method']=='rep_population/self_energy']
 rows=[r for r in rows if r['layer']>0]+[dict(r,method=name) for r in seeds for name in NAMES]
 path.write_text(json.dumps(rows,indent=2)+'\n')
