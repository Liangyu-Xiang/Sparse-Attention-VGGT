"""Offline QK-free joint-output coordinate refinement; not a production allocator."""
import torch
from .state_loss_reference import reduce_cluster


def refine(selected,costs,errors,prior,pi,y,special_y,special_p,labels,reps,qualified,cap,gamma=0.,iterations=32):
 g,f=selected.shape;device=selected.device;d=y.shape[-1];rows=torch.arange(g,device=device);clusters=torch.arange(f,device=device)[None]
 moment=reduce_cluster(prior[:,:,None]*y[None],labels,f);hp=prior*qualified
 hm=reduce_cluster(hp,labels,f);hv=reduce_cluster(hp[:,:,None]*y[None],labels,f)/hm[:,:,None].clamp_min(1e-30)
 total=prior.sum(1)+special_p.sum(1);mu=(prior@y+special_p@special_y)/total[:,None]
 delta=torch.stack((moment-pi[:,:,None]*mu[:,None],moment-pi[:,:,None]*y[reps][None],moment-pi[:,:,None]*hv,torch.zeros_like(moment)),2)
 mass=torch.stack((torch.zeros_like(pi),pi,pi,pi),2)
 legal=torch.isfinite(errors);energy=torch.where(legal,errors,0.)
 selected=selected.long().clone()
 def current():
  vector=delta[rows[:,None],clusters,selected].sum(1)
  z=mass.gather(2,selected[:,:,None])[:,:,0].sum(1)+special_p.sum(1)
  c=costs.gather(2,selected[:,:,None])[:,:,0].sum(1)
  reg=energy.gather(2,selected[:,:,None])[:,:,0].sum(1)
  return vector,z,c,reg
 before=None
 for _ in range(iterations):
  vector,z,c,reg=current();old_delta=delta[rows[:,None],clusters,selected]
  old_mass=mass.gather(2,selected[:,:,None]);old_cost=costs.gather(2,selected[:,:,None]);old_energy=energy.gather(2,selected[:,:,None])
  alt=vector[:,None,None,:]-old_delta[:,:,None,:]+delta
  alt_z=z[:,None,None]-old_mass+mass
  alt_cost=c[:,None,None]-old_cost+costs
  score=alt.square().sum(-1)/alt_z.square().clamp_min(1e-30)+gamma*(reg[:,None,None]-old_energy+energy)/total[:,None,None].square()
  score=score.masked_fill(~legal|(alt_cost>cap)|(alt_z<=1e-12),float('inf'))
  current_score=vector.square().sum(1)/z.square()+gamma*reg/total.square()
  if before is None:before=current_score.clone()
  best=score.flatten(1).argmin(1);cluster=best//4;state=best%4;improve=score.flatten(1).gather(1,best[:,None])[:,0]<current_score-1e-12
  selected[rows,cluster]=torch.where(improve,state,selected[rows,cluster])
 vector,z,c,reg=current();after=vector.square().sum(1)/z.square()+gamma*reg/total.square()
 return selected,before,after
