"""Strict one-layer actual-support logit residual transfer; no fine-QK refresh."""
import torch
from .one_layer_edge_history import mean_valid
VARIANTS=['uniform','direct','shrink025','shrink050','shrink075','coarse_beta','coarse_rms','centered_direct','centered050','headmean_direct','headmean050']

def correction(delta,ids,grid,mean,previous_mean,beta,variant):
 if variant=='direct':return delta
 if variant.startswith('headmean'):return delta.mean(0,keepdim=True).expand_as(delta)*(.5 if variant=='headmean050' else 1.)
 if variant.startswith('shrink'):return delta*{'shrink025':.25,'shrink050':.5,'shrink075':.75}[variant]
 valid=ids>=0;safe=ids.clamp_min(0).long();block=grid[safe];h,g,p,a=delta.shape;r=mean.shape[-1]
 if variant=='coarse_beta':return delta*beta.gather(1,block)[None,:,None]
 if variant=='coarse_rms':
  # Current coarse observations only; no current per-edge evidence.
  scale=(mean.var(-1,unbiased=False).sqrt()/previous_mean.var(-1,unbiased=False).sqrt().clamp_min(1e-6)).clamp(0,1)
  return delta*scale[...,None]
 if variant.startswith('centered'):
  ix=block[None,:,None].expand(h,-1,p,-1);count=torch.zeros(g,r,device=ids.device).scatter_add_(1,block,valid.float())
  total=torch.zeros(h,g,p,r,device=ids.device).scatter_add_(3,ix,delta.masked_fill(~valid[None,:,None],0.))
  offset=(total/count.clamp_min(1)[None,:,None]).gather(3,ix)
  return (delta-offset)*(.5 if variant=='centered050' else 1.)
 raise ValueError(variant)

def estimate(mean,coarse,previous_coarse,previous_mean,ids,delta,grid,sizes,eligible,variant='direct'):
 h,g,p,r=mean.shape;n=len(grid);scores=mean[...,grid].clone();beta=torch.zeros_like(coarse)
 if previous_coarse is not None and variant!='uniform':
  beta=(2*coarse.sqrt()*previous_coarse.sqrt()/(coarse+previous_coarse).clamp_min(1e-30)).clamp(0,1)
  valid=ids>=0;safe=ids.clamp_min(0).long();change=correction(delta,ids,grid,mean,previous_mean,beta,variant)
  changed=mean.gather(3,grid[safe][None,:,None].expand(h,-1,p,-1))+change
  # Current fine scores of unknown edges are only repeated coarse means.
  tombstone=safe.masked_fill(~valid,n)[None,:,None].expand(h,-1,p,-1)
  scores=torch.cat((scores,scores.new_zeros(h,g,p,1)),-1).scatter_(3,tombstone,changed.masked_fill(~valid[None,:,None],0.))[...,:n]
 prior=mean_valid(scores.softmax(-1),eligible)
 return prior.contiguous(),beta

def observe(mean,grid,ids,raw):
 valid=ids>=0;safe=ids.clamp_min(0).long();ix=grid[safe][None,:,None].expand_as(raw)
 return (raw-mean.gather(3,ix)).masked_fill(~valid[None,:,None],0.).contiguous()
