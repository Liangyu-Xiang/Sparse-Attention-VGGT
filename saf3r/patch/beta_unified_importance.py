"""Headmean global prior plus shrunken log-importance correction; no block lock."""
import torch
ALPHA=.5
TAU=.5
from .beta_headmean_importance import observe


def estimate(mean,coarse,previous,ids,history,grid,sizes,layer):
    g,r=coarse.shape;n=len(grid);uniform=coarse[:,grid]/sizes[grid]
    if previous is None:return uniform.contiguous(),torch.zeros_like(coarse)
    valid=ids>=0;safe=ids.clamp_min(0).long();block=grid[safe]
    tiny=torch.finfo(coarse.dtype).tiny
    beta=(torch.minimum(coarse,previous)/torch.maximum(coarse,previous).clamp_min(tiny)).clamp_min(0).pow(TAU)
    previous_base=previous.gather(1,block)/sizes[block]
    residual=(history/previous_base.clamp_min(1e-30)).clamp_min(1e-30).log()
    lift=(ALPHA*beta.gather(1,block)*residual).exp()
    values=uniform.gather(1,safe)*lift
    index=safe.masked_fill(~valid,n)
    prior=torch.cat((uniform,uniform.new_zeros(g,1)),1).scatter_(1,index,values.masked_fill(~valid,0.))[:,:n]
    return (prior/prior.sum(1,keepdim=True)).contiguous(),beta
