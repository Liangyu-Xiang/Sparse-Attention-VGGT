"""Only previous executed edges carry head-mean importance across layers."""
import torch


def observe(mean,grid,sizes,ids,raw):
    raw=raw[:,:,0];mean=mean[:,:,0]
    h,g,a=raw.shape;r=len(sizes);valid=ids>=0;safe=ids.clamp_min(0).long();block=grid[safe]
    count=raw.new_zeros(g,r).scatter_add_(1,block,valid.float())
    missing=(sizes[None]-count).clamp_min(0)
    unknown=(mean+missing.clamp_min(1).log()[None]).masked_fill(missing[None]==0,-torch.inf).logsumexp(-1)
    observed=raw.masked_fill(~valid[None],-torch.inf)
    denominator=torch.logaddexp(observed.logsumexp(-1),unknown)
    return (observed-denominator[...,None]).exp().mean(0).masked_fill(~valid,0.).contiguous()


def estimate(coarse,previous,ids,history,grid,sizes):
    g,r=coarse.shape;n=len(grid);uniform=coarse[:,grid]/sizes[grid]
    if previous is None:return uniform.contiguous(),torch.zeros_like(coarse)
    valid=ids>=0;safe=ids.clamp_min(0).long();block=grid[safe]
    tiny=torch.finfo(coarse.dtype).tiny
    beta=(-(coarse.clamp_min(tiny).log()-previous.clamp_min(tiny).log()).abs()).exp()
    b=beta.gather(1,block)
    transported=history*(coarse/previous.clamp_min(1e-30)).gather(1,block)
    values=b*transported+(1-b)*uniform.gather(1,safe)
    index=safe.masked_fill(~valid,n)
    prior=torch.cat((uniform,uniform.new_zeros(g,1)),1).scatter_(1,index,values.masked_fill(~valid,0.))[:,:n]
    return prior.contiguous(),beta
