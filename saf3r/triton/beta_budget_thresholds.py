"""Experimental interval-event index for the four-state dual search.

Intervals index costs only. Final state decoding still uses the original
objective and leftmost tie-breaking. A failed budget check falls back to
the original solver. Keep this path separate from production selection.
"""
import torch
import triton

from .reuse_compact import _choose, budget


def interval_events(errors, costs):
    events=[]
    changes=[]
    for state in range(4):
        error=errors[...,state]
        cost=costs[...,state]
        lower=torch.zeros_like(error)
        upper=torch.full_like(error,float('inf'))
        valid=torch.isfinite(error)
        for other in range(4):
            if state==other:
                continue
            other_error=errors[...,other]
            other_cost=costs[...,other]
            slope=cost-other_cost
            crossing=(other_error-error)/torch.where(slope==0,1.,slope)
            finite=torch.isfinite(other_error)
            lower=torch.where(finite&(slope<0),torch.maximum(lower,crossing),lower)
            upper=torch.where(finite&(slope>0),torch.minimum(upper,crossing),upper)
            dominated=finite&(slope==0)&((other_error<error)|((other_error==error)&(other<state)))
            valid=valid&~dominated
        valid=valid&(upper>lower)
        events.extend((torch.where(valid,lower,float('inf')),torch.where(valid,upper,float('inf'))))
        change=torch.where(valid,cost,0.)
        changes.extend((change,-change))
    positions,order=torch.cat(events,dim=1).sort(dim=1)
    cumulative=torch.cat(changes,dim=1).gather(1,order).cumsum(1)
    return positions,cumulative


def threshold_budget(errors,costs,upper,cap,iterations):
    if errors.shape[-1]!=4:
        raise ValueError('The interval index expects four states')
    positions,cumulative=interval_events(errors,costs)
    lo=torch.zeros_like(upper)
    hi=upper
    for _ in range(iterations):
        mid=(lo+hi)*.5
        index=torch.searchsorted(positions,mid[:,None],right=True).squeeze(1)-1
        count=cumulative.gather(1,index.clamp_min(0)[:,None]).squeeze(1)
        too_many=(index>=0)&(count>cap)
        lo=torch.where(too_many,mid,lo)
        hi=torch.where(too_many,hi,mid)
    groups,clusters,states=errors.shape
    tiles=triton.cdiv(clusters,128)
    selected=torch.empty((groups,clusters),device=errors.device,dtype=torch.int64)
    partial=torch.empty((groups,tiles),device=errors.device)
    dual=torch.stack((lo,hi),1).contiguous()
    _choose[(groups,tiles)](errors,costs,dual,partial,selected,clusters,states,tiles,True,128,4,
                           num_warps=4,enable_fp_fusion=False)
    counts=partial.sum(1)
    if bool((counts>cap).any()):
        return (*budget(errors,costs,upper,cap,iterations),True)
    return selected,counts,False
