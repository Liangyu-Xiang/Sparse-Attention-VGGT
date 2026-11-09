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


from dataclasses import dataclass


@dataclass(frozen=True)
class CompactPrior:
    """Normalized per-region weights and corrections for unique historical IDs."""

    block_weight: torch.Tensor
    grid: torch.Tensor
    ids: torch.Tensor
    delta: torch.Tensor | None


def estimate_compact(coarse, previous, ids, history, grid, sizes):
    base = coarse / sizes
    if previous is None:
        return CompactPrior(base.contiguous(), grid, ids, None), torch.zeros_like(coarse)

    valid = ids >= 0
    block = grid[ids.clamp_min(0).long()].long()
    tiny = torch.finfo(coarse.dtype).tiny
    beta = (torch.minimum(coarse, previous) / torch.maximum(coarse, previous).clamp_min(tiny)).clamp_min(0).pow(TAU)
    old_base = (previous / sizes).gather(1, block)
    residual = (history / old_base.clamp_min(1e-30)).clamp_min(1e-30).log()
    selected_base = base.gather(1, block)
    lift = (ALPHA * beta.gather(1, block) * residual).exp()
    delta = (selected_base * lift - selected_base).masked_fill(~valid, 0.)
    normalizer = (base * sizes).sum(1, keepdim=True) + delta.sum(1, keepdim=True)
    return CompactPrior((base / normalizer).contiguous(), grid, ids, (delta / normalizer).contiguous()), beta


def estimate_exact_expanded(coarse,previous,ids,history,grid,sizes,*,fused_lift=False):
    """Preserve the expanded reference prior and its FP32 reduction order."""
    import triton
    from saf3r.triton.beta_compact_prior import _expand_reference,_override_reference,_apply_reference_lift
    groups,regions=coarse.shape
    keys=len(grid)
    base=coarse/sizes
    if previous is None:
        return base[:,grid].contiguous(),torch.zeros_like(coarse)
    tiny=torch.finfo(coarse.dtype).tiny
    beta=(torch.minimum(coarse,previous)/torch.maximum(coarse,previous).clamp_min(tiny)).clamp_min(0).pow(TAU)
    expanded=torch.empty((groups,keys+1),device=coarse.device,dtype=coarse.dtype)
    _expand_reference[(groups,triton.cdiv(keys+1,512))](base,grid,expanded,keys,regions,512)
    slots=ids.shape[1]
    if fused_lift:
        previous_base=(previous/sizes).contiguous()
        _apply_reference_lift[(groups,triton.cdiv(slots,256))](
            ids,history.contiguous(),grid,base,previous_base,beta,expanded,
            keys,regions,slots,256,num_warps=4,enable_fp_fusion=False)
    else:
        safe=ids.clamp_min(0).long()
        region=grid[safe]
        previous_base=previous.gather(1,region)/sizes[region]
        residual=(history/previous_base.clamp_min(1e-30)).clamp_min(1e-30).log()
        lift=(ALPHA*beta.gather(1,region)*residual).exp()
        values=base.gather(1,region)*lift
        _override_reference[(groups,triton.cdiv(slots,512))](ids,values,expanded,keys,slots,512)
    prior=expanded[:,:keys]
    return (prior/prior.sum(1,keepdim=True)).contiguous(),beta
