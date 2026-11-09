"""Same probability models, fused normalization only;no per-head history saved."""
from .beta_unified_importance import estimate
from saf3r.triton.beta_importance_finalize import finalize

def observe(mean,grid,sizes,ids,raw,shift=None):return finalize(mean,grid,sizes,ids,raw,shift=shift)
def observe_raw(mean,grid,sizes,ids,raw):return finalize(mean,grid,sizes,ids,raw,logit=True)
def observe_subgroup(mean,grid,sizes,ids,raw,counts,shift=None,history_sum=None):
    return finalize(mean,grid,sizes,ids,raw,counts=counts,shift=shift,history_sum=history_sum)
