"""Exact duplicate-cost compression and a truth-free feasible8-step bracket."""
import torch
import triton
import triton.language as tl
from .reuse_compact import _dual

@triton.jit
def _compress(E,C,V,I,POP,BOUND,PROFIT,F:tl.constexpr,S:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BS:tl.constexpr):
    g=tl.program_id(0);tile=tl.program_id(1);row=tile*B+tl.arange(0,B);s=tl.arange(0,BS)
    error=tl.load(E+(g*F+row[:,None])*S+s[None,:],(row[:,None]<F)&(s[None,:]<S),float('inf'))
    cost=tl.load(C+(g*F+row[:,None])*S+s[None,:],(row[:,None]<F)&(s[None,:]<S),-1.)
    one=tl.min(tl.where(cost==1.,error,float('inf')),1)
    drop=tl.load(E+(g*F+row)*S,row<F,0.)
    tl.store(PROFIT+g*F+row,tl.maximum(drop-one,0.),row<F)
    gain=tl.where(cost>1,tl.maximum(0.,one[:,None]-error)/tl.maximum(cost-1,1.),0.)
    tl.store(BOUND+g*T+tile,tl.max(tl.max(gain,1),0))
    for k in tl.static_range(6):
        value=tl.min(tl.where(cost==k,error,float('inf')),1)
        state=tl.min(tl.where((cost==k)&(error==value[:,None])&(s[None,:]<S),s[None,:],255),1)
        tl.store(V+(g*F+row)*7+k,value,row<F);tl.store(I+(g*F+row)*7+k,state,row<F)
    exact=tl.load(E+(g*F+row)*S+S-1,row<F,float('inf'))
    population=tl.load(C+(g*F+row)*S+S-1,row<F,0.)
    tl.store(V+(g*F+row)*7+6,exact,row<F);tl.store(I+(g*F+row)*7+6,S-1,row<F)
    tl.store(POP+g*F+row,population,row<F)

@triton.jit
def _choose(V,I,POP,DUAL,PART,SELECT,F:tl.constexpr,T:tl.constexpr,FINAL:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);tile=tl.program_id(1);row=tile*B+tl.arange(0,B);s=tl.arange(0,8)
    value=tl.load(V+(g*F+row[:,None])*7+s[None,:],(row[:,None]<F)&(s[None,:]<7),float('inf'))
    original=tl.load(I+(g*F+row[:,None])*7+s[None,:],(row[:,None]<F)&(s[None,:]<7),255).to(tl.int32)
    population=tl.load(POP+g*F+row,row<F,0.)
    costs=tl.where(s[None,:]==6,population[:,None],s[None,:].to(tl.float32))
    lo=tl.load(DUAL+g*2);hi=tl.load(DUAL+g*2+1);price=hi if FINAL else (lo+hi)*.5
    objective=value+price*costs
    minimum=tl.min(objective,1)
    state=tl.min(tl.where(objective==minimum[:,None],original,255),1)
    # Exact may duplicate a small integer cost. Select a single copy when
    # counting, while original-state-index tie breaking matches11-state argmin.
    slot=tl.min(tl.where((objective==minimum[:,None])&(original==state[:,None]),s[None,:],8),1)
    count=tl.sum(tl.where(s[None,:]==slot[:,None],costs,0.),1)
    tl.store(PART+g*T+tile,tl.sum(tl.where(row<F,count,0.),0))
    if FINAL:tl.store(SELECT+g*F+row,state,row<F)

def budget(errors,costs,upper,cap,iterations,tighten=False):
    g,f,s=errors.shape
    if s!=11: # Standalone compatibility; this guarded experiment fixesH=4.
        from .reuse_compact import budget as reference
        return reference(errors,costs,upper,cap,iterations)
    tiles=triton.cdiv(f,128)
    values=torch.empty((g,f,7),device=errors.device);indices=torch.empty((g,f,7),device=errors.device,dtype=torch.uint8)
    population=torch.empty((g,f),device=errors.device);bound=torch.empty((g,tiles),device=errors.device);profit=torch.empty((g,f),device=errors.device)
    _compress[(g,tiles)](errors,costs,values,indices,population,bound,profit,f,s,tiles,128,16,num_warps=4)
    if tighten:
        high=bound.amax(1)
        if cap<f:
            threshold=profit.topk(cap+1,dim=1,sorted=False).values.amin(1)
            high=torch.maximum(high,threshold)
        upper=torch.minimum(upper,high+1e-12)
    dual=torch.stack((torch.zeros_like(upper),upper),1).contiguous();partial=torch.empty((g,tiles),device=errors.device)
    selected=torch.empty((g,f),device=errors.device,dtype=torch.int64)
    for _ in range(iterations):
        _choose[(g,tiles)](values,indices,population,dual,partial,selected,f,tiles,False,128,num_warps=4,enable_fp_fusion=False)
        _dual[(g,)](partial,dual,tiles,cap,triton.next_power_of_2(tiles),num_warps=4,enable_fp_fusion=False)
    _choose[(g,tiles)](values,indices,population,dual,partial,selected,f,tiles,True,128,num_warps=4,enable_fp_fusion=False)
    return selected,partial.sum(1)
