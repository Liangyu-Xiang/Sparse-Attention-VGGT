"""Opt-in fixed-order reductions shared by the reference and optimized execution.

Same cluster/probability sums and masks. Torch's deterministic scatter selects a
fixed accumulation order; no CUDA floating atomic adds are used in these sums.
"""
import torch

import triton
import triton.language as tl

@triton.jit
def _presence(IDS,SEEN,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1)
    tl.store(SEEN+g*N+key,True,(col<A)&(key>=0))

@triton.jit
def _fixed_corrections(PRIOR,UNIFORM,GRID,MEM,DROP,REP,SEEN,QUAL,D,R,K,
                       N:tl.constexpr,F:tl.constexpr,REGIONS:tl.constexpr,
                       WIDTH:tl.constexpr,M:tl.constexpr,C:tl.constexpr):
    g=tl.program_id(0);cluster=tl.program_id(1)*C+tl.arange(0,C);m=tl.arange(0,M)
    key=tl.load(MEM+cluster[:,None]*WIDTH+m[None,:],(cluster[:,None]<F)&(m[None,:]<WIDTH),-1)
    valid=(cluster[:,None]<F)&(key>=0)
    observed=tl.load(SEEN+g*N+tl.maximum(key,0),valid,False)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    probability=tl.load(PRIOR+g*N+tl.maximum(key,0),valid,0.)
    uniform=tl.load(UNIFORM+g*REGIONS+region,valid,0.)
    delta=probability-uniform
    a=tl.load(DROP+tl.maximum(key,0),valid,0.);b=tl.load(REP+tl.maximum(key,0),valid,0.)
    qualified=tl.load(QUAL+g*N+tl.maximum(key,0),valid,False)
    d=tl.sum(tl.where(observed&(delta!=0.),delta*a,0.),1)
    r=tl.sum(tl.where(observed&(delta!=0.),delta*b,0.),1)
    kept=tl.sum(tl.where(observed&qualified,probability*a,0.),1)
    tl.store(D+g*F+cluster,d,cluster<F);tl.store(R+g*F+cluster,r,cluster<F);tl.store(K+g*F+cluster,kept,cluster<F)

def corrections(prior,old_ids,uniform,labels,geometry,context):
    g,n=prior.shape;f=len(geometry['shared_cells']);width=geometry['width']
    seen=torch.zeros((g,n),device=prior.device,dtype=torch.bool)
    _presence[(g,triton.cdiv(old_ids.shape[1],256))](old_ids,seen,n,old_ids.shape[1],256,num_warps=4)
    d=torch.empty((g,f),device=prior.device,dtype=torch.float32);r=torch.empty_like(d);kept=torch.empty_like(d)
    _fixed_corrections[(g,triton.cdiv(f,2))](prior,uniform,geometry['shared_grid'],geometry['deterministic_members'],geometry['shared_token_drop'],geometry['shared_token_rep'],seen,context['qualified'],d,r,kept,n,f,uniform.shape[1],width,max(64,width),2,num_warps=4,enable_fp_fusion=False)
    return d,r,kept

def grid_sum(scores,ids,grid,maximum):
    g,p,a=scores.shape;r=maximum.shape[-1]
    valid=ids>=0;regions=grid[ids.long().clamp_min(0)].long()[:,None,:].expand(g,p,a)
    values=(scores-maximum.gather(2,regions)).exp().masked_fill(~valid[:,None,:],0.)
    return torch.zeros_like(maximum).scatter_add_(2,regions,values)
