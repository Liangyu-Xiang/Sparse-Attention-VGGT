"""Compact log-probability calibration; no Q/K inputs or attention routing.

Selected scores have had the shared compensation bias removed in log space.
Missing members receive a pooled-Key score in the SAME sparse-LSE scale.
Segment reductions accept arbitrary key order, holes and repeated Grid runs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _segment_max(a,fa,b,fb):
    return tl.where(fb,b,tl.maximum(a,b)),fa|fb


@triton.jit
def _segment_sum(a,fa,b,fb):
    return tl.where(fb,b,a+b),fa|fb


@triton.jit
def _reduce(SCORE,IDS,GRID,MAXIMUM,SUM,A:tl.constexpr,P:tl.constexpr,R:tl.constexpr,B:tl.constexpr,MAX:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);p=tl.program_id(1);g=tl.program_id(2)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    prev=tl.load(IDS+g*A+col-1,(col>0)&(col<A),-1)
    pregion=tl.load(GRID+tl.maximum(prev,0),prev>=0,0)
    flag=(col%B==0)|(prev<0)|(pregion!=region)|~valid
    nxt=tl.load(IDS+g*A+col+1,col+1<A,-1)
    nregion=tl.load(GRID+tl.maximum(nxt,0),nxt>=0,0)
    last=(col%B==B-1)|(col==A-1)|(nxt<0)|(nregion!=region)
    score=tl.load(SCORE+(g*P+p)*A+col,valid,-float('inf'))
    if MAX:
        value,_=tl.associative_scan((score,flag),0,_segment_max)
        tl.atomic_max(MAXIMUM+(g*P+p)*R+region,value,valid&last,sem='relaxed')
    else:
        maximum=tl.load(MAXIMUM+(g*P+p)*R+region,valid,0.)
        value=tl.where(valid,tl.exp(score-maximum),0.)
        value,_=tl.associative_scan((value,flag),0,_segment_sum)
        tl.atomic_add(SUM+(g*P+p)*R+region,value,valid&last,sem='relaxed')


@triton.jit
def _store(SCORE,DEN,IDS,GRID,CACHE,A:tl.constexpr,P:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);p=tl.program_id(1);g=tl.program_id(2)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    score=tl.load(SCORE+(g*P+p)*A+col,valid,0.)
    den=tl.load(DEN+(g*P+p)*R+region,valid,0.)
    # Log conditionals preserve very small probabilities in a half cache.
    tl.store(CACHE+(g*P+p)*A+col,tl.where(valid,tl.minimum(score-den,0.),0.),col<A)


def store(scores,missing_scores,ids,grid,sizes,cache):
    g,p,a=scores.shape;r=len(sizes)
    regions=grid[ids.long().clamp_min(0)]
    counts=torch.zeros((g,r),device=ids.device,dtype=torch.int32).scatter_add_(1,regions,(ids>=0).int())
    remaining=sizes[None]-counts
    maximum=missing_scores.masked_fill(remaining[:,None,:]==0,-torch.inf).contiguous()
    sums=torch.zeros_like(maximum)
    if a:
        _reduce[(triton.cdiv(a,256),p,g)](scores,ids,grid,maximum,sums,a,p,r,256,True,num_warps=4)
        if torch.are_deterministic_algorithms_enabled():
            from .reuse_deterministic import grid_sum
            sums=grid_sum(scores,ids,grid,maximum)
        else:_reduce[(triton.cdiv(a,256),p,g)](scores,ids,grid,maximum,sums,a,p,r,256,False,num_warps=4)
    sums.add_(torch.where(remaining[:,None,:]>0,remaining[:,None,:]*(missing_scores-maximum).exp(),0.))
    den=maximum+sums.log()
    uniform=(missing_scores-den).exp().masked_fill(remaining[:,None,:]==0,0.)
    if a:_store[(triton.cdiv(a,256),p,g)](scores,den,ids,grid,cache,a,p,r,256,num_warps=4)
    return uniform,counts,den


@triton.jit
def _predict(CACHE,IDS,GRID,SIZES,CUR,PREV,WEIGHTS,PRIOR,A:tl.constexpr,R:tl.constexpr,N:tl.constexpr,B:tl.constexpr,EPS:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);g=tl.program_id(1);p=tl.arange(0,4)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0);size=tl.load(SIZES+region,valid,1.)
    ix=(g*4+p[:,None])*R+region[None,:]
    current=tl.load(CUR+ix,valid[None,:],0.);previous=tl.load(PREV+ix,valid[None,:],0.)
    ratio=(current+EPS)/(previous+EPS)
    beta=tl.minimum(1.,tl.where(ratio<1.,ratio,1./(ratio*ratio)))
    logp=tl.load(CACHE+(g*4+p[:,None])*A+col[None,:],valid[None,:],-float('inf')).to(tl.float32)
    value=current*(beta*tl.exp(logp)+(1.-beta)/size[None,:])
    weights=tl.load(WEIGHTS+g*4+p)
    result=tl.sum(value*weights[:,None],0)
    tl.store(PRIOR+g*N+key,result,valid)


def predict(cache,ids,grid,sizes,current,previous,weights,prior,eps):
    _predict[(triton.cdiv(ids.shape[1],256),len(ids))](cache,ids,grid,sizes,current.contiguous(),previous,weights.contiguous(),prior,ids.shape[1],len(sizes),len(grid),256,eps,num_warps=4)
