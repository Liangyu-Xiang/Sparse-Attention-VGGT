"""One bounded pass over emitted Keys; no extra QK or population scatter."""
import torch
import triton
import triton.language as tl

_RATIO_TABLES={}


def ratio_table(device):
    key=str(device)
    if key not in _RATIO_TABLES:
        # All legal integer populations/divisors fit the existing width<=512
        # contract. Build exact FP32 quotients once; avoid billions of div.rn
        # operations across emitted Keys. This cache contains no input data.
        n=torch.arange(513,dtype=torch.float32)
        _RATIO_TABLES[key]=(n[:,None]/n.clamp_min(1)[None,:]).to(device)
    return _RATIO_TABLES[key]


@triton.jit
def _weights(IDS,LABEL,SELECTED,POP,ELIGIBLE,RATIO,OUT,A:tl.constexpr,C:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0)
    col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+group*A+col,col<A,-1)
    valid=(col<A)&(key>=0)
    cluster=tl.load(LABEL+tl.maximum(key,0),valid,0)
    state=tl.load(SELECTED+group*C+cluster,valid,0)
    population=tl.load(POP+cluster,valid,1.)
    history=tl.load(ELIGIBLE+group*C+cluster,valid,1)
    divisor=tl.where(state==5,history.to(tl.float32),tl.where(state==10,population,1.))
    index=population.to(tl.int32)*513+tl.maximum(divisor,1.).to(tl.int32)
    weight=tl.load(RATIO+index,valid,0.)
    tl.store(OUT+group*A+col,tl.where(valid,weight,0.),col<A)


def cluster_weights(selected,ids,labels,population,eligible):
    out=torch.empty(ids.shape,device=ids.device,dtype=torch.float32)
    ratios=ratio_table(ids.device)
    _weights[(len(ids),triton.cdiv(ids.shape[1],256))](ids,labels,selected,population,eligible,ratios,out,
        ids.shape[1],len(population),256,num_warps=4,enable_fp_fusion=False)
    return out
