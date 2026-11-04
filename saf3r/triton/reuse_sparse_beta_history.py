"""Export actual sparse probabilities; no estimated Dense/Grid denominator."""
import torch
import triton
import triton.language as tl
from .reuse_ready_history import _pack_bits, _load_cached


@triton.jit
def _export(RAW,BIAS,LSE,IDS,GRID,CACHE,MASS,
            A:tl.constexpr,H:tl.constexpr,ROWS:tl.constexpr,R:tl.constexpr,
            B:tl.constexpr,BH:tl.constexpr,PACKED:tl.constexpr):
    tile=tl.program_id(0);p=tl.program_id(1);g=tl.program_id(2)
    col=tile*B+tl.arange(0,B);h=tl.arange(0,BH)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    raw=tl.load(RAW+((g*H+h[:,None])*4+p)*A+col[None,:],(h[:,None]<H)&valid[None,:],-float('inf'))
    bias=tl.load(BIAS+g*A+col,valid,-float('inf'))
    row=tl.minimum(g*32+p*8+4,ROWS-1)
    lse=tl.load(LSE+h*ROWS+row,h<H,0.)
    prob=tl.exp(raw+bias[None,:]-lse[:,None])
    value=tl.sum(tl.where((h[:,None]<H)&valid[None,:],prob,0.),0)/H
    half=value.to(tl.float16)
    if PACKED:
        _pack_bits(half.to(tl.uint16,bitcast=True),CACHE,g*4+p,tile,A,B)
    else:
        tl.store(CACHE+(g*4+p)*A+col,half,col<A)
    # Sum the SAME quantized values that the next-layer consumer will decode.
    tl.atomic_add(MASS+(g*4+p)*R+region,half.to(tl.float32),valid,sem='relaxed')


@triton.jit
def _apply(CACHE,IDS,GRID,SCALE,FALLBACK,WEIGHTS,PRIOR,
           A:tl.constexpr,N:tl.constexpr,R:tl.constexpr,B:tl.constexpr,PACKED:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B);p=tl.arange(0,4)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    factor=tl.load(SCALE+(g*4+p[:,None])*R+region[None,:],valid[None,:],0.)
    fallback=tl.load(FALLBACK+(g*4+p[:,None])*R+region[None,:],valid[None,:],0.)
    old=_load_cached(CACHE,g*4+p[:,None],col[None,:],valid[None,:],A,PACKED)
    weight=tl.load(WEIGHTS+g*4+p)
    probability=tl.sum((old*factor+fallback)*weight[:,None],0)
    tl.store(PRIOR+g*N+key,probability,valid)


def export(raw,bias,lse,ids,grid,cache,mass):
    g,h,probes,a=raw.shape
    if probes!=4 or tuple(bias.shape)!=(g,a) or lse.shape[0]!=h:
        raise ValueError('Sparse-probability export shape mismatch')
    _export[(triton.cdiv(a,128),4,g)](raw,bias.contiguous(),lse,ids,grid,cache,mass,
        a,h,lse.shape[1],mass.shape[-1],128,triton.next_power_of_2(h),cache.dtype==torch.uint8,
        num_warps=4,enable_fp_fusion=False)


def apply(cache,ids,grid,scale,fallback,weights,prior):
    g,a=ids.shape
    _apply[(g,triton.cdiv(a,256))](cache,ids,grid,scale.contiguous(),fallback.contiguous(),
        weights.contiguous(),prior,a,len(grid),scale.shape[-1],256,cache.dtype==torch.uint8,
        num_warps=4,enable_fp_fusion=False)

