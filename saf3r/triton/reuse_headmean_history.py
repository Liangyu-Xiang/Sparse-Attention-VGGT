"""Fused per-head normalization, head mean, FP16 quantization and packing."""
import torch,triton
import triton.language as tl
from .reuse_ready_history import _pack_bits

@triton.jit
def _store_mean(RAW,LOGZ,IDS,GRID,CORRECTED,CACHE,A:tl.constexpr,H:tl.constexpr,R:tl.constexpr,B:tl.constexpr,BH:tl.constexpr,EPS:tl.constexpr,PACKED:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);p=tl.program_id(1);g=tl.program_id(2)
    head=tl.arange(0,BH)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    row=(g*H+head)*4+p
    raw=tl.load(RAW+row[:,None]*A+col[None,:],(head[:,None]<H)&valid[None,:],-float('inf'))
    logz=tl.load(LOGZ+row,head<H,0.)
    corrected=tl.load(CORRECTED+row[:,None]*R+region[None,:],(head[:,None]<H)&valid[None,:],1.)
    value=tl.exp(raw-logz[:,None])/tl.maximum(corrected,EPS)
    conditional=tl.sum(tl.where((head[:,None]<H)&valid[None,:],value,0.),0)/H
    if PACKED:
        encoded=conditional.to(tl.float16).to(tl.uint16,bitcast=True)
        _pack_bits(encoded,CACHE,g*4+p,tl.program_id(0),A,B)
    else:tl.store(CACHE+(g*4+p)*A+col,conditional,col<A)

def store_mean(raw,logz,ids,grid,corrected,cache,eps):
    g,h,_,a=raw.shape
    _store_mean[(triton.cdiv(a,128),4,g)](raw,logz,ids,grid,corrected,cache,a,h,corrected.shape[-1],128,triton.next_power_of_2(h),eps,cache.dtype==torch.uint8,num_warps=4,enable_fp_fusion=False)
