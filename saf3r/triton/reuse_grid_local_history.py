"""Stable per-grid normalizers, then fused head reduction and cache store.

Uses the attention backend's already-paid stable grouping. RAW stays in logical
support order; no QK, sorting, atomics, or segmented scans are launched here.
"""
import torch
import triton
import triton.language as tl
from .reuse_ready_history import _pack_bits


@triton.jit
def _normalizers(RAW, BASE, ORDER, OFF, SIZE, DEN, ERROR,
                 A:tl.constexpr,H:tl.constexpr,R:tl.constexpr,B:tl.constexpr,T:tl.constexpr):
    g=tl.program_id(0);hp=tl.program_id(1)
    rr=tl.program_id(2)*T+tl.arange(0,T)
    col=tl.arange(0,B)
    begin=tl.load(OFF+g*(R+1)+rr,rr<R,0)
    end=tl.load(OFF+g*(R+1)+rr+1,rr<R,0)
    pos=begin[:,None]+col[None,:]
    valid=(rr[:,None]<R)&(pos<end[:,None])
    logical=tl.load(ORDER+g*A+pos,valid,0)
    raw=tl.load(RAW+(g*H*4+hp)*A+logical,valid,-float('inf'))
    base=tl.load(BASE+(g*H*4+hp)*R+rr,rr<R,0.)
    size=tl.load(SIZE+rr,rr<R,0.)
    missing=size-(end-begin)
    shift=tl.maximum(tl.max(raw,1),tl.where(missing>0,base,-float('inf')))
    safe=tl.where(shift==-float('inf'),0.,shift)
    selected=tl.sum(tl.exp(raw-safe[:,None]),1)
    omitted=tl.where(missing>0,missing*tl.exp(base-safe),0.)
    total=selected+omitted
    logd=tl.where(total>0,safe+tl.log(total),-float('inf'))
    tl.store(DEN+(g*H*4+hp)*R+rr,logd,rr<R)
    err=tl.abs(tl.exp(safe-logd)*total-1.)
    tl.store(ERROR+(g*H*4+hp)*R+rr,err,rr<R)


@triton.jit
def _compat(DEN,SPECIAL,OUT,R:tl.constexpr,S:tl.constexpr,BR:tl.constexpr,BS:tl.constexpr):
    # Only needed to preserve the old max(corrected_grid, EPS) semantics.
    # This reduces small grid/special arrays, never scans selected-edge RAW.
    row=tl.program_id(0);r=tl.arange(0,BR);s=tl.arange(0,BS)
    d=tl.load(DEN+row*R+r,r<R,-float('inf'))
    z=tl.load(SPECIAL+row*S+s,s<S,-float('inf'))
    m=tl.maximum(tl.max(d,0),tl.max(z,0))
    total=tl.sum(tl.exp(d-m),0)+tl.sum(tl.exp(z-m),0)
    tl.store(OUT+row,m+tl.log(total))


@triton.jit
def _uniform(BASE,DEN,LOGZ,OUT,H:tl.constexpr,R:tl.constexpr,BH:tl.constexpr,T:tl.constexpr,LOGEPS:tl.constexpr,LEGACY:tl.constexpr):
    g=tl.program_id(0);p=tl.program_id(1);r=tl.program_id(2)*T+tl.arange(0,T)
    h=tl.arange(0,BH);row=(g*H+h)*4+p
    d=tl.load(DEN+row[:,None]*R+r[None,:],(h[:,None]<H)&(r[None,:]<R),0.)
    b=tl.load(BASE+row[:,None]*R+r[None,:],(h[:,None]<H)&(r[None,:]<R),0.)
    extra=tl.full((BH,T),0.,tl.float32)
    if LEGACY:
        z=tl.load(LOGZ+row,h<H,0.)
        extra=tl.minimum(d-z[:,None]-LOGEPS,0.)
    u=tl.exp(b-d+extra)
    value=tl.sum(tl.where(h[:,None]<H,u,0.),0)/H
    tl.store(OUT+(g*4+p)*R+r,value,r<R)


@triton.jit
def _store_local(RAW,DEN,LOGZ,IDS,GRID,CACHE,A:tl.constexpr,H:tl.constexpr,R:tl.constexpr,B:tl.constexpr,BH:tl.constexpr,LOGEPS:tl.constexpr,LEGACY:tl.constexpr,PACKED:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);p=tl.program_id(1);g=tl.program_id(2)
    h=tl.arange(0,BH);row=(g*H+h)*4+p
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    raw=tl.load(RAW+row[:,None]*A+col[None,:],(h[:,None]<H)&valid[None,:],-float('inf'))
    d=tl.load(DEN+row[:,None]*R+region[None,:],(h[:,None]<H)&valid[None,:],0.)
    extra=tl.full((BH,B),0.,tl.float32)
    if LEGACY:
        z=tl.load(LOGZ+row,h<H,0.)
        extra=tl.minimum(d-z[:,None]-LOGEPS,0.)
    value=tl.exp(raw-d+extra)
    conditional=tl.sum(tl.where((h[:,None]<H)&valid[None,:],value,0.),0)/H
    if PACKED:
        encoded=conditional.to(tl.float16).to(tl.uint16,bitcast=True)
        _pack_bits(encoded,CACHE,g*4+p,tl.program_id(0),A,B)
    else:
        tl.store(CACHE+(g*4+p)*A+col,conditional,col<A)


def normalizers(raw,base,order,offsets,sizes,max_members,tile=4):
    g,h,_,a=raw.shape;r=base.shape[-1]
    base=base.contiguous()
    den=torch.empty(base.shape,device=base.device,dtype=base.dtype);error=torch.empty_like(den)
    _normalizers[(g,h*4,triton.cdiv(r,tile))](raw,base,order,offsets,sizes,den,error,a,h,r,triton.next_power_of_2(max_members),tile,num_warps=4)
    return den,error


def compatibility(den,special):
    special=special.contiguous()
    result=torch.empty(den.shape[:-1],device=den.device,dtype=torch.float32)
    _compat[(result.numel(),)](den,special,result,den.shape[-1],special.shape[-1],triton.next_power_of_2(den.shape[-1]),triton.next_power_of_2(special.shape[-1]),num_warps=4)
    return result


def store(raw,den,logz,ids,grid,base,cache,uniform,eps=1e-30,legacy=True):
    import math
    base=base.contiguous()
    g,h,_,a=raw.shape;r=den.shape[-1];bh=triton.next_power_of_2(h)
    _uniform[(g,4,triton.cdiv(r,8))](base,den,logz,uniform,h,r,bh,8,math.log(eps),legacy,num_warps=4)
    _store_local[(triton.cdiv(a,128),4,g)](raw,den,logz,ids,grid,cache,a,h,r,128,bh,math.log(eps),legacy,cache.dtype==torch.uint8,num_warps=4,enable_fp_fusion=False)
