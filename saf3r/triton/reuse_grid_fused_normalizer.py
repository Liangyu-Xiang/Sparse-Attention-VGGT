"""Combine attention's raw per-grid selected LSE with unobserved mass."""
import torch,triton
import triton.language as tl

@triton.jit
def _finish(RAW,BASE,OFF,SIZE,DEN,ERROR,H:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);hp=tl.program_id(1);r=tl.program_id(2)*B+tl.arange(0,B)
    a=tl.load(RAW+(g*H*4+hp)*R+r,r<R,-float('inf'));b=tl.load(BASE+(g*H*4+hp)*R+r,r<R,0.)
    begin=tl.load(OFF+g*(R+1)+r,r<R,0);end=tl.load(OFF+g*(R+1)+r+1,r<R,0);n=tl.load(SIZE+r,r<R,0.)
    remaining=n-(end-begin);missing=tl.where(remaining>0,b+tl.log(remaining),-float('inf'))
    maximum=tl.maximum(a,missing);total=tl.exp(a-maximum)+tl.exp(missing-maximum);d=maximum+tl.log(total)
    tl.store(DEN+(g*H*4+hp)*R+r,d,r<R)
    tl.store(ERROR+(g*H*4+hp)*R+r,tl.abs(tl.exp(maximum-d)*total-1.),r<R)

def finish(raw,base,offsets,sizes):
    g,h,_,r=raw.shape;den=torch.empty_like(raw);error=torch.empty_like(raw)
    _finish[(g,h*4,triton.cdiv(r,256))](raw,base,offsets,sizes,den,error,h,r,256,num_warps=4)
    return den,error
