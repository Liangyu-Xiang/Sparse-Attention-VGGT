"""Two FP32 operations with the SAME intervening rounding as LayerScale + add."""
import torch,triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _add(X,Y,S,O,N:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,0.).to(tl.float32)
    y=tl.load(Y+i,i<N,0.).to(tl.float32)
    scale=tl.load(S+i%D)
    product=libdevice.mul_rn(y,scale)
    result=libdevice.add_rn(x,product)
    tl.store(O+i,result,i<N)

def scaled_add(x,y,scale):
    if x.shape!=y.shape or not x.is_contiguous() or not y.is_contiguous() or scale.dtype!=torch.float32:
        return x+y*scale
    # A FP32 scale promotes the multiply and add to FP32 in the native path.
    out=torch.empty_like(x,dtype=torch.float32)
    _add[(triton.cdiv(x.numel(),1024),)](x,y,scale,out,x.numel(),x.shape[-1],1024,enable_fp_fusion=False)
    return out
