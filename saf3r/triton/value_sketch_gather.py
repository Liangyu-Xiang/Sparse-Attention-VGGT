"""Gather patch sketches from the original FP32 projection."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gather(PROJECTED, PATCHES, OUT, N:tl.constexpr, T:tl.constexpr,
            H:tl.constexpr, S:tl.constexpr, B:tl.constexpr, C:tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B)
    col=tl.arange(0,C)
    key=tl.load(PATCHES+row,row<N,0)
    offset=((col[None,:]//S)*T+key[:,None])*S+col[None,:]%S
    valid=(row[:,None]<N)&(col[None,:]<H*S)
    projected=tl.load(PROJECTED+offset,valid,0.)
    tl.store(OUT+row[:,None]*(H*S)+col[None,:],projected,valid)


def gather(v,matrix,patches):
    projected=torch.bmm(v.float(),matrix)
    h,t,s=projected.shape
    n=len(patches)
    out=torch.empty((n,h*s),device=v.device,dtype=projected.dtype)
    _gather[(triton.cdiv(n,32),)](
        projected,patches,out,n,t,h,s,32,triton.next_power_of_2(h*s),num_warps=4)
    return out
