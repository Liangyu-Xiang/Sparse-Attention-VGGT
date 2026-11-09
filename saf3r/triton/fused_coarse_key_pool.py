"""Gather and pool frame-local coarse Key blocks without a dense FP32 copy."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pool(K, PATCHES, STARTS, SIZES, OUT, R:tl.constexpr, D:tl.constexpr,
          KH:tl.constexpr, BT:tl.constexpr, BD:tl.constexpr):
    region=tl.program_id(0)
    head=tl.program_id(1)
    token=tl.arange(0,BT)
    feature=tl.program_id(2)*BD+tl.arange(0,BD)
    begin=tl.load(STARTS+region)
    count=tl.load(SIZES+region).to(tl.int32)
    key=tl.load(PATCHES+begin+token,token<count,0)
    values=tl.load(K+head*KH+key[:,None]*D+feature[None,:],
                   (token[:,None]<count)&(feature[None,:]<D),0.).to(tl.float32)
    pooled=tl.sum(values,0)/count
    tl.store(OUT+(head*R+region)*D+feature,pooled,feature<D)


def pool(k,patches,sizes,max_block,*,feature_block=32):
    h,_,d=k.shape
    r=len(sizes)
    counts=sizes.to(torch.int32)
    starts=counts.cumsum(0,dtype=torch.int32)-counts
    pooled=torch.empty((h,r,d),device=k.device,dtype=torch.float32)
    _pool[(r,h,triton.cdiv(d,feature_block))](
        k,patches,starts,counts,pooled,r,d,k.stride(0),
        triton.next_power_of_2(max_block),feature_block,num_warps=4)
    return pooled
