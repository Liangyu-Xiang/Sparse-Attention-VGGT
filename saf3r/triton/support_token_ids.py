"""Map packed patch IDs to token IDs while preserving invalid slots."""
import torch
import triton
import triton.language as tl


@triton.jit
def _map(IDS, PATCHES, OUT, N:tl.constexpr, B:tl.constexpr):
    slot=tl.program_id(0)*B+tl.arange(0,B)
    patch=tl.load(IDS+slot,slot<N,-1)
    token=tl.load(PATCHES+patch,patch>=0,-1)
    tl.store(OUT+slot,token,slot<N)


def map_support(ids,patches):
    out=torch.empty(ids.shape,device=ids.device,dtype=torch.int32)
    _map[(triton.cdiv(ids.numel(),512),)](ids,patches,out,ids.numel(),512,num_warps=4)
    return out
