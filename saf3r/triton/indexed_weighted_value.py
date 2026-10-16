"""AV from already computed selected logits; never recomputes QK."""
import torch
import triton
import triton.language as tl


@triton.jit
def _av(P,I,V,O,R:tl.constexpr,K:tl.constexpr,N:tl.constexpr,D:tl.constexpr,
        BV:tl.constexpr,BD:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    dd=tl.arange(0,BD)
    acc=tl.zeros((BD,),tl.float32)
    for start in range(tl.cdiv(K,BV)):
        kk=start*BV+tl.arange(0,BV)
        ids=tl.load(I+(head*R+row)*K+kk,kk<K,0)
        p=tl.load(P+(head*R+row)*K+kk,kk<K,0)
        v=tl.load(V+head*N*D+ids[:,None]*D+dd[None,:],
                  (kk[:,None]<K)&(dd[None,:]<D),0).to(tl.float32)
        acc+=tl.sum(p[:,None]*v,0)
    tl.store(O+(head*R+row)*D+dd,acc,dd<D)


def indexed_weighted_value(probabilities,indices,values):
    """P/I [H,Qchunk,K], V [H,N,D]; bounded workspace, float accumulation."""
    p=probabilities.float().contiguous();ids=indices.contiguous();v=values.contiguous()
    h,r,k=p.shape
    out=torch.empty((h,r,v.shape[-1]),device=v.device,dtype=v.dtype)
    _av[(r,h)](p,ids,v,out,r,k,v.shape[1],v.shape[2],128,
               triton.next_power_of_2(v.shape[2]),num_warps=4)
    return out
