"""GPU-only budget completion after partial Top-K. No host scalar fallback."""
import torch
import triton
import triton.language as tl


@triton.jit
def _finish(KEYS,AV,NEED,TOP,O,FLAGS, P:tl.constexpr,K:tl.constexpr,BP:tl.constexpr,BK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BK)
    ids=tl.load(TOP+row*K+col,col<K,0)
    count=tl.load(AV+row*P+ids,col<K,0)
    before=tl.cumsum(count,0)-count;target=tl.load(NEED+row)
    chosen=(col<K)&(count>0)&(before<target)
    tl.store(O+row*P+col,ids,chosen)
    covered=tl.sum(count,0);cursor=tl.sum(chosen.to(tl.int32),0)
    fallback=covered<target
    if fallback:
        pc=tl.arange(0,BP)
        key=tl.load(KEYS+row*P+pc,pc<P,-1)
        cut=tl.load(KEYS+row*P+tl.load(TOP+row*K+K-1))
        remain=tl.where((pc<P)&(key<cut),key,-1)
        live=tl.full((),True,tl.int1)
        while (covered<target)&(cursor<P)&live:
            best=tl.max(remain,0)
            live=best>=0
            if live:
                parent=P-1-(best&4294967295).to(tl.int32)
                tl.store(O+row*P+cursor,parent)
                covered+=tl.load(AV+row*P+parent);cursor+=1
                remain=tl.where(pc==parent,-1,remain)
    tl.store(FLAGS+row*2,fallback.to(tl.int32))
    tl.store(FLAGS+row*2+1,(covered<target).to(tl.int32))


def partial_parent_prefix(scores,available,missing,k):
    p=scores.shape[1];k=min(k,p)
    # Probabilities are nonnegative float32. Encode exact lexicographic order:
    # descending score, then ascending Parent ID. No arbitrary boundary ties.
    bits=scores.contiguous().view(torch.int32).to(torch.int64)
    key=(bits<<32)+(p-1-torch.arange(p,device=scores.device,dtype=torch.int64))
    key=key.masked_fill(available==0,-1).contiguous()
    top=key.topk(k,dim=1,largest=True,sorted=True).indices.contiguous()
    out=torch.full_like(available,-1,dtype=torch.int32)
    flags=torch.empty((len(scores),2),device=scores.device,dtype=torch.int32)
    _finish[(len(scores),)](key,available.contiguous(),missing,top,out,flags,p,k,
        triton.next_power_of_2(p),triton.next_power_of_2(k),num_warps=4)
    return out.long(),out>=0,dict(partial_topk_initial=k,partial_fallback_groups=flags[:,0].sum(),
        partial_unfilled_groups=flags[:,1].sum())
