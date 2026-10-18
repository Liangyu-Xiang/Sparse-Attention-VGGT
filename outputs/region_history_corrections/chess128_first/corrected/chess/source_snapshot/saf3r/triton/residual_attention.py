"""Separate indexed QK and cached-logit AV: selected QK is never recomputed."""
import torch
import triton
import triton.language as tl


@triton.jit
def _qk(Q,K,I,L,D:tl.constexpr,COLS:tl.constexpr,ROWS:tl.constexpr,
        QH:tl.constexpr,QS:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,
        BD:tl.constexpr,BK:tl.constexpr):
    row,head,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    ds=tl.arange(0,BD)
    col=tile*BK+tl.arange(0,BK)
    ids=tl.load(I+row*COLS+col,col<COLS,-1)
    valid=(col<COLS)&(ids>=0)
    qq=tl.load(Q+head*QH+row*QS+ds,ds<D,0).to(tl.float32)
    kk=tl.load(K+head*KH+tl.maximum(ids,0)[:,None]*D+ds[None,:],
               valid[:,None]&(ds[None,:]<D),0).to(tl.float32)
    z=tl.where(valid,tl.sum(qq[None,:]*kk,1)*SCALE,-float('inf'))
    tl.store(L+(head*ROWS+row)*COLS+col,z,col<COLS)


@triton.jit
def _av(L,V,I,O,D:tl.constexpr,COLS:tl.constexpr,ROWS:tl.constexpr,
        VH:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    ds=tl.arange(0,BD)
    m=tl.full((),-float('inf'),tl.float32)
    den=tl.full((),0.,tl.float32)
    acc=tl.zeros((BD,),tl.float32)
    for tile in range(tl.cdiv(COLS,BK)):
        col=tile*BK+tl.arange(0,BK)
        ids=tl.load(I+row*COLS+col,col<COLS,-1)
        valid=(col<COLS)&(ids>=0)
        z=tl.load(L+(head*ROWS+row)*COLS+col,valid,-float('inf'))
        new_m=tl.maximum(m,tl.max(z,0))
        correction=tl.where(m==-float('inf'),0.,tl.exp(m-new_m))
        weights=tl.where(valid,tl.exp(z-new_m),0.)
        vv=tl.load(V+head*VH+tl.maximum(ids,0)[:,None]*D+ds[None,:],
                   valid[:,None]&(ds[None,:]<D),0).to(tl.float32)
        acc=acc*correction+tl.sum(weights[:,None]*vv,0)
        den=den*correction+tl.sum(weights,0)
        m=new_m
    tl.store(O+(head*ROWS+row)*D+ds,acc/den,ds<D)


def indexed_qk(q,k,indices,scale):
    h,rows,d=q.shape
    logits=torch.empty((h,rows,indices.shape[1]),device=q.device,dtype=torch.float32)
    _qk[(rows,h,triton.cdiv(indices.shape[1],64))](q,k,indices,logits,d,
        indices.shape[1],rows,q.stride(0),q.stride(1),k.stride(0),scale,
        triton.next_power_of_2(d),64,num_warps=4)
    return logits


def cached_logits_av(logits,v,indices):
    h,rows,cols=logits.shape;d=v.shape[-1]
    out=torch.empty((h,rows,d),device=v.device,dtype=v.dtype)
    _av[(rows,h)](logits,v,indices,out,d,cols,rows,v.stride(0),
        triton.next_power_of_2(d),64,num_warps=4)
    return out
