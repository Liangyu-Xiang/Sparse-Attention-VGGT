"""Boundary-only probe scoring and streaming whole-region member expansion."""
import torch
import triton
import triton.language as tl


@triton.jit
def _boundary(Q,K,POP,IX,ACTIVE,O, N:tl.constexpr,D:tl.constexpr,H:tl.constexpr,
              QH:tl.constexpr,KH:tl.constexpr,C:tl.constexpr,START,
              SCALE:tl.constexpr,BD:tl.constexpr,OFFSET):
    group=tl.program_id(0);rr=tl.arange(0,16);cc=tl.arange(0,16);dd=tl.arange(0,BD)
    if tl.load(ACTIVE+group):
        qr=(tl.minimum((START+group)*32+4+rr*8,N-1)+OFFSET)%N
        ids=tl.load(IX+group*C+cc,cc<C,0)
        pop=tl.load(POP+ids,cc<C,0)
        acc=tl.zeros((16,),tl.float32)
        for head in range(H):
            qq=tl.load(Q+head*QH+qr[:,None]*D+dd[None,:],(rr[:,None]<4)&(dd[None,:]<D),0)
            kk=tl.load(K+head*KH+ids[None,:]*D+dd[:,None],(cc[None,:]<C)&(dd[:,None]<D),0)
            z=tl.dot(qq,kk)*SCALE+pop[None,:]
            z=tl.where(cc[None,:]<C,z,-float('inf'))
            w=tl.exp(z-tl.max(z,1)[:,None]);w=w/tl.sum(w,1)[:,None]
            acc+=tl.sum(tl.where(rr[:,None]<4,w,0.),0)
        tl.store(O+group*C+cc,acc/(4*H),cc<C)
    else:tl.store(O+group*C+cc,0.,cc<C)


def boundary_probabilities(q,keys,log_population,candidates,active,start_group,scale,query_offset=0):
    groups,width=candidates.shape
    if width>16:raise ValueError('Boundary fanout must be <=16')
    if not(q.is_contiguous() and keys.is_contiguous()):raise ValueError('Contiguous scoring banks required')
    out=torch.empty((groups,width),device=q.device,dtype=torch.float32)
    _boundary[(groups,)](q,keys,log_population,candidates.contiguous(),active,out,q.shape[1],q.shape[2],q.shape[0],
        q.stride(0),keys.stride(0),width,start_group,scale,max(16,triton.next_power_of_2(q.shape[2])),query_offset,num_warps=4)
    return out


@triton.jit
def _expand(HISTORY,ENDS,STARTS,BANK,IS_VALUE,SEEN,OUT,COUNTS,
            N:tl.constexpr,HC:tl.constexpr,S:tl.constexpr,A:tl.constexpr,RAW:tl.constexpr,
            SEARCH:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);col=tl.arange(0,B);cursor=tl.full((),0,tl.int32)
    for block in range(tl.cdiv(HC,B)):
        j=block*B+col
        ids=tl.load(HISTORY+group*HC+j,j<HC,-1)
        valid=(j<HC)&(ids>=0)
        prefix=tl.cumsum(valid.to(tl.int32),0)-1
        tl.store(OUT+group*A+cursor+prefix,ids,valid)
        cursor+=tl.sum(valid.to(tl.int32),0)
    total=tl.load(ENDS+group*S+S-1)
    for block in range(tl.cdiv(RAW,B)):
        if block*B<total:
            rank=block*B+col;valid=rank<total
            lo=tl.full((B,),0,tl.int32);hi=tl.full((B,),S,tl.int32)
            for _ in range(SEARCH):
                mid=(lo+hi)//2
                end=tl.load(ENDS+group*S+mid,mid<S,2147483647)
                move=end<=rank
                lo=tl.where(move,mid+1,lo);hi=tl.where(move,hi,mid)
            slot=tl.minimum(lo,S-1)
            start=tl.load(STARTS+group*S+slot)
            previous=tl.load(ENDS+group*S+slot-1,slot>0,0)
            ids=tl.load(BANK+start+rank-previous,valid,-1)
            safe=tl.maximum(ids,0)
            seen=tl.load(SEEN+group*(N+1)+safe,valid&(ids>=0),0)
            value=tl.load(IS_VALUE+safe,valid&(ids>=0),0)
            valid=valid&(ids>=0)&(seen==0)&(value==0)
            prefix=tl.cumsum(valid.to(tl.int32),0)-1
            tl.store(OUT+group*A+cursor+prefix,ids,valid&((cursor+prefix)<A))
            cursor+=tl.sum(valid.to(tl.int32),0)
    tl.store(COUNTS+group,cursor)


def expand_member_regions(history,starts,lengths,bank,is_value,width,raw_capacity):
    """Bounded Gwindow×N uint8 history lookup, NOT a Token×Token output Mask.

    Region descriptors are disjoint by construction. Remove only Keys already
    in the common Value bank or history, then concatenate all remaining members.
    """
    groups=len(starts);n=len(is_value)
    seen=torch.zeros((groups,n+1),device=bank.device,dtype=torch.uint8)
    if history.shape[1]:
        target=torch.where(history>=0,history.long(),n)
        seen.scatter_(1,target,1)
    ends=lengths.cumsum(1).contiguous();starts=starts.contiguous()
    out=torch.full((groups,width),-1,device=bank.device,dtype=torch.int32)
    counts=torch.empty(groups,device=bank.device,dtype=torch.long)
    _expand[(groups,)](history,ends,starts,bank,is_value,seen,out,counts,n,history.shape[1],starts.shape[1],
        width,raw_capacity,starts.shape[1].bit_length()+1,1024,num_warps=4)
    return out,counts
