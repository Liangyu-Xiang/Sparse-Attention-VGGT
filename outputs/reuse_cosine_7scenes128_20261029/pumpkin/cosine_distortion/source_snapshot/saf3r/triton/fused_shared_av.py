"""Tensor-core attention over a common Value bank and query-block token indices.

One online normalization, no retained token logits. Optional tile log-masses
are a compressed *history approximation*, not per-token confidence scores.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _run(Q,K,V,PK,PV,IX,O,E,LSE,BIAS,WEIGHTED:tl.constexpr, N:tl.constexpr,R:tl.constexpr,P:tl.constexpr,
         A:tl.constexpr,D:tl.constexpr,VC:tl.constexpr,QH:tl.constexpr,QS:tl.constexpr,
         KH:tl.constexpr,PH:tl.constexpr,SCALE:tl.constexpr,BQ:tl.constexpr,
         BK:tl.constexpr,BD:tl.constexpr,SLOTS:tl.constexpr,EVIDENCE:tl.constexpr,
         QN:tl.constexpr,QSTART,QOFFSET,SHIFTED:tl.constexpr):
    group,head=tl.program_id(0),tl.program_id(1)
    qr=group*BQ+tl.arange(0,BQ);dd=tl.arange(0,BD);cc=tl.arange(0,BK)
    qi=(QSTART+qr+QOFFSET)%QN if SHIFTED else qr
    qq=tl.load(Q+head*QH+qi[:,None]*QS+dd[None,:],(qr[:,None]<R)&(dd[None,:]<D),0)
    maximum=tl.full((BQ,),-float('inf'),tl.float32)
    den=tl.zeros((BQ,),tl.float32);acc=tl.zeros((BQ,BD),tl.float32)
    # Compile two independent pipelinable loops. A dynamic bank branch in
    # every key iteration prevents useful load/dot software pipelining.
    for bank in tl.static_range(2):
        for slot in range(tl.cdiv(P,BK) if bank==0 else tl.cdiv(A,BK)):
            col=slot*BK+cc
            if bank==0:
                valid=col<P
                kk=tl.load(PK+head*PH+dd[:,None]+col[None,:]*D,
                           (dd[:,None]<D)&valid[None,:],0)
                vv=tl.load(PV+head*PH+col[:,None]*D+dd[None,:],
                           valid[:,None]&(dd[None,:]<D),0)
                patch=valid&(col<VC)
            else:
                ids=tl.load(IX+group*A+col,col<A,-1)
                valid=(col<A)&(ids>=0)&(ids<N);patch=valid
                kk=tl.load(K+head*KH+dd[:,None]+tl.maximum(ids,0)[None,:]*D,
                           (dd[:,None]<D)&valid[None,:],0)
                vv=tl.load(V+head*KH+tl.maximum(ids,0)[:,None]*D+dd[None,:],
                           valid[:,None]&(dd[None,:]<D),0)
            z=tl.dot(qq,kk)*SCALE
            if bank==0 and WEIGHTED:
                bias=tl.load(BIAS+group*P+col,col<P,0.)
                z+=bias[None,:]
            z=tl.where(valid[None,:],z,-float('inf'))
            local_max=tl.max(z,1);new_max=tl.maximum(maximum,local_max)
            safe=tl.where(new_max==-float('inf'),0.,new_max)
            correction=tl.exp(maximum-safe)
            w=tl.exp(z-safe[:,None])
            acc=acc*correction[:,None]+tl.dot(w.to(vv.dtype),vv)
            den=den*correction+tl.sum(w,1);maximum=new_max
            if EVIDENCE:
                mass=tl.sum(tl.where(patch[None,:],w,0.),1)
                logmass=tl.where(mass>0,tl.log(mass)+safe,-float('inf'))
                tl.store(E+(head*R+qr)*SLOTS+bank*tl.cdiv(P,BK)+slot,logmass,qr<R)
    tl.store(O+(head*R+qr[:,None])*D+dd[None,:],acc/den[:,None],
             (qr[:,None]<R)&(dd[None,:]<D))
    if EVIDENCE:tl.store(LSE+head*R+qr,maximum+tl.log(den),qr<R)


def shared_av(q,k,v,packed_k,packed_v,indices,value_count,*,scale=None,
              query_block=32,key_tile=64,evidence=False,query_start=0,query_rows=None,query_offset=0,
              common_log_weights=None):
    """Q[H,R,D], KV[H,N,D], common bank[H,P,D], IX[ceil(R/BQ),A].

    Bank order: actual Value representative tokens, then specials. Attention
    indices must be unique per group and disjoint from the bank. -1 is padding.
    Callers stream bounded Query windows when requesting history evidence.
    Optional query_start/query_rows/query_offset read a cyclic logical Query
    window directly from the full Q bank; K/V indices are never permuted.
    """
    h,qn,d=q.shape;r=qn if query_rows is None else query_rows
    p=packed_k.shape[1];a=indices.shape[1]
    if not 0<=query_offset<qn or not 0<=query_start<qn or not 0<r<=qn-query_start:
        raise ValueError('Invalid cyclic Query window')
    if query_block!=32 or key_tile!=64:raise ValueError('Supported tile is32x64')
    if indices.shape[0]!=triton.cdiv(r,query_block):raise ValueError('Invalid Query-group index count')
    if not 0<=value_count<=p:raise ValueError('Invalid Value bank size')
    if p==0:raise ValueError('Common bank must include special Keys')
    if common_log_weights is not None:
        if common_log_weights.shape!=(indices.shape[0],p) or not common_log_weights.is_contiguous():
            raise ValueError('Common log weights must be contiguous [Query groups, common Keys]')
        if common_log_weights.device!=q.device or common_log_weights.dtype!=torch.float32:
            raise ValueError('Common log weights require device-matched FP32')
    if q.dtype not in (torch.bfloat16,torch.float16):raise ValueError('BF16/FP16 Tensor-core path only')
    if not (k.is_contiguous() and v.is_contiguous() and packed_k.is_contiguous() and packed_v.is_contiguous()):
        raise ValueError('KV banks must be contiguous')
    slots=triton.cdiv(p,key_tile)+triton.cdiv(a,key_tile)
    out=torch.empty((h,r,d),device=q.device,dtype=q.dtype)
    e=torch.empty((h,r,slots) if evidence else (0,),device=q.device,dtype=torch.float32)
    lse=torch.empty((h,r) if evidence else (0,),device=q.device,dtype=torch.float32)
    _run[(indices.shape[0],h)](q,k,v,packed_k,packed_v,indices,out,e,lse,
        common_log_weights if common_log_weights is not None else q,common_log_weights is not None,
        k.shape[1],r,p,a,d,value_count,q.stride(0),q.stride(1),k.stride(0),packed_k.stride(0),
        d**-.5 if scale is None else scale,query_block,key_tile,max(16,triton.next_power_of_2(d)),slots,evidence,
        qn,query_start,query_offset,bool(query_start or query_offset),
        num_warps=4,num_stages=2)
    return out,e,lse
