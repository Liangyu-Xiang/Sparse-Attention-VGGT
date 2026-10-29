"""Weighted group-shared attention and compact edge-mass diagnostic.

The evidence kernel repeats selected QK once; this explicit prototype overhead
is counted in timings. Never materializes H x N x selected-key logits.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q,K,V,IDS,BIAS,SPECIAL,O,LSE,
               LOGITS,SAVE:tl.constexpr,POOL,GRID,DELTA,RELATIVE:tl.constexpr,G:tl.constexpr,
               H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,A:tl.constexpr,S:tl.constexpr,
               D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr):
    group,head=tl.program_id(0),tl.program_id(1)
    rr=group*32+tl.arange(0,32);dd=tl.arange(0,BD);cc=tl.arange(0,64)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<R)&(dd[None,:]<D),0)
    maximum=tl.full((32,),-float('inf'),tl.float32)
    den=tl.zeros((32,),tl.float32);acc=tl.zeros((32,BD),tl.float32)
    for bank in tl.static_range(2):
        for tile in range(tl.cdiv(S,64) if bank==0 else tl.cdiv(A,64)):
            col=tile*64+cc
            if bank==0:
                ids=tl.load(SPECIAL+col,col<S,0);valid=col<S;bias=tl.full((64,),0,tl.float32)
            else:
                ids=tl.load(IDS+group*A+col,col<A,-1);valid=(col<A)&(ids>=0)
                bias=tl.load(BIAS+group*A+col,col<A,-float('inf'))
            k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],
                      valid[None,:]&(dd[:,None]<D),0)
            v=tl.load(V+head*KH+tl.maximum(ids,0)[:,None]*D+dd[None,:],
                      valid[:,None]&(dd[None,:]<D),0)
            z=tl.where(valid[None,:],tl.dot(q,k)*SCALE+bias[None,:],-float('inf'))
            if RELATIVE and bank==1:
                # Compact raw evidence: one scalar / group / selected Key.
                # No H x query x support logit bank is written to memory.
                region=tl.load(GRID+tl.maximum(ids,0),valid,0)
                pooled=tl.load(POOL+head*G*D+region[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
                qmean=tl.sum(q.to(tl.float32),0)/tl.minimum(32,R-group*32)
                reference=tl.sum(pooled.to(tl.float32)*qmean[None,:],1)*SCALE
                raw=tl.sum(tl.where(rr[:,None]<R,z-bias[None,:],0.),0)/tl.minimum(32,R-group*32)
                tl.atomic_add(DELTA+group*A+col,(raw-reference)/H,valid,sem='relaxed')
            if SAVE and bank==1:
                tl.store(LOGITS+head*R*A+rr[:,None]*A+col[None,:],z,(rr[:,None]<R)&(col[None,:]<A))
            mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
            correction=tl.exp(maximum-safe)
            p=tl.exp(z-safe[:,None])
            acc=acc*correction[:,None]+tl.dot(p.to(v.dtype),v)
            den=den*correction+tl.sum(p,1);maximum=mx
    tl.store(O+head*R*D+rr[:,None]*D+dd[None,:],acc/den[:,None],(rr[:,None]<R)&(dd[None,:]<D))
    tl.store(LSE+head*R+rr,maximum+tl.log(den),rr<R)


@triton.jit
def _mass(Q,K,IDS,BIAS,LSE,M,H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,D:tl.constexpr,
          QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr):
    group,head,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    rr=group*32+tl.arange(0,32);col=tile*64+tl.arange(0,64);dd=tl.arange(0,BD)
    ids=tl.load(IDS+group*A+col,col<A,-1);valid=(col<A)&(ids>=0)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<R)&(dd[None,:]<D),0)
    k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
    bias=tl.load(BIAS+group*A+col,col<A,-float('inf'))
    lse=tl.load(LSE+head*R+rr,rr<R,0.)
    z=tl.dot(q,k)*SCALE+bias[None,:]
    p=tl.where((rr[:,None]<R)&valid[None,:],tl.exp(z-lse[:,None]),0.)
    mass=tl.sum(p,0)/(H*tl.minimum(32,R-group*32))
    tl.atomic_add(M+group*A+col,mass,valid,sem='relaxed')


def weighted_attention(q,k,v,indices,log_weights,specials,scale,single_qk=False,relative_reference=None):
    h,r,d=q.shape;g,a=indices.shape
    if g!=triton.cdiv(r,32):raise ValueError('Invalid group count')
    if q.stride(1)!=d or q.stride(2)!=1 or not all(t.is_contiguous() for t in (k,v,indices,log_weights,specials)):
        raise ValueError('Contiguous banks required')
    output=torch.empty_like(q);lse=torch.empty((h,r),device=q.device,dtype=torch.float32)
    mass=torch.zeros((g,a),device=q.device,dtype=torch.float32)
    logits=torch.empty((h,r,a),device=q.device,dtype=torch.float32) if single_qk else lse
    bd=max(16,triton.next_power_of_2(d))
    relative=relative_reference is not None
    pooled,grid=relative_reference if relative else (k,indices)
    delta=torch.zeros_like(mass) if relative else mass
    _attention[(g,h)](q,k,v,indices,log_weights,specials,output,lse,logits,single_qk,pooled,grid,delta,relative,pooled.shape[1],h,r,k.shape[1],a,len(specials),d,
                      q.stride(0),k.stride(0),scale,bd,num_warps=4)
    if a and single_qk:
        _saved_mass[(g,h,triton.cdiv(a,64))](logits,indices,lse,mass,h,r,a,num_warps=4)
    elif a:
        _mass[(g,h,triton.cdiv(a,64))](q,k,indices,log_weights,lse,mass,h,r,a,d,
                                    q.stride(0),k.stride(0),scale,bd,num_warps=4)
    return (output,mass,delta) if relative else (output,mass)


@triton.jit
def _saved_mass(LOGITS,IDS,LSE,M,H:tl.constexpr,R:tl.constexpr,A:tl.constexpr):
    group,head,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    rr=group*32+tl.arange(0,32);col=tile*64+tl.arange(0,64)
    ids=tl.load(IDS+group*A+col,col<A,-1);valid=(col<A)&(ids>=0)
    z=tl.load(LOGITS+head*R*A+rr[:,None]*A+col[None,:],(rr[:,None]<R)&(col[None,:]<A),-float('inf'))
    lse=tl.load(LSE+head*R+rr,rr<R,0.)
    p=tl.where((rr[:,None]<R)&valid[None,:],tl.exp(z-lse[:,None]),0.)
    mass=tl.sum(p,0)/(H*tl.minimum(32,R-group*32))
    tl.atomic_add(M+group*A+col,mass,valid,sem='relaxed')
