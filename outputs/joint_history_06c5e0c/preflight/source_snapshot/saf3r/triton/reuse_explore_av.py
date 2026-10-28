"""Weighted group-shared attention and compact edge-salience diagnostics.

The legacy exact evidence path is retained for controlled tests. The refactored
selector uses a 4/32-query probe evidence path, so next-layer History salience no
longer requires a second full selected-QK pass. Neither path materializes a
H x N x selected-key logit tensor.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q,K,V,IDS,BIAS,SPECIAL,O,LSE,
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


def weighted_attention(q,k,v,indices,log_weights,specials,scale):
    h,r,d=q.shape;g,a=indices.shape
    if g!=triton.cdiv(r,32):raise ValueError('Invalid group count')
    if q.stride(1)!=d or q.stride(2)!=1 or not all(t.is_contiguous() for t in (k,v,indices,log_weights,specials)):
        raise ValueError('Contiguous banks required')
    output=torch.empty_like(q);lse=torch.empty((h,r),device=q.device,dtype=torch.float32)
    mass=torch.zeros((g,a),device=q.device,dtype=torch.float32)
    bd=max(16,triton.next_power_of_2(d))
    _attention[(g,h)](q,k,v,indices,log_weights,specials,output,lse,h,r,k.shape[1],a,len(specials),d,
                      q.stride(0),k.stride(0),scale,bd,num_warps=4)
    if a:
        _mass[(g,h,triton.cdiv(a,64))](q,k,indices,log_weights,lse,mass,h,r,a,d,
                                    q.stride(0),k.stride(0),scale,bd,num_warps=4)
    return output,mass


@triton.jit
def _probe_mass(Q,K,IDS,BIAS,LSE,M,
                H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,D:tl.constexpr,
                QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr):
    """Approximate group-mean selected-key mass on the same four probes used by routing."""
    group,head,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    pp=tl.arange(0,4);rr=tl.minimum(group*32+4+pp*8,R-1)
    col=tile*64+tl.arange(0,64);dd=tl.arange(0,BD)
    ids=tl.load(IDS+group*A+col,col<A,-1);valid=(col<A)&(ids>=0)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],dd[None,:]<D,0)
    k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],
              valid[None,:]&(dd[:,None]<D),0)
    bias=tl.load(BIAS+group*A+col,col<A,-float('inf'))
    lse=tl.load(LSE+head*R+rr)
    z=tl.dot(q,k)*SCALE+bias[None,:]
    p=tl.where(valid[None,:],tl.exp(z-lse[:,None]),0.)
    mass=tl.sum(p,0)/(H*4)
    tl.atomic_add(M+group*A+col,mass,valid,sem='relaxed')


def weighted_attention_probe_mass(q,k,v,indices,log_weights,specials,scale):
    """Weighted sparse attention plus low-cost salience evidence.

    The output is exact for the supplied support.  Per-key evidence is estimated
    only on offsets [4,12,20,28] of each 32-query group, reducing the evidence
    QK work from one full repeated sparse pass to 4/32 of that work.
    """
    h,r,d=q.shape;g,a=indices.shape
    if g!=triton.cdiv(r,32):raise ValueError('Invalid group count')
    if q.stride(1)!=d or q.stride(2)!=1 or not all(t.is_contiguous() for t in (k,v,indices,log_weights,specials)):
        raise ValueError('Contiguous banks required')
    output=torch.empty_like(q);lse=torch.empty((h,r),device=q.device,dtype=torch.float32)
    mass=torch.zeros((g,a),device=q.device,dtype=torch.float32)
    bd=max(16,triton.next_power_of_2(d))
    _attention[(g,h)](q,k,v,indices,log_weights,specials,output,lse,h,r,k.shape[1],a,len(specials),d,
                      q.stride(0),k.stride(0),scale,bd,num_warps=4)
    if a:
        _probe_mass[(g,h,triton.cdiv(a,64))](q,k,indices,log_weights,lse,mass,h,r,a,d,
                                             q.stride(0),k.stride(0),scale,bd,num_warps=4)
    return output,mass
