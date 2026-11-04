"""Grid-free sparse Flash-style AV; compact history A/B exports are optional.

Q=[H,R,D], K/V=[H,N,D], IDs=[G,A], G=ceil(R/32). Bias is already
computed by the selector. Probe positions are local-window query indices.
No Grid, routing, sorting, mass transfer or Head-specific mask is accepted.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _attention(Q,K,V,IDS,SPECIAL,BIAS,POS,OUT,LSE,TILE_P,TILE_LSE,
 H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
 QH:tl.constexpr,KH:tl.constexpr,VH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,
 HAS_BIAS:tl.constexpr,INLINE:tl.constexpr,P:tl.constexpr,T:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);row=tl.arange(0,32);rr=g*32+row;d=tl.arange(0,BD);col=tl.arange(0,BK)
 q=tl.load(Q+h*QH+rr[:,None]*D+d[None,:],(rr[:,None]<R)&(d[None,:]<D),0.)
 maximum=tl.full((32,),-float('inf'),tl.float32);den=tl.zeros((32,),tl.float32);acc=tl.zeros((32,BD),tl.float32)
 for tile in range(tl.cdiv(S+A,BK)):
  at=tile*BK+col
  special=tl.load(SPECIAL+at,at<S,0);key=tl.load(IDS+g*A+at-S,(at>=S)&(at<S+A),-1);key=tl.where(at<S,special,key)
  valid=(at<S+A)&(key>=0);bias=tl.full((BK,),0.,tl.float32)
  if HAS_BIAS:bias=tl.load(BIAS+g*A+at-S,(at>=S)&valid,0.)
  k=tl.load(K+h*KH+tl.maximum(key,0)[None,:]*D+d[:,None],valid[None,:]&(d[:,None]<D),0.)
  v=tl.load(V+h*VH+tl.maximum(key,0)[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0.)
  z=tl.where(valid[None,:],tl.dot(q,k,allow_tf32=False)*SCALE+bias[None,:],-float('inf'))
  mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx);alpha=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
  acc=acc*alpha[:,None]+tl.dot(p.to(v.dtype),v,allow_tf32=False);den=den*alpha+tl.sum(p,1);maximum=mx

 result=acc/tl.where(den>0,den,1.)[:,None]
 tl.store(OUT+(h*R+rr[:,None])*D+d[None,:],result,(rr[:,None]<R)&(d[None,:]<D))
 tl.store(LSE+h*R+rr,maximum+tl.log(den),rr<R)

def direct_sparse_attention(q,k,v,sparse_key_ids,special_key_ids,scale,logit_bias=None,probe_positions=None,return_history_probability=False,*,block_keys=128,num_warps=4,num_stages=2):
 if return_history_probability:raise NotImplementedError('Probability exports are a separate phase')
 h,r,d=q.shape;g,a=sparse_key_ids.shape;s=len(special_key_ids)
 if g!=triton.cdiv(r,32):raise ValueError('One sparse support per32 queries required')
 out=torch.empty((h,r,d),device=q.device,dtype=q.dtype);lse=torch.empty((h,r),device=q.device,dtype=torch.float32)
 _attention[(g,h)](q,k,v,sparse_key_ids,special_key_ids,logit_bias if logit_bias is not None else lse,lse,out,lse,lse,lse,h,r,k.shape[1],a,s,d,q.stride(0),k.stride(0),v.stride(0),scale,max(16,triton.next_power_of_2(d)),block_keys,logit_bias is not None,False,4,triton.cdiv(s+a,block_keys),num_warps=num_warps,num_stages=num_stages)
 return out,dict(lse=lse)
