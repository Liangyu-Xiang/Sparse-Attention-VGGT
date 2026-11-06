"""Reuse the exact BF16 probability tiles already consumed by formal AV.

Masks,compensation,softmax formula and four probes are unchanged. Historical
conditionals are recovered with tile shifts,raw selected LSE and inverse weights.
"""
import torch,triton
import triton.language as tl

@triton.jit
def _bias(W,LOG,INV,N:tl.constexpr,B:tl.constexpr):
 j=tl.program_id(0)*B+tl.arange(0,B);w=tl.load(W+j,j<N,1.);valid=w>0
 tl.store(LOG+j,tl.where(valid,tl.log(w),-float('inf')),j<N);tl.store(INV+j,tl.where(valid,1./w,0.),j<N)

@triton.jit
def _attention(Q,K,V,IDS,SPECIAL,BIAS,INV,OUT,RAW,SHIFT,RAW_LSE,
 H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
 QH:tl.constexpr,KH:tl.constexpr,VH:tl.constexpr,OH:tl.constexpr,LH:tl.constexpr,
 SCALE:tl.constexpr,PROBES:tl.constexpr,CAPTURE:tl.constexpr,FULL:tl.constexpr,
 BD:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,GS:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);g=t//(GS//BM);row=tl.arange(0,BM);rr=t*BM+row;dd=tl.arange(0,BD);kk=tl.arange(0,BK)
 q=tl.load(Q+h*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<R)&(dd[None,:]<D),0.)
 maximum=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
 if CAPTURE:
  if FULL:
   stride=GS//PROBES;tile_in_group=t%(GS//BM);local_row=tile_in_group*BM+row;emit=(BM>=stride)|(tile_in_group*BM%stride==0);slot=(local_row-GS//8)//stride;selected=(local_row>=GS//8)&((local_row-GS//8)%stride==0)
   raw_den=tl.zeros((BM,),tl.float32);last_shift=tl.zeros((BM,),tl.float32)
  else:
   slot=tl.arange(0,PROBES);positions=tl.minimum(slot*(GS//PROBES)+GS//8,R-1);active=positions//BM==t;local=tl.minimum(tl.maximum(positions-t*BM,0),BM-1)
   raw_den=tl.zeros((PROBES,),tl.float32);last_shift=tl.zeros((PROBES,),tl.float32)
 for tile in range(tl.cdiv(S+A,BK)):
  at=tile*BK+kk;key=tl.load(IDS+g*A+at-S,(at>=S)&(at<S+A),-1);special=tl.load(SPECIAL+at,at<S,0);key=tl.where(at<S,special,key);valid=(at<S+A)&(key>=0)
  bias=tl.load(BIAS+g*A+at-S,(at>=S)&valid,0.)
  k=tl.load(K+h*KH+tl.maximum(key,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0.)
  raw=tl.dot(q,k,allow_tf32=False)*SCALE;z=tl.where(valid[None,:],raw+bias[None,:],-float('inf'))
  mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx);alpha=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
  if CAPTURE:
   inverse=tl.load(INV+g*A+at-S,(at>=S)&valid,1.)
   if FULL:
    if emit:
     last_shift=safe
     # Export the BF16 probability tile already consumed by formal AV.
     tl.store(RAW+((g*H+h)*PROBES+slot[:,None])*(S+A)+at[None,:],p.to(V.dtype.element_ty),selected[:,None]&(at[None,:]<S+A))
     tl.store(SHIFT+((g*H+h)*PROBES+slot)*tl.cdiv(S+A,BK)+tile,safe,selected)
   else:
    probe_p=tl.gather(p,tl.broadcast_to(local[:,None],(PROBES,BK)),0);shift=tl.gather(safe,local,0);factor=tl.gather(alpha,local,0);last_shift=shift
    tl.store(RAW+((g*H+h)*PROBES+slot[:,None])*(S+A)+at[None,:],probe_p.to(V.dtype.element_ty),active[:,None]&(at[None,:]<S+A))
    tl.store(SHIFT+((g*H+h)*PROBES+slot)*tl.cdiv(S+A,BK)+tile,shift,active)
  den=den*alpha+tl.sum(p,1)
  v=tl.load(V+h*VH+tl.maximum(key,0)[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0.)
  acc=acc*alpha[:,None]+tl.dot(p.to(V.dtype.element_ty),v,allow_tf32=False)
  if CAPTURE:
   if FULL:
    if emit:raw_den=raw_den*alpha+tl.sum(p*inverse[None,:],1)
   else:raw_den=raw_den*factor+tl.sum(probe_p*inverse[None,:],1)
  maximum=mx
 out=acc/den[:,None]
 tl.store(OUT+h*OH+rr[:,None]*D+dd[None,:],out,(rr[:,None]<R)&(dd[None,:]<D))
 if CAPTURE:
  if FULL:
   if emit:tl.store(RAW_LSE+h*LH+g*PROBES+slot+tl.zeros((BM,),tl.int32),last_shift+tl.log(tl.maximum(raw_den,1.e-37)),selected)
  else:tl.store(RAW_LSE+h*LH+g*PROBES+slot,last_shift+tl.log(tl.maximum(raw_den,1.e-37)),active)

def attention(q,k,v,ids,specials,weights,scale,probes=4,capture=True,warps=4,stages=2,block_queries=64,block_keys=128):
 h,r,d=q.shape;g,a=ids.shape;s=len(specials);assert g==triton.cdiv(r,256) and probes in (1,2,4)
 bias=torch.empty_like(weights);inverse=torch.empty_like(weights);_bias[(triton.cdiv(weights.numel(),256),)](weights,bias,inverse,weights.numel(),256,num_warps=4)
 out=torch.empty_like(q);raw=torch.empty((g,h,probes,s+a),device=q.device,dtype=torch.bfloat16) if capture else out
 shift=torch.empty((g,h,probes,triton.cdiv(s+a,block_keys)),device=q.device) if capture else out
 lse=torch.empty((h,g,probes),device=q.device) if capture else out
 full_rows=r//256*256;segments=[]
 if full_rows:segments.append((0,full_rows,0,full_rows//256,True))
 if full_rows<r:segments.append((full_rows,r,full_rows//256,g,False))
 for first,end,begin_group,end_group,full in segments:
  qq=q[:,first:end];oo=out[:,first:end];ll=lse[:,begin_group:end_group] if capture else lse
  _attention[(triton.cdiv(end-first,block_queries),h)](qq,k,v,ids[begin_group:end_group],specials,bias[begin_group:end_group],inverse[begin_group:end_group],oo,raw[begin_group:end_group] if capture else raw,shift[begin_group:end_group] if capture else shift,ll,h,end-first,k.shape[1],a,s,d,qq.stride(0),k.stride(0),v.stride(0),oo.stride(0),ll.stride(0),scale,probes,capture,full,triton.next_power_of_2(d),block_queries,block_keys,256,num_warps=warps,num_stages=stages)
 return out,dict(encoded=raw,shifts=shift,raw_lse=lse,probes=probes,block_keys=block_keys,probability_tiles=True,inverse_weights=inverse)
