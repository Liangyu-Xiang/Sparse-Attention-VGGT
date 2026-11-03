"""Indexed Flash-style QK/AV; contiguous support tiles, no Grid execution.

Mask sharing remains 32 queries. Probe raw logits and scalar selected/special
log-sums are exported by the SAME QK. No selected-edge QK is repeated.
"""
import torch,triton
import triton.language as tl
from .reuse_split_fused_history import _merge

@triton.jit
def _attention(Q,K,V,IDS,BIAS,SPECIAL,O,LSE,RAW,RLSE,SLSE,
 H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
 QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,PARTS:tl.constexpr,
 PER:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,EVIDENCE:tl.constexpr,P:tl.constexpr,RAW_ONLY:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);part=tl.program_id(2)
 row=tl.arange(0,32);rr=g*32+row;d=tl.arange(0,BD);j=tl.arange(0,BK);p=tl.arange(0,P)
 probe=tl.minimum(p*8+4 if P==4 else p*0+16,R-g*32-1)
 q=tl.load(Q+h*QH+rr[:,None]*D+d[None,:],(rr[:,None]<R)&(d[None,:]<D),0.)
 maximum=tl.full((32,),-float('inf'),tl.float32);den=tl.zeros((32,),tl.float32);acc=tl.zeros((32,BD),tl.float32)
 for bank in tl.static_range(2):
  rawmax=tl.full((P,),-float('inf'),tl.float32);rawden=tl.zeros((P,),tl.float32)
  for tile in range((tl.cdiv(S,BK) if part==0 else 0) if bank==0 else PER):
   col=tile*BK+j if bank==0 else (part*PER+tile)*BK+j
   if bank==0:
    ids=tl.load(SPECIAL+col,col<S,0);valid=col<S;bias=tl.full((BK,),0.,tl.float32)
   else:
    ids=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(ids>=0);bias=tl.load(BIAS+g*A+col,col<A,-float('inf'))
   k=tl.load(K+h*KH+tl.maximum(ids,0)[None,:]*D+d[:,None],valid[None,:]&(d[:,None]<D),0.)
   v=tl.load(V+h*KH+tl.maximum(ids,0)[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0.)
   dot=tl.dot(q,k,allow_tf32=False)*SCALE
   if EVIDENCE:
    zraw=tl.where(valid[None,:],dot,-float('inf'));value=tl.gather(zraw,tl.broadcast_to(probe[:,None],(P,BK)),0)
    if not RAW_ONLY or bank==0:
     mx=tl.maximum(rawmax,tl.max(value,1));safe=tl.where(mx==-float('inf'),0.,mx)
     rawden=rawden*tl.exp(rawmax-safe)+tl.sum(tl.exp(value-safe[:,None]),1);rawmax=mx
    if bank==1:tl.store(RAW+((g*H+h)*P+p[:,None])*A+col[None,:],value,col[None,:]<A)
   z=tl.where(valid[None,:],dot+bias[None,:],-float('inf'));mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
   correction=tl.exp(maximum-safe);prob=tl.exp(z-safe[:,None])
   acc=acc*correction[:,None]+tl.dot(prob.to(v.dtype),v,allow_tf32=False);den=den*correction+tl.sum(prob,1);maximum=mx
  if EVIDENCE:
   if bank==0:
    if part==0:tl.store(SLSE+(g*H+h)*P+p,rawmax+tl.log(rawden))
   else:
    if not RAW_ONLY:tl.store(RLSE+((g*H+h)*PARTS+part)*P+p,rawmax+tl.log(rawden))
 tl.store(O+(((g*H+h)*PARTS+part)*32+row[:,None])*BD+d[None,:],tl.where(den[:,None]>0,acc/tl.maximum(den[:,None],1.e-30),0.),d[None,:]<D)
 tl.store(LSE+((g*H+h)*PARTS+part)*32+row,maximum+tl.log(den))

def weighted_attention(q,k,v,indices,log_weights,specials,scale,parts=4,block_keys=64,stages=1,warps=4,emit_evidence=True,probe_count=4,fast_evidence=False):
 h,r,d=q.shape;g,a=indices.shape
 if g!=triton.cdiv(r,32):raise ValueError('32-query shared mask required')
 if not all(t.is_contiguous() for t in [k,v,indices,log_weights,specials]):raise ValueError('Contiguous banks required')
 bd=triton.next_power_of_2(d);output=torch.empty_like(q);lse=torch.empty((h,r),device=q.device)
 tmp=torch.empty((g,h,parts,32,bd),device=q.device);tlse=torch.empty((g,h,parts,32),device=q.device)
 raw=torch.empty((g,h,probe_count,a),device=q.device) if emit_evidence else lse
 rawlse=torch.empty((g,h,parts,probe_count),device=q.device) if emit_evidence else lse
 speciallse=torch.empty((g,h,probe_count),device=q.device) if emit_evidence else lse
 _attention[(g,h,parts)](q,k,v,indices,log_weights,specials,tmp,tlse,raw,rawlse,speciallse,h,r,a,len(specials),d,q.stride(0),k.stride(0),scale,parts,triton.cdiv(triton.cdiv(a,block_keys),parts),bd,block_keys,emit_evidence,probe_count,fast_evidence,num_warps=warps,num_stages=stages)
 _merge[(g,h,32)](tmp,tlse,output,lse,h,r,d,bd,parts,triton.next_power_of_2(parts),num_warps=4)
 return output,dict(raw=raw,selected_lse=(raw.logsumexp(-1) if fast_evidence else rawlse.logsumexp(2)) if emit_evidence else None,special_lse=speciallse)
