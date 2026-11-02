"""Group-shared sparse support with128-query Attention tiles; exact per-query softmax."""
import torch,triton
import triton.language as tl

@triton.jit
def _attention(Q,K,V,IDS,BIAS,SPECIAL,O,LSE,
 NQ:tl.constexpr,NK:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,GS:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
 block=tl.program_id(0);head=tl.program_id(1);group=block//(GS//BM)
 rr=block*BM+tl.arange(0,BM);cc=tl.arange(0,BN);dd=tl.arange(0,BD)
 q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<NQ)&(dd[None,:]<D),0.)
 maximum=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
 for bank in tl.static_range(2):
  for tile in range(tl.cdiv(S,BN) if bank==0 else tl.cdiv(A,BN)):
   col=tile*BN+cc
   if bank==0:
    ids=tl.load(SPECIAL+col,col<S,0);valid=col<S;bias=tl.full((BN,),0.,tl.float32)
   else:
    ids=tl.load(IDS+group*A+col,col<A,-1);valid=(col<A)&(ids>=0);bias=tl.load(BIAS+group*A+col,col<A,-float('inf'))
   k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0.)
   v=tl.load(V+head*KH+tl.maximum(ids,0)[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0.)
   z=tl.where(valid[None,:],tl.dot(q,k)*SCALE+bias[None,:],-float('inf'))
   mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx);alpha=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
   acc=acc*alpha[:,None]+tl.dot(p.to(v.dtype),v);den=den*alpha+tl.sum(p,1);maximum=mx
 tl.store(O+head*NQ*D+rr[:,None]*D+dd[None,:],acc/den[:,None],(rr[:,None]<NQ)&(dd[None,:]<D))
 tl.store(LSE+head*NQ+rr,maximum+tl.log(den),rr<NQ)

@triton.jit
def _mass(Q,K,IDS,BIAS,LSE,M,NQ:tl.constexpr,A:tl.constexpr,D:tl.constexpr,H:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,GS:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
 block=tl.program_id(0);head=tl.program_id(1);tile=tl.program_id(2);group=block//(GS//BM)
 rr=block*BM+tl.arange(0,BM);col=tile*BN+tl.arange(0,BN);dd=tl.arange(0,BD)
 ids=tl.load(IDS+group*A+col,col<A,-1);valid=(col<A)&(ids>=0)
 q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<NQ)&(dd[None,:]<D),0.)
 k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0.)
 bias=tl.load(BIAS+group*A+col,col<A,-float('inf'));lse=tl.load(LSE+head*NQ+rr,rr<NQ,0.)
 z=tl.dot(q,k)*SCALE+bias[None,:]
 p=tl.where((rr[:,None]<NQ)&valid[None,:],tl.exp(z-lse[:,None]),0.)
 mass=tl.sum(p,0)/(H*tl.minimum(GS,NQ-group*GS))
 tl.atomic_add(M+group*A+col,mass,valid,sem='relaxed')

def weighted_attention(q,k,v,indices,log_weights,specials,scale,single_qk=False,relative_reference=None,group_size=512,mass_mode="probes"):
 if single_qk or relative_reference is not None:raise ValueError('Grouped executor uses M5c external evidence')
 h,r,d=q.shape;g,a=indices.shape;gs=group_size
 if g!=triton.cdiv(r,gs):raise ValueError('Group count mismatch')
 output=torch.empty((h,r,d),device=q.device,dtype=q.dtype);lse=torch.empty((h,r),device=q.device);mass=torch.zeros((g,a),device=q.device)
 bm,bn=64,64
 _attention[(triton.cdiv(r,bm),h)](q,k,v,indices,log_weights,specials,output,lse,r,k.shape[1],a,len(specials),d,q.stride(0),k.stride(0),gs,scale,bm,bn,triton.next_power_of_2(d),num_warps=4)
 if mass_mode=='full':
  _mass[(triton.cdiv(r,128),h,triton.cdiv(a,bn))](q,k,indices,log_weights,lse,mass,r,a,d,h,q.stride(0),k.stride(0),gs,scale,128,bn,triton.next_power_of_2(d),num_warps=8)
 else:
  global PROBE_CACHE
  if PROBE_CACHE is not None:raise RuntimeError('Previous query probe cache was not consumed')
  raw=torch.empty((g,h,4,a),device=q.device)
  _probe_mass[(g,h,triton.cdiv(a,64))](q,k,indices,log_weights,lse,mass,raw,r,a,d,h,q.stride(0),k.stride(0),gs,scale,triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
  PROBE_CACHE=(indices.data_ptr(),raw)
 return output,mass

PROBE_CACHE=None

@triton.jit
def _probe_mass(Q,K,IDS,BIAS,LSE,M,RAW,NQ:tl.constexpr,A:tl.constexpr,D:tl.constexpr,H:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,GS:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr):
 group=tl.program_id(0);head=tl.program_id(1);tile=tl.program_id(2);rows=tl.arange(0,16);col=tile*64+tl.arange(0,64);dd=tl.arange(0,BD)
 rr=tl.minimum(group*GS+GS//8+rows*(GS//4),NQ-1);previous=tl.minimum(group*GS+GS//8+(rows-1)*(GS//4),NQ-1)
 unique=(rows<4)&((rows==0)|(rr!=previous));count=tl.sum(unique.to(tl.float32),0)
 ids=tl.load(IDS+group*A+col,col<A,-1);valid=(col<A)&(ids>=0)
 q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rows[:,None]<4)&(dd[None,:]<D),0.)
 k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0.)
 raw=tl.dot(q,k,allow_tf32=False)*SCALE
 tl.store(RAW+((group*H+head)*4+rows[:,None])*A+col[None,:],tl.where(valid[None,:],raw,-float('inf')),(rows[:,None]<4)&(col[None,:]<A))
 bias=tl.load(BIAS+group*A+col,col<A,-float('inf'));lse=tl.load(LSE+head*NQ+rr,rows<4,0.)
 prob=tl.exp(raw+bias[None,:]-lse[:,None]);mass=tl.sum(tl.where(unique[:,None]&valid[None,:],prob,0.),0)/(H*count)
 tl.atomic_add(M+group*A+col,mass,valid,sem='relaxed')

def consume_probes(actual):
 global PROBE_CACHE
 if PROBE_CACHE is None:return None
 pointer,raw=PROBE_CACHE
 if pointer!=actual.data_ptr():raise RuntimeError('Unmatched probe cache owner')
 PROBE_CACHE=None;return raw
