"""Grid-free sparse Flash-style AV; compact history A/B exports are optional.

Q=[H,R,D], K/V=[H,N,D], IDs=[G,A], G=ceil(R/32). Bias is already
computed by the selector. Probe positions are local-window query indices.
No Grid, routing, sorting, mass transfer or Head-specific mask is accepted.
"""
import torch
import triton
import triton.language as tl
from contextlib import nullcontext

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
  if INLINE:
   probe=tl.arange(0,P);pos=tl.load(POS+g*P+probe)-g*32
   raw=tl.gather(z,tl.broadcast_to(pos[:,None],(P,BK)),0);m=tl.max(raw,1);sm=tl.where(m==-float('inf'),0.,m);ex=tl.exp(raw-sm[:,None]);s=tl.sum(ex,1)
   local=ex/tl.where(s>0,s,1.)[:,None]
   tl.store(TILE_P+((g*H+h)*P+probe[:,None])*(S+A)+at[None,:],local,at[None,:]<S+A)
   tl.store(TILE_LSE+((g*H+h)*P+probe)*T+tile,tl.where(s>0,sm+tl.log(s),-float('inf')))
 result=acc/tl.where(den>0,den,1.)[:,None]
 tl.store(OUT+(h*R+rr[:,None])*D+d[None,:],result,(rr[:,None]<R)&(d[None,:]<D))
 tl.store(LSE+h*R+rr,maximum+tl.log(den),rr<R)

@triton.jit
def _inline_mean(LOCAL,LOCAL_LSE,LSE,POS,PROB,H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,S:tl.constexpr,P:tl.constexpr,T:tl.constexpr,BK:tl.constexpr,BH:tl.constexpr):
 tile=tl.program_id(0);g=tl.program_id(1);p=tl.program_id(2);col=tile*BK+tl.arange(0,BK);h=tl.arange(0,BH)
 row=tl.load(POS+g*P+p);final=tl.load(LSE+h*R+row,h<H,0.)
 local=tl.load(LOCAL+((g*H+h[:,None])*P+p)*(S+A)+col[None,:],(h[:,None]<H)&(col[None,:]<S+A),0.).to(tl.float32)
 norm=tl.load(LOCAL_LSE+((g*H+h)*P+p)*T+tile,h<H,-float('inf'))
 value=tl.sum(tl.where(h[:,None]<H,local*tl.exp(norm-final)[:,None],0.),0)/H
 tl.store(PROB+(g*P+p)*(S+A)+col,value,col<S+A)

@triton.jit
def _probe(Q,K,IDS,SPECIAL,BIAS,POS,LSE,PROB,H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,P:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,HAS_BIAS:tl.constexpr,DEBIASED_LOG:tl.constexpr=False):
 tile=tl.program_id(0);g=tl.program_id(1);p=tl.arange(0,16);d=tl.arange(0,BD);col=tile*BK+tl.arange(0,BK)
 special=tl.load(SPECIAL+col,col<S,0);key=tl.load(IDS+g*A+col-S,(col>=S)&(col<S+A),-1);key=tl.where(col<S,special,key);valid=(col<S+A)&(key>=0)
 bias=tl.full((BK,),0.,tl.float32)
 if HAS_BIAS:bias=tl.load(BIAS+g*A+col-S,(col>=S)&valid,0.)
 row=tl.load(POS+g*P+p,p<P,0);total=tl.zeros((16,BK),tl.float32)
 maximum=tl.full((16,BK),-float('inf'),tl.float32)
 for h in range(H):
  q=tl.load(Q+h*QH+row[:,None]*D+d[None,:],(p[:,None]<P)&(d[None,:]<D),0.)
  k=tl.load(K+h*KH+tl.maximum(key,0)[None,:]*D+d[:,None],valid[None,:]&(d[:,None]<D),0.)
  z=tl.dot(q,k,allow_tf32=False)*SCALE;lse=tl.load(LSE+h*R+row,p<P,0.)
  if DEBIASED_LOG:
   # log(mean_h(P_comp,h / exp(bias))). Compute before exponentiation:
   # even a zero assignment weight or an FP16-underflowing P is recoverable.
   score=tl.where(valid[None,:]&(p[:,None]<P),z-lse[:,None],-float('inf'))
   mx=tl.maximum(maximum,score);safe=tl.where(mx==-float('inf'),0.,mx)
   total=total*tl.exp(maximum-safe)+tl.exp(score-safe);maximum=mx
  else:total+=tl.where(valid[None,:]&(p[:,None]<P),tl.exp(z+bias[None,:]-lse[:,None]),0.)
 if DEBIASED_LOG:value=tl.where(total>0,maximum+tl.log(total)-tl.log(float(H)),-float('inf'))
 else:value=total/H
 tl.store(PROB+(g*P+p[:,None])*(S+A)+col[None,:],value,(p[:,None]<P)&(col[None,:]<S+A))

def direct_sparse_attention(q,k,v,sparse_key_ids,special_key_ids,scale,logit_bias=None,probe_positions=None,return_history_probability=False,*,history_export='lse_probe',block_keys=128,num_warps=4,num_stages=2,scope=None):
 scope=scope or (lambda name:nullcontext())
 h,r,d=q.shape;g,a=sparse_key_ids.shape;s=len(special_key_ids)
 if g!=triton.cdiv(r,32):raise ValueError('One sparse support per32 queries is required')
 if q.stride(-1)!=1 or q.stride(-2)!=d or k.stride(-1)!=1 or k.stride(-2)!=d or v.stride(-1)!=1 or v.stride(-2)!=d:raise ValueError('Contiguous head rows required')
 if block_keys not in (64,128,256):raise ValueError('block_keys must be64/128/256')
 if probe_positions is None and not return_history_probability:
  probe_positions=torch.empty((g,4),device=q.device,dtype=torch.int32)
 elif probe_positions is None:
  probe_positions=(torch.arange(g,device=q.device)[:,None]*32+(torch.arange(4,device=q.device)*8+4)[None]).clamp_max(r-1).int().contiguous()
 p=probe_positions.shape[1]
 if p not in (1,4):raise ValueError('One or four probes supported')
 if return_history_probability and history_export not in ('inline','lse_probe','debiased_log_lse_probe'):raise ValueError('Unknown history export')
 inline=return_history_probability and history_export=='inline';tiles=triton.cdiv(s+a,block_keys)
 out=torch.empty((h,r,d),device=q.device,dtype=q.dtype);lse=torch.empty((h,r),device=q.device,dtype=torch.float32)
 local=torch.empty((g,h,p,s+a),device=q.device,dtype=torch.float16) if inline else lse
 local_lse=torch.empty((g,h,p,tiles),device=q.device,dtype=torch.float32) if inline else lse
 with scope('direct_av'):
  _attention[(g,h)](q,k,v,sparse_key_ids,special_key_ids,logit_bias if logit_bias is not None else lse,probe_positions,out,lse,local,local_lse,h,r,k.shape[1],a,s,d,q.stride(0),k.stride(0),v.stride(0),scale,max(16,triton.next_power_of_2(d)),block_keys,logit_bias is not None,inline,p,tiles,num_warps=num_warps,num_stages=num_stages)
 evidence=dict(lse=lse)
 if return_history_probability:
  probability=torch.empty((g,p,s+a),device=q.device,dtype=torch.float32)
  with scope('direct_probability_export'):
   if inline:_inline_mean[(tiles,g,p)](local,local_lse,lse,probe_positions,probability,h,r,a,s,p,tiles,block_keys,triton.next_power_of_2(h),num_warps=4)
   else:_probe[(tiles,g)](q,k,sparse_key_ids,special_key_ids,logit_bias if logit_bias is not None else lse,probe_positions,lse,probability,h,r,k.shape[1],a,s,d,q.stride(0),k.stride(0),scale,p,max(16,triton.next_power_of_2(d)),block_keys,logit_bias is not None,history_export=='debiased_log_lse_probe',num_warps=4,num_stages=1)
  prefix='debiased_log_' if history_export=='debiased_log_lse_probe' else ''
  evidence.update({prefix+'special_probability':probability[:,:,:s],prefix+'patch_probability':probability[:,:,s:]})
  evidence.update(history_export=history_export,scratch_bytes=(local.numel()*local.element_size()+local_lse.numel()*local_lse.element_size()) if inline else 0)
 return out,evidence
