"""Export group-mean raw logits from formal sparse AV;no extra QK."""
import torch,triton
import triton.language as tl

@triton.jit
def _pool(Q,MASK,COUNT,OUT,N:tl.constexpr,G:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,BD:tl.constexpr):
 p=tl.program_id(0);h=tl.program_id(1);r=p*64+tl.arange(0,64);d=tl.arange(0,BD)
 valid=tl.load(MASK+r,r<N,False);x=tl.load(Q+h*QH+r[:,None]*D+d[None],(r[:,None]<N)&(d[None]<D),0.).to(tl.float32)
 count=tl.maximum(tl.load(COUNT+p),1);value=tl.sum(tl.where(valid[:,None],x,0.),0)/count
 tl.store(OUT+(h*G*4+p)*D+d,value,d<D)

def pool_queries(q,mask,counts):
 h,n,d=q.shape;g=len(counts);out=torch.empty(h,g,4,d,device=q.device,dtype=torch.float32)
 _pool[(g*4,h)](q,mask,counts,out,n,g,d,q.stride(0),triton.next_power_of_2(d),num_warps=4)
 return out

@triton.jit
def _attention(Q,K,V,IDS,SPECIAL,BIAS,MASK,COUNT,SHIFT,OUT,MEAN,H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,G:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,VH:tl.constexpr,OH:tl.constexpr,SCALE:tl.constexpr,CAPTURE:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);g=t//4;row=tl.arange(0,BM);rr=t*BM+row;dd=tl.arange(0,BD);kk=tl.arange(0,BK)
 q=tl.load(Q+h*QH+rr[:,None]*D+dd[None],(rr[:,None]<R)&(dd[None]<D),0.)
 maximum=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
 if CAPTURE:qm=tl.load(MASK+rr,rr<R,False);count=tl.maximum(tl.load(COUNT+g*4+t%4),1)
 for tile in range(tl.cdiv(S+A,BK)):
  at=tile*BK+kk;key=tl.load(IDS+g*A+at-S,(at>=S)&(at<S+A),-1);special=tl.load(SPECIAL+at,at<S,0);key=tl.where(at<S,special,key);valid=(at<S+A)&(key>=0)
  bias=tl.load(BIAS+g*A+at-S,(at>=S)&valid,0.)
  k=tl.load(K+h*KH+tl.maximum(key,0)[None,:]*D+dd[:,None],valid[None]&(dd[:,None]<D),0.)
  v=tl.load(V+h*VH+tl.maximum(key,0)[:,None]*D+dd[None],valid[:,None]&(dd[None]<D),0.)
  raw=tl.dot(q,k,allow_tf32=False)*SCALE;z=tl.where(valid[None],raw+bias[None],-float('inf'))
  if CAPTURE:
   shift=tl.load(SHIFT+(h*G+g)*4+t%4)
   value=tl.sum(tl.where(qm[:,None],tl.exp(raw-shift),0.),0)/count
   tl.store(MEAN+((h*G+g)*4+t%4)*A+at-S,value,(at>=S)&(at<S+A))
  mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx);alpha=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
  acc=acc*alpha[:,None]+tl.dot(p.to(v.dtype),v,allow_tf32=False);den=den*alpha+tl.sum(p,1);maximum=mx
 out=acc/den[:,None]
 tl.store(OUT+h*OH+rr[:,None]*D+dd[None],out,(rr[:,None]<R)&(dd[None]<D))

def attention(q,k,v,ids,specials,weights,scale,mask,counts,capture=True,warps=4,stages=1,shift=None):
 h,r,d=q.shape;g,a=ids.shape;assert g==triton.cdiv(r,256)
 bias=weights.log().contiguous();out=torch.empty_like(q);mean=torch.empty(h,g,4,a,device=q.device,dtype=torch.float32) if capture else out
 if capture and r%256:mean[:,-1,triton.cdiv(r%256,64):].zero_()
 _attention[(triton.cdiv(r,64),h)](q,k,v,ids,specials,bias,mask,counts,shift,out,mean,h,r,k.shape[1],g,a,len(specials),d,q.stride(0),k.stride(0),v.stride(0),out.stride(0),scale,capture,triton.next_power_of_2(d),64,128,num_warps=warps,num_stages=stages)
 return out,dict(mean_exp=mean,block_keys=128)
