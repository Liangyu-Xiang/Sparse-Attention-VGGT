"""Fused current-Key pooling,anchor gather and fixed head-sketch projection."""
import torch,triton
import triton.language as tl

@triton.jit
def _prepare(K,PATCH,SPECIAL,W,BANK,ANCHOR,Z,N:tl.constexpr,R:tl.constexpr,S:tl.constexpr,H:tl.constexpr,D:tl.constexpr,SK:tl.constexpr,KH:tl.constexpr,BLOCK:tl.constexpr,M:tl.constexpr,BD:tl.constexpr):
 r=tl.program_id(0);h=tl.program_id(1);t=tl.arange(0,BLOCK);d=tl.arange(0,BD)
 if r<R:
  token=r*BLOCK+t;key=tl.load(PATCH+token,token<N,0);values=tl.load(K+h*KH+key[:,None]*D+d[None,:],(token[:,None]<N)&(d[None,:]<D),0.);size=tl.minimum(BLOCK,N-r*BLOCK)
  mean=tl.sum(values.to(tl.float32),0)/size;tl.store(BANK+(h*(R+S)+r)*D+d,mean,d<D)
  slot=tl.arange(0,M);effective=tl.minimum(M,size);offset=tl.minimum((2*slot+1)*size//(2*effective),size-1);sample=tl.gather(values,tl.broadcast_to(offset[:,None],(M,BD)),0)
  tl.store(ANCHOR+((h*R+r)*M+slot[:,None])*D+d[None,:],sample,(slot[:,None]<M)&(d[None,:]<D))
  col=tl.arange(0,16);weight=tl.load(W+h*D*SK+d[:,None]*SK+col[None,:],(d[:,None]<D)&(col[None,:]<SK),0.).to(values.dtype)
  projected=tl.dot(values,weight,allow_tf32=False)
  tl.store(Z+token[:,None]*(H*SK)+h*SK+col[None,:],projected,(token[:,None]<N)&(col[None,:]<SK))
 else:
  special_key=tl.load(SPECIAL+r-R);special_values=tl.load(K+h*KH+special_key*D+d,d<D,0.);tl.store(BANK+(h*(R+S)+r)*D+d,special_values,d<D)

def prepare(k,patches,specials,matrix,block=64,samples=8):
 h,_,d=k.shape;n=len(patches);r=triton.cdiv(n,block);sk=matrix.shape[-1]
 if h!=16 or sk!=4:raise ValueError('Exact BF16 projection weights require16heads,sketch4')
 bank=torch.empty((h,r+len(specials),d),device=k.device,dtype=torch.float16);anchors=torch.empty((h,r*samples,d),device=k.device,dtype=torch.float16);z=torch.empty((n,h*sk),device=k.device)
 _prepare[(r+len(specials),h)](k,patches,specials,matrix,bank,anchors,z,n,r,len(specials),h,d,sk,k.stride(0),block,samples,triton.next_power_of_2(d),num_warps=4,num_stages=1)
 return bank,anchors,z
