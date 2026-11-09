"""Count-free observed+coarse-complement normalization,then head mean."""
import torch,triton
import triton.language as tl

@triton.jit
def _norm(RAW,IDS,GRID,BASE,SIZE,SHIFT,DEN,G:tl.constexpr,P:tl.constexpr,A:tl.constexpr,R:tl.constexpr,B:tl.constexpr,BR:tl.constexpr,LOGIT:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);p=tl.program_id(2);j=tl.arange(0,B);b=tl.arange(0,BR)
 offset=(h*G+g)*P+p;sel=tl.full((),0.,tl.float32);total=tl.full((),0.,tl.float32);maximum=tl.full((),-float('inf'),tl.float32)
 for tile in range(tl.cdiv(A,B)):
  at=tile*B+j;key=tl.load(IDS+g*A+at,at<A,-1);valid=(at<A)&(key>=0);block=tl.load(GRID+tl.maximum(key,0),valid,0);base=tl.load(BASE+offset*R+block,valid,0.);sel+=tl.sum(base,0)
  if LOGIT:
   z=tl.load(RAW+offset*A+at,valid,-float('inf'));mx=tl.maximum(maximum,tl.max(z,0));safe=tl.where(mx==-float('inf'),0.,mx);total=total*tl.exp(maximum-safe)+tl.sum(tl.exp(z-safe),0);maximum=mx
  else:
   z=tl.load(RAW+offset*A+at,valid,0.);total+=tl.sum(z,0)
 full=tl.sum(tl.load(BASE+offset*R+b,b<R,0.)*tl.load(SIZE+b,b<R,0.),0);missing=tl.maximum(full-sel,0.)
 if LOGIT:
  shift=tl.load(SHIFT+offset);gauge=tl.maximum(maximum,shift);value=tl.log(tl.maximum(total*tl.exp(maximum-gauge)+missing*tl.exp(shift-gauge),1e-30))+gauge
 else:value=tl.maximum(total+missing,1e-30)
 tl.store(DEN+offset,value)

@triton.jit
def _mean(RAW,DEN,IDS,COUNT,OUT,G:tl.constexpr,H:tl.constexpr,P:tl.constexpr,A:tl.constexpr,B:tl.constexpr,BH:tl.constexpr,LOGIT:tl.constexpr):
 g=tl.program_id(0);tile=tl.program_id(1);a=tile*B+tl.arange(0,B);h=tl.arange(0,BH);p=tl.arange(0,P)
 key=tl.load(IDS+g*A+a,a<A,-1);valid=(a<A)&(key>=0)
 z=tl.load(RAW+((h[:,None,None]*G+g)*P+p[None,:,None])*A+a[None,None,:],(h[:,None,None]<H)&valid[None,None,:],-float('inf') if LOGIT else 0.)
 den=tl.load(DEN+(h[:,None]*G+g)*P+p[None,:],h[:,None]<H,0. if LOGIT else 1.)
 value=tl.exp(z-den[:,:,None]) if LOGIT else z/den[:,:,None]
 if P==1:weight=tl.full((P,),1.,tl.float32)
 else:
  count=tl.load(COUNT+g*P+p);weight=count/tl.maximum(tl.sum(count,0),1.)
 result=tl.sum(tl.sum(value*weight[None,:,None],1),0)/H
 tl.store(OUT+g*A+a,result,a<A)

def finalize(mean,grid,sizes,ids,raw,counts=None,shift=None,logit=False):
 h,g,p,r=mean.shape;a=ids.shape[1]
 if shift is None:shift=(mean+sizes.log()[None,None,None]).logsumexp(-1).contiguous()
 else:shift=shift.reshape(h,g,p).contiguous()
 base=(mean-shift[...,None]).exp().contiguous();den=torch.empty(h,g,p,device=raw.device);out=torch.empty(g,a,device=raw.device)
 _norm[(g,h,p)](raw,ids,grid,base,sizes,shift,den,g,p,a,r,4096,triton.next_power_of_2(r),logit,num_warps=8)
 block=256 if p==1 else 128
 _mean[(g,triton.cdiv(a,block))](raw,den,ids,ids if counts is None else counts,out,g,h,p,a,block,triton.next_power_of_2(h),logit,num_warps=4 if p==1 else 8)
 return out
