"""Compress equal integer-cost parent candidates, preserving original ID tie rules."""
import torch,triton
import triton.language as tl
from .reuse_compact import _dual

@triton.jit
def _run_min(ac,ae,ai,bc,be,bi):
 same=ac==bc;use_b=(be<ae)|((be==ae)&(bi<ai))
 return bc,tl.where(same&~use_b,ae,be),tl.where(same&~use_b,ai,bi)

@triton.jit
def _compress(E,C,CE,CC,INDEX,COUNT,P:tl.constexpr,T:tl.constexpr,BP:tl.constexpr,BT:tl.constexpr):
 group=tl.program_id(0);p=tl.program_id(1)*BP+tl.arange(0,BP);i=tl.arange(0,BT)
 mask=(p[:,None]<P)&(i[None,:]<T)
 cost=tl.load(C+(group*P+p[:,None])*T+i[None,:],mask,0.)
 error=tl.load(E+(group*P+p[:,None])*T+i[None,:],mask,float('inf'))
 key=tl.where(i[None,:]<T,cost.to(tl.int32)*BT+i[None,:],2147483647)
 sorted_key=tl.sort(key,descending=False,dim=1);ordered=sorted_key%BT;ordered_cost=sorted_key//BT
 oe=tl.gather(error,ordered,1)
 _,best,idx=tl.associative_scan((ordered_cost,oe,ordered),axis=1,combine_fn=_run_min)
 next_cost=tl.gather(ordered_cost,tl.broadcast_to(tl.minimum(i+1,BT-1)[None,:],(BP,BT)),1)
 last=((i[None,:]==BT-1)|(next_cost!=ordered_cost))&(sorted_key<2147483647)&(best<float('inf'))
 slot=tl.cumsum(last.to(tl.int32),1)-1
 tl.store(CE+(group*P+p[:,None])*64+slot,best,(p[:,None]<P)&last&(slot<64))
 tl.store(CC+(group*P+p[:,None])*64+slot,ordered_cost.to(tl.float32),(p[:,None]<P)&last&(slot<64))
 tl.store(INDEX+(group*P+p[:,None])*64+slot,idx,(p[:,None]<P)&last&(slot<64))
 tl.store(COUNT+group*P+p,tl.sum(last.to(tl.int32),1),p<P)

@triton.jit
def _choose(E,C,INDEX,COUNT,DUAL,PART,SEL,P:tl.constexpr,TILES:tl.constexpr,FINAL:tl.constexpr,BP:tl.constexpr):
 group=tl.program_id(0);tile=tl.program_id(1);p=tile*BP+tl.arange(0,BP);ss=tl.arange(0,32)
 lo=tl.load(DUAL+group*2);hi=tl.load(DUAL+group*2+1);lam=hi if FINAL else (lo+hi)*.5
 offset=(group*P+p[:,None])*64+ss[None,:];mask=p[:,None]<P
 e=tl.load(E+offset,mask,float('inf'));cost=tl.load(C+offset,mask,0.);index=tl.load(INDEX+offset,mask,2147483647)
 value=e+lam*cost;minimum=tl.min(value,1);winner=tl.min(tl.where(value==minimum[:,None],index,2147483647),1)
 selected_cost=tl.sum(tl.where(index==winner[:,None],cost,0.),1)
 count=tl.load(COUNT+group*P+p,p<P,0)
 if tl.sum((count>32).to(tl.int32),0)>0:
  ee=tl.load(E+offset+32,mask,float('inf'));c2=tl.load(C+offset+32,mask,0.);i2=tl.load(INDEX+offset+32,mask,2147483647)
  v2=ee+lam*c2;m2=tl.min(v2,1);w2=tl.min(tl.where(v2==m2[:,None],i2,2147483647),1)
  use=(m2<minimum)|((m2==minimum)&(w2<winner));cc=tl.sum(tl.where(i2==w2[:,None],c2,0.),1)
  winner=tl.where(use,w2,winner);selected_cost=tl.where(use,cc,selected_cost)
 tl.store(PART+group*TILES+tile,tl.sum(tl.where(p<P,selected_cost,0.),0))
 if FINAL:tl.store(SEL+group*P+p,winner,p<P)

def choose(errors,costs,cap,iterations):
 g,p,t=errors.shape
 ce=torch.full((g,p,64),float('inf'),device=errors.device);cc=torch.zeros_like(ce);index=torch.full((g,p,64),2147483647,device=errors.device,dtype=torch.int32);count=torch.empty((g,p),device=errors.device,dtype=torch.int32)
 _compress[(g,triton.cdiv(p,4))](errors,costs,ce,cc,index,count,p,t,4,triton.next_power_of_2(t),num_warps=4,enable_fp_fusion=False)
 # The 108 frozen templates have <=54 different physical integer costs.
 upper=torch.where(costs==0,errors,0.).amax((1,2))+1e-12
 dual=torch.stack((torch.zeros_like(upper),upper),1).contiguous();tiles=triton.cdiv(p,64);part=torch.empty((g,tiles),device=errors.device);selected=torch.empty((g,p),device=errors.device,dtype=torch.int64)
 for _ in range(iterations):
  _choose[(g,tiles)](ce,cc,index,count,dual,part,selected,p,tiles,False,64,num_warps=4,enable_fp_fusion=False)
  _dual[(g,)](part,dual,tiles,cap,triton.next_power_of_2(tiles),num_warps=4,enable_fp_fusion=False)
 _choose[(g,tiles)](ce,cc,index,count,dual,part,selected,p,tiles,True,64,num_warps=4,enable_fp_fusion=False)
 return selected
