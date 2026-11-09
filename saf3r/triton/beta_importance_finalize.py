"""Count-free observed+coarse-complement normalization,then head mean."""
import torch,triton
import triton.language as tl

@triton.jit
def _norm(RAW,IDS,GRID,BASE,SIZE,SHIFT,DEN,HISTORY_SUM,G:tl.constexpr,P:tl.constexpr,A:tl.constexpr,RA:tl.constexpr,R:tl.constexpr,B:tl.constexpr,BR:tl.constexpr,LOGIT:tl.constexpr,HAS_SUM:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1);p=tl.program_id(2);j=tl.arange(0,B);b=tl.arange(0,BR)
 offset=(h*G+g)*P+p;sel=tl.full((),0.,tl.float32);total=tl.full((),0.,tl.float32);maximum=tl.full((),-float('inf'),tl.float32)
 for tile in range(tl.cdiv(A,B)):
  at=tile*B+j;key=tl.load(IDS+g*A+at,at<A,-1);valid=(at<A)&(key>=0);block=tl.load(GRID+tl.maximum(key,0),valid,0);base=tl.load(BASE+offset*R+block,valid,0.);sel+=tl.sum(base,0)
  if LOGIT:
   z=tl.load(RAW+offset*RA+at,valid,-float('inf'));mx=tl.maximum(maximum,tl.max(z,0));safe=tl.where(mx==-float('inf'),0.,mx);total=total*tl.exp(maximum-safe)+tl.sum(tl.exp(z-safe),0);maximum=mx
  elif not HAS_SUM:
   z=tl.load(RAW+offset*RA+at,valid,0.);total+=tl.sum(z,0)
 full=tl.sum(tl.load(BASE+offset*R+b,b<R,0.)*tl.load(SIZE+b,b<R,0.),0);missing=tl.maximum(full-sel,0.)
 if LOGIT:
  shift=tl.load(SHIFT+offset);gauge=tl.maximum(maximum,shift);value=tl.log(tl.maximum(total*tl.exp(maximum-gauge)+missing*tl.exp(shift-gauge),1e-30))+gauge
 else:
  if HAS_SUM:total=tl.load(HISTORY_SUM+offset)
  value=tl.maximum(total+missing,1e-30)
 tl.store(DEN+offset,value)

@triton.jit
def _mean(RAW,DEN,IDS,COUNT,OUT,G:tl.constexpr,H:tl.constexpr,P:tl.constexpr,A:tl.constexpr,RA:tl.constexpr,B:tl.constexpr,BH:tl.constexpr,LOGIT:tl.constexpr):
 g=tl.program_id(0);tile=tl.program_id(1);a=tile*B+tl.arange(0,B);h=tl.arange(0,BH);p=tl.arange(0,P)
 key=tl.load(IDS+g*A+a,a<A,-1);valid=(a<A)&(key>=0)
 z=tl.load(RAW+((h[:,None,None]*G+g)*P+p[None,:,None])*RA+a[None,None,:],(h[:,None,None]<H)&valid[None,None,:],-float('inf') if LOGIT else 0.)
 den=tl.load(DEN+(h[:,None]*G+g)*P+p[None,:],h[:,None]<H,0. if LOGIT else 1.)
 value=tl.exp(z-den[:,:,None]) if LOGIT else z/den[:,:,None]
 if P==1:weight=tl.full((P,),1.,tl.float32)
 else:
  count=tl.load(COUNT+g*P+p);weight=count/tl.maximum(tl.sum(count,0),1.)
 result=tl.sum(tl.sum(value*weight[None,:,None],1),0)/H
 tl.store(OUT+g*A+a,result,a<A)

@triton.jit
def _count_selected(IDS, GRID, SELECTED, A:tl.constexpr, R:tl.constexpr, B:tl.constexpr):
 group=tl.program_id(0)
 index=tl.program_id(1)*B+tl.arange(0,B)
 key=tl.load(IDS+group*A+index,index<A,-1)
 valid=key>=0
 region=tl.load(GRID+tl.maximum(key,0),valid,0)
 tl.atomic_add(SELECTED+group*R+region,1,valid,sem="relaxed")

@triton.jit
def _count_region_ends(IDS, ENDS, OFFSETS, A:tl.constexpr, R:tl.constexpr,
                       B:tl.constexpr, STEPS:tl.constexpr):
 group=tl.program_id(0)
 region=tl.program_id(1)*B+tl.arange(0,B)
 end=tl.load(ENDS+region,region<R,0)
 lo=tl.full((B,),0,tl.int32)
 hi=tl.full((B,),A,tl.int32)
 for _ in range(STEPS):
  mid=(lo+hi)//2
  key=tl.load(IDS+group*A+mid,(mid<A)&(region<R),-1)
  below=(key>=0)&(key<end)
  lo=tl.where(below,mid+1,lo)
  hi=tl.where(below,hi,mid)
 tl.store(OFFSETS+group*R+region,lo,region<R)


@triton.jit
def _compact_den(OBSERVED, BASE, SIZE, SELECTED, DEN, G:tl.constexpr, P:tl.constexpr,
                 R:tl.constexpr, B:tl.constexpr):
 row=tl.program_id(0)
 group=(row//P)%G
 region=tl.arange(0,B)
 size=tl.load(SIZE+region,region<R,0.)
 selected=tl.load(SELECTED+group*R+region,region<R,0)
 base=tl.load(BASE+row*R+region,region<R,0.)
 unknown=tl.sum(base*tl.maximum(size-selected,0.),0)
 tl.store(DEN+row,tl.maximum(tl.load(OBSERVED+row)+unknown,1.e-30))

@triton.jit
def _compact_den_fused(RAW, OBSERVED, MEAN, SHIFT, SIZE, SELECTED, DEN,
                        G:tl.constexpr, P:tl.constexpr, R:tl.constexpr,
                        A:tl.constexpr, RA:tl.constexpr, BR:tl.constexpr,
                        BK:tl.constexpr, SCAN:tl.constexpr):
 row=tl.program_id(0)
 group=(row//P)%G
 region=tl.arange(0,BR)
 size=tl.load(SIZE+region,region<R,0.)
 selected=tl.load(SELECTED+group*R+region,region<R,0)
 coarse=tl.load(MEAN+row*R+region,region<R,0.)
 base=tl.exp(coarse-tl.load(SHIFT+row))
 unknown=tl.sum(base*tl.maximum(size-selected,0.),0)
 if SCAN:
  slot=tl.arange(0,BK)
  accum=tl.full((BK,),0.,tl.float32)
  for tile in range(tl.cdiv(A,BK)):
   offset=tile*BK+slot
   accum+=tl.load(RAW+row*RA+offset,offset<A,0.)
  observed=tl.sum(accum,0)
 else:
  observed=tl.load(OBSERVED+row)
 tl.store(DEN+row,tl.maximum(observed+unknown,1.e-30))


def finalize(mean,grid,sizes,ids,raw,counts=None,shift=None,logit=False,*,norm_block=4096,norm_warps=8,mean_block=None,mean_warps=None,history_sum=None,diagnostics=None,compact_norm=False,fused_compact=False, den_mode=0, sorted_histogram=False):
 h,g,p,r=mean.shape;a=ids.shape[1]
 if shift is None:shift=(mean+sizes.log()[None,None,None]).logsumexp(-1).contiguous()
 else:shift=shift.reshape(h,g,p).contiguous()
 if den_mode and not (compact_norm and fused_compact):raise ValueError("Fused denominator requires compact normalization")
 base=(mean-shift[...,None]).exp().contiguous() if not den_mode else None
 den=torch.empty(h,g,p,device=raw.device);out=torch.empty(g,a,device=raw.device)
 if diagnostics is not None:diagnostics['norm_start'].record()
 if compact_norm:
  if logit:
   raise ValueError('Compact denominator expects linear per-key history statistics')
  if fused_compact:
   if sorted_histogram:
    ends=sizes.to(torch.int32).cumsum(0,dtype=torch.int32)
    offsets=torch.empty((g,r),device=ids.device,dtype=torch.int32)
    _count_region_ends[(g,triton.cdiv(r,128))](ids,ends,offsets,a,r,128,a.bit_length(),num_warps=4)
    selected=torch.cat((offsets[:,:1],offsets[:,1:]-offsets[:,:-1]),dim=1)
   else:
    selected=torch.zeros((g,r),device=ids.device,dtype=torch.int32)
    _count_selected[(g,triton.cdiv(a,256))](ids,grid,selected,a,r,256,num_warps=4)
   scan=den_mode==2 and history_sum is None
   observed=raw if scan else (raw.sum(-1,dtype=torch.float32).contiguous() if history_sum is None else history_sum)
   if den_mode:
    _compact_den_fused[(h*g*p,)](raw,observed,mean.contiguous(),shift,sizes,selected,den,
                                 g,p,r,a,raw.stride(-2),triton.next_power_of_2(r),1024,scan,num_warps=4)
   else:
    _compact_den[(h*g*p,)](observed,base,sizes,selected,den,g,p,r,triton.next_power_of_2(r),num_warps=4)
  else:
   valid=ids>=0
   regions=grid[ids.clamp_min(0).long()].long()
   selected=torch.zeros((g,r),device=ids.device,dtype=torch.int32)
   selected.scatter_add_(1,regions,valid.int())
   remaining=(sizes[None]-selected).clamp_min_(0)
   unknown=(base*remaining[None,:,None,:]).sum(-1)
   observed=raw.sum(-1,dtype=torch.float32) if history_sum is None else history_sum
   den=(observed+unknown).clamp_min_(1e-30).contiguous()
 else:
  _norm[(g,h,p)](raw,ids,grid,base,sizes,shift,den,history_sum if history_sum is not None else den,g,p,a,raw.stride(-2),r,norm_block,triton.next_power_of_2(r),logit,history_sum is not None,num_warps=norm_warps)
 if diagnostics is not None:diagnostics['norm_end'].record();diagnostics['denominator']=den
 block=mean_block if mean_block is not None else (256 if p==1 else 128)
 _mean[(g,triton.cdiv(a,block))](raw,den,ids,ids if counts is None else counts,out,g,h,p,a,raw.stride(-2),block,triton.next_power_of_2(h),logit,num_warps=mean_warps if mean_warps is not None else (4 if p==1 else 8))
 if diagnostics is not None:diagnostics['mean_end'].record()
 return out
