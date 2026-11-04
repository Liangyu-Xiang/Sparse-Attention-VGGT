"""Independent v8 history observation. Grid data never enters AV kernel.

Legacy full-query Grid summaries need an auxiliary QK pass. Four-probe
conditional normalization uses compact Grid statistics,not Head×Probe×Key RAW.
"""
import torch,triton
import triton.language as tl
from .reuse_ready_history import _pack_bits

@triton.jit
def _seg_sum(va,fa,vb,fb):return tl.where(fb,vb,va+vb),fa|fb
@triton.jit
def _seg_max(va,fa,vb,fb):return tl.where(fb,vb,tl.maximum(va,vb)),fa|fb

@triton.jit
def _observe(Q,K,IDS,GRID,BIAS,POS,LSE,SUMMARY,MAXIMUM,SUM,BOUNDS,FAST,H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,D:tl.constexpr,REGIONS:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
 tile=tl.program_id(0);g=tl.program_id(1);h=tl.program_id(2);col=tile*BK+tl.arange(0,BK);d=tl.arange(0,BD);row=tl.arange(0,32);rr=g*32+row
 key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0);region=tl.load(GRID+g*A+col,valid,-1)
 prev=tl.load(GRID+g*A+col-1,(col>0)&valid,-2);nxt=tl.load(GRID+g*A+col+1,(col+1<A)&valid,-2)
 prevkey=tl.load(IDS+g*A+col-1,(col>0)&(col<A),-1);nextkey=tl.load(IDS+g*A+col+1,col+1<A,-1)
 head=(tl.arange(0,BK)==0)|(prev!=region)|(prevkey<0);end=(tl.arange(0,BK)==BK-1)|(nxt!=region)|(nextkey<0)|(col==A-1)
 q=tl.load(Q+h*QH+rr[:,None]*D+d[None,:],(rr[:,None]<R)&(d[None,:]<D),0.)
 k=tl.load(K+h*KH+tl.maximum(key,0)[None,:]*D+d[:,None],valid[None,:]&(d[:,None]<D),0.)
 raw=tl.dot(q,k,allow_tf32=False)*SCALE;bias=tl.load(BIAS+g*A+col,valid,0.)
 lse=tl.load(LSE+h*R+rr,rr<R,0.)
 p=tl.where(valid[None,:]&(rr[:,None]<R),tl.exp(raw+bias[None,:]-lse[:,None]),0.)
 rows=tl.minimum(32,R-g*32);factor=1./(H*rows)
 weight=tl.sum(p,0)*factor;unit=tl.sum(p*tl.exp(-bias)[None,:],0)*factor
 total,_=tl.associative_scan((weight,head),0,_seg_sum);original,_=tl.associative_scan((unit,head),0,_seg_sum)
 tl.atomic_add(SUMMARY+(g*2)*REGIONS+region,total,valid&end,sem='relaxed');tl.atomic_add(SUMMARY+(g*2+1)*REGIONS+region,original,valid&end,sem='relaxed')
 for probe in tl.static_range(4):
  position=tl.load(POS+g*4+probe)-g*32
  values=tl.gather(raw,tl.full((1,BK),position,tl.int32),0).reshape(BK)
  values=tl.where(valid,values,-float('inf'));
  if tl.load(FAST+g*H+h):
   bound=tl.load(BOUNDS+(g*H+h)*4+probe)
   scaled=tl.where(valid,tl.exp(values-bound),0.);total,_=tl.associative_scan((scaled,head),0,_seg_sum)
   tl.atomic_add(SUM+((g*H+h)*4+probe)*REGIONS+region,total,valid&end,sem='relaxed')
  else:
   maximum,_=tl.associative_scan((values,head),0,_seg_max)
   tl.atomic_max(MAXIMUM+((g*H+h)*4+probe)*REGIONS+region,maximum,valid&end,sem='relaxed')

@triton.jit
def _sum_probe(Q,K,IDS,GRID,POS,MAXIMUM,SUM,FAST,H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,D:tl.constexpr,REGIONS:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
 g=tl.program_id(0);h=tl.program_id(1)
 if not tl.load(FAST+g*H+h):
  for tile in range(tl.cdiv(A,BK)):
   col=tile*BK+tl.arange(0,BK);p=tl.arange(0,16);d=tl.arange(0,BD)
   key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0);region=tl.load(GRID+g*A+col,valid,-1)
   prev=tl.load(GRID+g*A+col-1,(col>0)&valid,-2);nxt=tl.load(GRID+g*A+col+1,(col+1<A)&valid,-2)
   prevkey=tl.load(IDS+g*A+col-1,(col>0)&(col<A),-1);nextkey=tl.load(IDS+g*A+col+1,col+1<A,-1)
   head=(tl.arange(0,BK)==0)|(prev!=region)|(prevkey<0);end=(tl.arange(0,BK)==BK-1)|(nxt!=region)|(nextkey<0)|(col==A-1)
   row=tl.load(POS+g*4+p,p<4,0);q=tl.load(Q+h*QH+row[:,None]*D+d[None,:],(p[:,None]<4)&(d[None,:]<D),0.)
   k=tl.load(K+h*KH+tl.maximum(key,0)[None,:]*D+d[:,None],valid[None,:]&(d[:,None]<D),0.);raw=tl.dot(q,k,allow_tf32=False)*SCALE
   for probe in tl.static_range(4):
    values=tl.gather(raw,tl.full((1,BK),probe,tl.int32),0).reshape(BK)
    maximum=tl.load(MAXIMUM+((g*H+h)*4+probe)*REGIONS+region,valid,0.)
    value=tl.where(valid,tl.exp(values-maximum),0.);total,_=tl.associative_scan((value,head),0,_seg_sum)
    tl.atomic_add(SUM+((g*H+h)*4+probe)*REGIONS+region,total,valid&end,sem='relaxed')

@triton.jit
def _cache(Q,K,IDS,GRID,POS,DEN,LOGZ,CACHE,H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,D:tl.constexpr,REGIONS:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr,LOGEPS:tl.constexpr):
 tile=tl.program_id(0);g=tl.program_id(1);p=tl.arange(0,16);col=tile*BK+tl.arange(0,BK);d=tl.arange(0,BD)
 key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0);region=tl.load(GRID+g*A+col,valid,0);row=tl.load(POS+g*4+p,p<4,0)
 total=tl.zeros((16,BK),tl.float32)
 for h in range(H):
  q=tl.load(Q+h*QH+row[:,None]*D+d[None,:],(p[:,None]<4)&(d[None,:]<D),0.)
  k=tl.load(K+h*KH+tl.maximum(key,0)[None,:]*D+d[:,None],valid[None,:]&(d[:,None]<D),0.)
  raw=tl.dot(q,k,allow_tf32=False)*SCALE
  den=tl.load(DEN+((g*H+h)*4+p[:,None])*REGIONS+region[None,:],(p[:,None]<4)&valid[None,:],0.);z=tl.load(LOGZ+(g*H+h)*4+p,p<4,0.)
  total+=tl.where(valid[None,:]&(p[:,None]<4),tl.exp(raw-den+tl.minimum(den-z[:,None]-LOGEPS,0.)),0.)
 for probe in tl.static_range(4):
  value=tl.gather(total,tl.full((1,BK),probe,tl.int32),0).reshape(BK)/H
  encoded=value.to(tl.float16).to(tl.uint16,bitcast=True);_pack_bits(encoded,CACHE,g*4+probe,tile,A,BK)

def observe(q,k,ids,key_grid,bias,positions,lse,base,sizes,scale,max_key_norm=None,probe_q=None):
 g,h,_,regions=base.shape;a=ids.shape[1];r=q.shape[1];d=q.shape[2]
 if max_key_norm is None:max_key_norm=torch.linalg.vector_norm(k,dim=-1,dtype=torch.float32).amax(-1)
 if probe_q is None:probe_q=q[:,positions.long()]
 bounds=(torch.linalg.vector_norm(probe_q,dim=-1,dtype=torch.float32).permute(1,0,2)*max_key_norm[None,:,None]*scale*1.000001+0.0001).contiguous()
 fast=(bounds.amax(-1)<=32).contiguous()
 maxima=base.clone().contiguous();summary=torch.zeros((g,2,regions),device=q.device);sums=torch.zeros_like(base)
 _observe[(triton.cdiv(a,128),g,h)](q,k,ids,key_grid,bias,positions,lse,summary,maxima,sums,bounds,fast,h,r,a,d,regions,q.stride(0),k.stride(0),scale,128,triton.next_power_of_2(d),num_warps=4,num_stages=1)
 _sum_probe[(g,h)](q,k,ids,key_grid,positions,maxima,sums,fast,h,r,a,d,regions,q.stride(0),k.stride(0),scale,128,triton.next_power_of_2(d),num_warps=4,num_stages=1)
 counts=torch.zeros((g,regions),device=q.device,dtype=torch.int32);counts.scatter_add_(1,key_grid.long(),(ids>=0).int());missing=sizes[None]-counts
 shift=torch.where(fast[:,:,None,None],bounds[:,:,:,None],maxima)
 den=shift+(sums+missing[:,None,None,:]*torch.exp(base-shift)).log()
 return summary,den,counts

def store(q,k,ids,key_grid,positions,den,logz,cache,scale):
 import math
 g,h,_,regions=den.shape;a=ids.shape[1];r=q.shape[1];d=q.shape[2]
 _cache[(triton.cdiv(a,128),g)](q,k,ids,key_grid,positions,den,logz,cache,h,r,a,d,regions,q.stride(0),k.stride(0),scale,128,triton.next_power_of_2(d),math.log(1e-30),num_warps=4,enable_fp_fusion=False)
