"""Bucketed member reductions replace per-token state-loss atomics."""
import torch,triton
import triton.language as tl

@triton.jit
def _loss(MEM,DR,DIST,RANK,HIDS,PRIOR,CLUST,E,
 P:tl.constexpr,F:tl.constexpr,G,C,W:tl.constexpr,M:tl.constexpr,BG:tl.constexpr,BC:tl.constexpr):
 gg=tl.program_id(0)*BG+tl.arange(0,BG);cc=tl.program_id(1)*BC+tl.arange(0,BC);mm=tl.arange(0,M)
 cl=tl.load(CLUST+cc,cc<C,0);member=tl.load(MEM+cl[:,None]*W+mm[None,:],cc[:,None]<C,-1)
 valid=(gg[:,None,None]<G)&(cc[None,:,None]<C)&(member[None,:,:]>=0)
 a=tl.load(PRIOR+gg[:,None,None]*P+tl.maximum(member,0)[None,:,:],valid,0.)
 dr=tl.load(DR+tl.maximum(member,0),member>=0,0.)
 rep=tl.sum(a*dr[None,:,:],2)
 base=(gg[:,None]*F+cl[None,:])*11;outmask=(gg[:,None]<G)&(cc[None,:]<C)
 tl.store(E+base,0.,outmask);tl.store(E+base+1,rep,outmask);tl.store(E+base+10,0.,outmask)
 nearest=tl.full((BG,BC,M),2.,tl.float32);chosen=tl.full((BG,BC,M),False,tl.int1)
 for slot in tl.static_range(4):
  h=tl.load(HIDS+(gg[:,None]*F+cl[None,:])*4+slot,outmask,-1)
  rank=tl.load(RANK+tl.maximum(h,0),outmask&(h>=0),0)
  distance=tl.load(DIST+tl.maximum(member,0)[None,:,:]*W+rank[:,:,None],valid,2.)
  nearest=tl.minimum(nearest,tl.where(h[:,:,None]>=0,distance,2.));chosen=chosen|(member[None,:,:]==h[:,:,None])
  eh=tl.sum(a*nearest,2);ehr=tl.sum(a*tl.where(chosen,0.,dr[None,:,:]),2)
  tl.store(E+base+2+slot,eh,outmask);tl.store(E+base+6+slot,ehr,outmask)

def buckets(population,width):
 out=[];lower=0
 for m in (1,2,4,8,16,32,64,128,256):
  if m>width:break
  idx=torch.nonzero((population>lower)&(population<=m),as_tuple=False).flatten().int().contiguous()
  if idx.numel():out.append((m,idx))
  lower=m
 return out

def losses(labels,hids,prior,geometry,clusters):
 if 'fast_buckets' not in geometry:raise RuntimeError('Bucket geometry missing')
 g,p=prior.shape;out=torch.empty((g,clusters,11),device=prior.device)
 for m,index in geometry['fast_buckets']:
  bg,bc=(4,16) if m<=16 else (2,8) if m<=64 else (1,4)
  _loss[(triton.cdiv(g,bg),triton.cdiv(len(index),bc))](geometry['members'],geometry['rep_distance'],geometry['distances'],geometry['rank'],hids,prior,index,out,p,clusters,g,len(index),geometry['width'],m,bg,bc,num_warps=4,enable_fp_fusion=False)
 return out
