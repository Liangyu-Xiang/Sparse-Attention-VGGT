"""Optional cluster sums plus sparse history corrections; same loss formulas.

Explicit experimental opt-in. Floating-point reduction order changes.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _geometry(MEMBERS,Y,OUT,F:tl.constexpr,W:tl.constexpr,D:tl.constexpr,C:tl.constexpr,M:tl.constexpr):
 c=tl.program_id(0)*C+tl.arange(0,C);m=tl.arange(0,M);d=tl.arange(0,D)
 key=tl.load(MEMBERS+c[:,None]*W+m[None,:],c[:,None]<F,-1);valid=(c[:,None]<F)&(key>=0)
 y=tl.load(Y+tl.maximum(key,0)[:,:,None]*D+d[None,None,:],valid[:,:,None],0.)
 moment=tl.sum(y,1);norm=tl.sum(tl.sum(y*y,2),1)
 tl.store(OUT+c[:,None]*(D+1)+d[None,:],moment,c[:,None]<F)
 tl.store(OUT+c*(D+1)+D,norm,c<F)

def prepare(geometry):
 y=geometry['loss_patch_values'];f=len(geometry['members']);d=y.shape[-1];w=geometry['width'];c=max(1,min(16,128//w))
 result=torch.empty((f,d+1),device=y.device)
 _geometry[(triton.cdiv(f,c),)](geometry['members'],y,result,f,w,d,c,w,num_warps=4,enable_fp_fusion=False)
 geometry['loss_cluster_sums']=result
 return geometry

@triton.jit(do_not_specialize=['COUNT'])
def _stats(PRIOR,PI,POP,REP,MEMBERS,Y,OBS,QUAL,CLUSTERS,BASE,OUT,
 N:tl.constexpr,F:tl.constexpr,W:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,
 RESIDUAL:tl.constexpr,COUNT,C:tl.constexpr,M:tl.constexpr,COUNT_PTR:tl.constexpr):
 if COUNT_PTR:COUNT=tl.load(COUNT)
 g=tl.program_id(0);index=tl.program_id(1)*C+tl.arange(0,C)
 c=tl.load(CLUSTERS+index,index<COUNT,0);m=tl.arange(0,M);d=tl.arange(0,D)
 key=tl.load(MEMBERS+c[:,None]*W+m[None,:],index[:,None]<COUNT,-1)
 valid=(index[:,None]<COUNT)&(key>=0);safe=tl.maximum(key,0)
 known=tl.load(OBS+g*N+safe,valid,False)&(g%STRIDE==0)
 qualified=tl.load(QUAL+g*N+safe,valid,False)&valid
 p=tl.load(PRIOR+g*N+safe,known,0.)
 rep=tl.load(REP+c,index<COUNT,0);history=qualified
 if RESIDUAL:history=qualified&(key!=rep[:,None])
 mass=tl.load(PI+g*F+c,index<COUNT,0.);pop=tl.load(POP+c,index<COUNT,1.)
 known_mass=tl.sum(p,1);known_count=tl.sum(known.to(tl.int32),1)
 uniform=tl.maximum(0.,mass-known_mass)/tl.maximum(1.,pop-known_count)
 prob=tl.where(valid,tl.where(known,p,uniform[:,None]),0.)
 y=tl.load(Y+safe[:,:,None]*D+d[None,None,:],(known|history)[:,:,None],0.)
 base=tl.load(BASE+c[:,None]*(D+1)+d[None,:],index[:,None]<COUNT,0.)
 normbase=tl.load(BASE+c*(D+1)+D,index<COUNT,0.)
 correction=tl.where(known,p-uniform[:,None],0.)
 moment=uniform[:,None]*base+tl.sum(correction[:,:,None]*y,1)
 hp=tl.where(history,prob,0.);hm=tl.sum(hp,1);hvec=tl.sum(hp[:,:,None]*y,1)
 norm=tl.sum(y*y,2)
 a=uniform*normbase+tl.sum(correction*norm,1);ah=tl.sum(hp*norm,1)
 hc=tl.sum(history.to(tl.int32),1);available=tl.sum(qualified.to(tl.int32),1)
 L:tl.constexpr=2*D+6;off=(g*F+c)*L
 tl.store(OUT+off[:,None]+d[None,:],moment,index[:,None]<COUNT)
 tl.store(OUT+off[:,None]+D+d[None,:],hvec,index[:,None]<COUNT)
 tl.store(OUT+off+2*D,a,index<COUNT);tl.store(OUT+off+2*D+1,ah,index<COUNT)
 tl.store(OUT+off+2*D+2,hm,index<COUNT);tl.store(OUT+off+2*D+3,uniform,index<COUNT)
 tl.store(OUT+off+2*D+4,hc.to(tl.float32),index<COUNT);tl.store(OUT+off+2*D+5,available.to(tl.float32),index<COUNT)

def statistics(prior,pi,labels,reps,population,geometry,context):
 g,n=prior.shape;f=len(reps);d=geometry['loss_patch_values'].shape[-1]
 result=torch.empty((g,f,2*d+6),device=prior.device)
 for i,(size,clusters) in enumerate(geometry['drop_bins']):
  use_ptr='drop_bin_counts' in geometry;count=geometry['drop_bin_counts'][i] if use_ptr else len(clusters);c=min(16,max(1,128//size))
  _stats[(g,triton.cdiv(len(clusters),c))](prior,pi,population,reps,geometry['members'],geometry['loss_patch_values'],
   context['observed'],context['qualified'],clusters,geometry['loss_cluster_sums'],result,n,f,geometry['width'],d,
   geometry['loss_history_stride'],geometry['loss_operator']=='residual_rep',count,c,size,use_ptr,num_warps=4,enable_fp_fusion=False)
 return result

def enable():
 """Patch only this process; prepare cluster sums anew for each current layer."""
 from saf3r.patch import state_loss_controller as controller
 from saf3r.triton import state_loss_controller as kernel
 original=controller.prepare_geometry
 def prepare_current(*args,**kwargs):return prepare(original(*args,**kwargs))
 controller.prepare_geometry=prepare_current
 kernel.statistics=statistics
