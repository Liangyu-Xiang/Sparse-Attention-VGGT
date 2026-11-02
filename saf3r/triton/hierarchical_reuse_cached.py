"""Equivalent parent solver with reusable atom/state coverage tables."""
import torch
import triton
import triton.language as tl
from .hierarchical_reuse import _atoms

@triton.jit
def _coverage(E,PI,Z,MOMENT,REP,HIDS,PRIOR,PARENT,EXTREME,DIST,OWNER,
 P:tl.constexpr,F:tl.constexpr,D:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BD:tl.constexpr,EXACT:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1)
 fine,valid,safe,mass,key,kvalid,cross=_atoms(E,PI,Z,MOMENT,REP,HIDS,PRIOR,PARENT,EXTREME,F,4,D,N,S,BD,EXACT)
 atom=tl.arange(0,32);keyslot=tl.arange(0,32);source=keyslot//6;slot=keyslot%6;state=tl.arange(0,16)
 amount=tl.where((state>=2)&(state<=5),state-1,tl.where((state>=6)&(state<=9),state-5,0))
 active_rep=(slot[None,:]==0)&((state[:,None]==1)|((state[:,None]>=6)&(state[:,None]<=9))|(state[:,None]==10))
 active_hist=(slot[None,:]>=1)&(slot[None,:]<=4)&(slot[None,:]<=amount[:,None])
 active_exact=(slot[None,:]==5)&(state[:,None]==10)
 active=(active_rep|active_hist|active_exact)&kvalid[None,:]
 for child in tl.static_range(4):
  candidates=active&(source[None,:]==child)
  distances=tl.where(candidates[:,None,:],cross[None,:,:],float('inf'))
  best=tl.min(distances,2)
  owner=tl.min(tl.where(candidates[:,None,:]&(cross[None,:,:]==best[:,:,None]),key[None,None,:],2147483647),2)
  best=tl.where(mass[None,:]>0,best,0.)
  owner=tl.where((mass[None,:]>0)&(owner<2147483647),owner,-1)
  offset=(((group*P+parent)*20+atom[None,:])*4+child)*11+state[:,None]
  tl.store(DIST+offset,best,(atom[None,:]<20)&(state[:,None]<11))
  tl.store(OWNER+offset,owner,(atom[None,:]<20)&(state[:,None]<11))

@triton.jit
def _bank(E,C,BASE,PARENT,TEMPLATE,TABLE,OUTE,OUTC,
 P:tl.constexpr,F:tl.constexpr,S:tl.constexpr,T:tl.constexpr,BC:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1);candidate=tl.arange(0,BC);child=tl.arange(0,4);atom=tl.arange(0,32)
 fine=tl.load(PARENT+parent*4+child);valid=fine>=0;safe=tl.maximum(fine,0)
 template=tl.load(TEMPLATE+candidate[:,None]*4+child[None,:],candidate[:,None]<T,0)
 base=tl.load(BASE+group*F+safe,valid,0);state=tl.where(valid[None,:],tl.where(template<0,base[None,:],template),0)
 native=tl.load(E+(group*F+safe[None,:])*S+state,valid[None,:]&(candidate[:,None]<T),0.)
 cost=tl.load(C+(group*F+safe[None,:])*S+state,valid[None,:]&(candidate[:,None]<T),0.)
 best=tl.full((BC,32),float('inf'),tl.float32)
 for source in tl.static_range(4):
  source_state=tl.sum(tl.where(child[None,:]==source,state,0),1)
  distance=tl.load(TABLE+(((group*P+parent)*20+atom[None,:])*4+source)*11+source_state[:,None],(atom[None,:]<20)&(candidate[:,None]<T),float('inf'))
  best=tl.minimum(best,distance)
 total=tl.full((BC,),0.,tl.float32)
 for cc in tl.static_range(4):
  drop=tl.full((BC,),0.,tl.float32)
  for aa in tl.static_range(5):drop+=tl.sum(tl.where(atom[None,:]==cc*5+aa,best,0.),1)
  native_c=tl.sum(tl.where(child[None,:]==cc,native,0.),1)
  state_c=tl.sum(tl.where(child[None,:]==cc,state,0),1)
  drop_c=tl.sum(tl.where(child==cc,tl.load(E+(group*F+safe)*S,valid,0.),0))
  total+=tl.where(state_c>0,native_c,tl.minimum(drop_c,drop))
 tl.store(OUTE+(group*P+parent)*T+candidate,total,candidate<T)
 tl.store(OUTC+(group*P+parent)*T+candidate,tl.sum(cost,1),candidate<T)

@triton.jit
def _route(E,BASE,PARENT,TEMPLATE,TABLE,OWNERS,CHOICE,OUTSTATE,OUTOWNER,
 P:tl.constexpr,F:tl.constexpr,S:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1);child=tl.arange(0,4);atom=tl.arange(0,32)
 fine=tl.load(PARENT+parent*4+child);valid=fine>=0;safe=tl.maximum(fine,0)
 ci=tl.load(CHOICE+group*P+parent);template=tl.load(TEMPLATE+ci*4+child)
 base=tl.load(BASE+group*F+safe,valid,0);state=tl.where(valid,tl.where(template<0,base,template),0)
 best=tl.full((32,),float('inf'),tl.float32);owner=tl.full((32,),2147483647,tl.int32)
 for source in tl.static_range(4):
  ss=tl.sum(tl.where(child==source,state,0))
  off=(((group*P+parent)*20+atom)*4+source)*11+ss
  distance=tl.load(TABLE+off,atom<20,float('inf'));key=tl.load(OWNERS+off,atom<20,-1)
  use=(distance<best)|((distance==best)&(key>=0)&(key<owner))
  owner=tl.where(use,key,owner);best=tl.minimum(best,distance)
 drop=tl.full((4,),0.,tl.float32)
 for slot in tl.static_range(5):drop+=tl.sum(tl.where(atom[None,:]==child[:,None]*5+slot,best[None,:],0.),1)
 cdrop=tl.load(E+(group*F+safe)*S,valid,0.)
 delegated=(state==0)&(drop<cdrop)&valid
 atom_child=tl.minimum(atom//5,3);af=tl.gather(fine,atom_child,0)
 owner=tl.where((atom<20)&tl.gather(delegated,atom_child,0)&(owner>=0)&(owner<2147483647),owner,-1)
 tl.store(OUTSTATE+group*F+safe,state,valid)
 tl.store(OUTOWNER+(group*F+tl.maximum(af,0))*5+atom%5,owner,(atom<20)&(af>=0))

def coverage(errors,pi,z,reps,hids,prior,prepared,moment):
 g,f,s=errors.shape;p=len(prepared['parents']);dist=torch.empty((g,p,20,4,11),device=z.device);owner=torch.empty_like(dist,dtype=torch.int32)
 _coverage[(g,p)](errors,pi,z,pi if moment is None else moment,reps,hids,prior,prepared['parents'],prepared['extreme'],dist,owner,p,f,z.shape[-1],len(z),s,triton.next_power_of_2(z.shape[-1]),moment is not None,num_warps=8)
 return dist,owner

def bank(errors,costs,baseline,prepared,table):
 g,f,s=errors.shape;p=len(prepared['parents']);t=len(prepared['template']);e=torch.empty((g,p,t),device=errors.device);c=torch.empty_like(e)
 _bank[(g,p)](errors,costs,baseline,prepared['parents'],prepared['template'],table,e,c,p,f,s,t,triton.next_power_of_2(t),num_warps=8)
 return e,c

def route(errors,baseline,prepared,choice,table,owners):
 g,f,s=errors.shape;p=len(prepared['parents']);state=torch.empty_like(baseline);out=torch.full((*baseline.shape,5),-1,device=baseline.device,dtype=baseline.dtype)
 _route[(g,p)](errors,baseline,prepared['parents'],prepared['template'],table,owners,choice,state,out,p,f,s,num_warps=4)
 return state,out

def allocate(errors,costs,pi,z,reps,hids,prior,baseline,prepared,cap,iterations,passes=2,geometry=None,moment_mode='exact'):
 from .hierarchical_reuse import moments,choose
 moment=moments(z,prior,prepared['labels'],reps,geometry) if moment_mode=='exact' else None
 table,table_owner=coverage(errors,pi,z,reps,hids,prior,prepared,moment)
 state=baseline;records=[]
 for _ in range(passes):
  e,c=bank(errors,costs,state,prepared,table);selected=choose(e,c,cap,iterations)
  loss=e.gather(2,selected[:,:,None]).sum((1,2));base_loss=e[:,:,0].sum(1);count=c.gather(2,selected[:,:,None]).sum((1,2))
  fallback=(count>cap)|(loss>base_loss+1e-6*pi.sum(1))|~torch.isfinite(loss)
  selected=torch.where(fallback[:,None],0,selected);state,owners=route(errors,state,prepared,selected,table,table_owner)
  records.append(dict(fallback=fallback,proxy_loss=e.gather(2,selected[:,:,None]).sum((1,2)),baseline_proxy_loss=base_loss))
 return state,owners,records
