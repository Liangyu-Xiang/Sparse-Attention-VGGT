"""Parent operating points: four fine clusters, Value-moment atoms,108 states."""
import torch
import triton
import triton.language as tl
from .reuse_compact import _choose,_dual

@triton.jit
def _atoms(E,PI,Z,MOMENT,REP,HIDS,PRIOR,PARENT,EXTREME,F:tl.constexpr,H:tl.constexpr,D:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BD:tl.constexpr,EXACT:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1)
 child=tl.arange(0,4);dd=tl.arange(0,BD);fine=tl.load(PARENT+parent*4+child);valid=fine>=0;safe=tl.maximum(fine,0)
 rep=tl.load(REP+safe);pi=tl.load(PI+group*F+safe)*valid
 er=tl.load(E+(group*F+safe)*S+1)*valid
 zr=tl.load(Z+rep[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0.)
 residual=pi;known_error=tl.full((4,),0.,tl.float32);known_vector=tl.full((4,BD),0.,tl.float32)
 atom=tl.arange(0,32);atom_child=atom//5;atom_slot=atom%5;atom_valid=atom_child<4
 ac=tl.load(PARENT+parent*4+atom_child,atom_valid,-1);av=atom_valid&(ac>=0);ac=tl.maximum(ac,0)
 ah=tl.load(HIDS+(group*F+ac)*H+tl.maximum(atom_slot-1,0),av&(atom_slot>0),-1)
 atom_mass=tl.load(PRIOR+group*N+tl.maximum(ah,0),av&(atom_slot>0)&(ah>=0),0.)
 atom_z=tl.load(Z+tl.maximum(ah,0)[:,None]*D+dd[None,:],av[:,None]&(atom_slot[:,None]>0)&(ah[:,None]>=0)&(dd[None,:]<D),0.)
 for hslot in tl.static_range(1,5):
  chosen=(atom_slot==hslot)&av
  mass=tl.sum(tl.where((atom_child[None,:]==child[:,None])&chosen[None,:],atom_mass[None,:],0.),1)
  vector=tl.sum(tl.where((atom_child[None,:,None]==child[:,None,None])&chosen[None,:,None],atom_mass[None,:,None]*atom_z[None,:,:],0.),1)
  residual-=mass;known_vector+=vector
  known_error+=mass-tl.sum(vector*zr,1)
 residual=tl.maximum(residual,0.)
 if EXACT:
  true_m=tl.load(MOMENT+(group*F+safe[:,None])*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0.)
  residual_m=true_m-known_vector
 else:
  intrinsic=tl.minimum(tl.maximum(er-known_error,0.),2.*residual)
  residual_m=(residual-intrinsic)[:,None]*zr
 residual_for_atom=tl.sum(tl.where(atom_child[:,None]==child[None,:],residual[None,:],0.),1)
 residual_vectors=tl.sum(tl.where(atom_child[:,None,None]==child[None,:,None],residual_m[None,:,:],0.),1)
 atom_mass=tl.where(atom_slot==0,residual_for_atom,atom_mass)
 mm=tl.where(atom_slot[:,None]==0,residual_vectors,atom_mass[:,None]*atom_z)
 mm=tl.where(av[:,None],mm,0.)
 keyslot=tl.arange(0,32);source=keyslot//6;slot=keyslot%6;source_valid=source<4
 cid=tl.load(PARENT+parent*4+source,source_valid,-1);kidvalid=source_valid&(cid>=0);cid=tl.maximum(cid,0)
 kr=tl.load(REP+cid,kidvalid,0);kh=tl.load(HIDS+(group*F+cid)*H+tl.maximum(slot-1,0),kidvalid&(slot>=1)&(slot<=4),-1)
 ke=tl.load(EXTREME+cid,kidvalid,0);key=tl.where(slot==0,kr,tl.where(slot==5,ke,kh));kidvalid=kidvalid&(key>=0)
 zv=tl.load(Z+tl.maximum(key,0)[None,:]*D+dd[:,None],kidvalid[None,:]&(dd[:,None]<D),0.)
 dot=tl.dot(mm,zv,allow_tf32=False)
 cross=tl.maximum(atom_mass[:,None]-dot,0.)
 return fine,valid,safe,atom_mass,key,kidvalid,cross

@triton.jit
def _parent(E,C,PI,Z,MOMENT,REP,HIDS,PRIOR,BASE,PARENT,EXTREME,TEMPLATE,CHOICE,OUTE,OUTC,OUTSTATE,OWNER,
 P:tl.constexpr,F:tl.constexpr,H:tl.constexpr,D:tl.constexpr,N:tl.constexpr,S:tl.constexpr,T:tl.constexpr,BD:tl.constexpr,BC:tl.constexpr,WRITE:tl.constexpr,EXACT:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1)
 fine,valid,safe,atom_mass,key,kidvalid,cross=_atoms(E,PI,Z,MOMENT,REP,HIDS,PRIOR,PARENT,EXTREME,F,H,D,N,S,BD,EXACT)
 child=tl.arange(0,4);atom=tl.arange(0,32);source=atom//6;slot=atom%6
 candidate=tl.arange(0,BC)
 selected=tl.load(CHOICE+group*P+parent) if WRITE else 0
 ci=tl.full((BC,),selected,tl.int32) if WRITE else candidate
 template=tl.load(TEMPLATE+ci[:,None]*4+child[None,:],(ci[:,None]<T),0)
 base=tl.load(BASE+group*F+safe,valid,0)
 state=tl.where(valid[None,:],tl.where(template<0,base[None,:],template),0)
 native=tl.load(E+(group*F+safe[None,:])*S+state,valid[None,:]&(ci[:,None]<T),0.)
 cost=tl.load(C+(group*F+safe[None,:])*S+state,valid[None,:]&(ci[:,None]<T),0.)
 ks=tl.gather(state,tl.broadcast_to(tl.minimum(source,3)[None,:],(BC,32)),1)
 active_rep=(slot[None,:]==0)&((ks==1)|((ks>=6)&(ks<=9))|(ks==10))
 amount=tl.where((ks>=2)&(ks<=5),ks-1,tl.where((ks>=6)&(ks<=9),ks-5,0))
 active_hist=(slot[None,:]>=1)&(slot[None,:]<=4)&(slot[None,:]<=amount)
 active_exact=(slot[None,:]==5)&(ks==10)
 active=(active_rep|active_hist|active_exact)&kidvalid[None,:]
 total=tl.full((BC,),0.,tl.float32)
 for c in tl.static_range(4):
  c_native=tl.sum(tl.where(child[None,:]==c,native,0.),1)
  c_state=tl.sum(tl.where(child[None,:]==c,state,0),1)
  c_pi=tl.sum(tl.where(child==c,tl.load(E+(group*F+safe)*S,valid,0.),0))
  drop=tl.full((BC,),0.,tl.float32)
  for a in tl.static_range(5):
   aa=c*5+a
   distance=tl.sum(tl.where(atom[:,None]==aa,cross,0.),0)
   mass=tl.sum(tl.where(atom==aa,atom_mass,0.))
   best=tl.min(tl.where(active,distance[None,:],float('inf')),1)
   drop+=tl.where(mass>0,best,0.)
  use_delegate=(c_state==0)&(drop<c_pi)
  total+=tl.where(c_state>0,c_native,tl.minimum(c_pi,drop))
  if WRITE:
   fc=tl.sum(tl.where(child==c,fine,0))
   tl.store(OUTSTATE+group*F+tl.maximum(fc,0),tl.sum(c_state,0),fc>=0)
   for a in tl.static_range(5):
    aa=c*5+a
    distance=tl.sum(tl.where(atom[:,None]==aa,cross,0.),0)
    mass=tl.sum(tl.where(atom==aa,atom_mass,0.))
    best=tl.min(tl.where(active,distance[None,:],float('inf')),1)
    owner=tl.min(tl.where(active&(distance[None,:]==best[:,None]),key[None,:],2147483647),1)
    owner=tl.where(use_delegate&(mass>0)&(owner<2147483647),owner,-1)
    tl.store(OWNER+(group*F+tl.maximum(fc,0))*5+a,tl.sum(owner,0),fc>=0)
 if not WRITE:
  tl.store(OUTE+(group*P+parent)*T+candidate,total,candidate<T)
  tl.store(OUTC+(group*P+parent)*T+candidate,tl.sum(cost,1),candidate<T)

@triton.jit
def _route(E,PI,Z,MOMENT,REP,HIDS,PRIOR,BASE,PARENT,EXTREME,TEMPLATE,CHOICE,OUTSTATE,OWNER,
 P:tl.constexpr,F:tl.constexpr,H:tl.constexpr,D:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BD:tl.constexpr,EXACT:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1)
 fine,valid,safe,mass,key,kvalid,cross=_atoms(E,PI,Z,MOMENT,REP,HIDS,PRIOR,PARENT,EXTREME,F,H,D,N,S,BD,EXACT)
 child=tl.arange(0,4);a=tl.arange(0,32);source=a//6;slot=a%6
 ci=tl.load(CHOICE+group*P+parent);template=tl.load(TEMPLATE+ci*4+child)
 base=tl.load(BASE+group*F+safe,valid,0);state=tl.where(valid,tl.where(template<0,base,template),0)
 ks=tl.gather(state,tl.minimum(source,3),0)
 amount=tl.where((ks>=2)&(ks<=5),ks-1,tl.where((ks>=6)&(ks<=9),ks-5,0))
 active=kvalid&(((slot==0)&((ks==1)|((ks>=6)&(ks<=9))|(ks==10)))|((slot>=1)&(slot<=4)&(slot<=amount))|((slot==5)&(ks==10)))
 best=tl.min(tl.where(active[None,:],cross,float('inf')),1)
 atom_cost=tl.where(mass>0,best,0.)
 total=tl.sum(tl.where((a[None,:]//5==child[:,None])&(a[None,:]<20),atom_cost[None,:],0.),1)
 drop=tl.load(E+(group*F+safe)*S,valid,0.)
 delegate=(state==0)&(total<drop)&valid
 owner=tl.min(tl.where(active[None,:]&(cross==best[:,None]),key[None,:],2147483647),1)
 owner=tl.where((a<20)&(mass>0)&tl.gather(delegate,tl.minimum(a//5,3),0)&(owner<2147483647),owner,-1)
 atom_fine=tl.gather(fine,tl.minimum(a//5,3),0)
 tl.store(OUTSTATE+group*F+safe,state,valid)
 tl.store(OWNER+(group*F+tl.maximum(atom_fine,0))*5+a%5,owner,(a<20)&(atom_fine>=0))

def bank(errors,costs,pi,z,reps,hids,prior,baseline,prepared,exact_moment=None):
 g,f,s=errors.shape;p=len(prepared['parents']);t=len(prepared['template']);e=torch.empty((g,p,t),device=z.device);c=torch.empty_like(e)
 _parent[(g,p)](errors,costs,pi,z,pi if exact_moment is None else exact_moment,reps,hids,prior,baseline,prepared['parents'],prepared['extreme'],prepared['template'],baseline,e,c,baseline,baseline,p,f,4,z.shape[-1],len(z),s,t,triton.next_power_of_2(z.shape[-1]),triton.next_power_of_2(t),False,exact_moment is not None,num_warps=8)
 return e,c

def choose(errors,costs,cap,iterations):
 g,p,t=errors.shape;tiles=triton.cdiv(p,32)
 # Valid zero-cost all-drop candidates provide a simple finite upper bound.
 upper=torch.where(costs==0,errors,0).amax((1,2))+1e-12
 dual=torch.stack((torch.zeros_like(upper),upper),1).contiguous();part=torch.empty((g,tiles),device=errors.device);selected=torch.empty((g,p),device=errors.device,dtype=torch.int64)
 for _ in range(iterations):
  _choose[(g,tiles)](errors,costs,dual,part,selected,p,t,tiles,False,32,triton.next_power_of_2(t),num_warps=4,enable_fp_fusion=False)
  _dual[(g,)](part,dual,tiles,cap,triton.next_power_of_2(tiles),num_warps=4,enable_fp_fusion=False)
 _choose[(g,tiles)](errors,costs,dual,part,selected,p,t,tiles,True,32,triton.next_power_of_2(t),num_warps=4,enable_fp_fusion=False)
 return selected

def route(errors,costs,pi,z,reps,hids,prior,baseline,prepared,choice,exact_moment=None):
 g,f,s=errors.shape;p=len(prepared['parents']);t=len(prepared['template']);state=torch.empty_like(baseline);owner=torch.full((*baseline.shape,5),-1,device=baseline.device,dtype=baseline.dtype)
 _route[(g,p)](errors,pi,z,pi if exact_moment is None else exact_moment,reps,hids,prior,baseline,prepared['parents'],prepared['extreme'],prepared['template'],choice,state,owner,p,f,4,z.shape[-1],len(z),s,triton.next_power_of_2(z.shape[-1]),exact_moment is not None,num_warps=4)
 return state,owner

@triton.jit
def _moments(Z,PRIOR,MEM,OUT,F:tl.constexpr,P:tl.constexpr,D:tl.constexpr,M:tl.constexpr,BD:tl.constexpr):
 group=tl.program_id(0);cluster=tl.program_id(1);col=tl.arange(0,M);dd=tl.arange(0,BD)
 ids=tl.load(MEM+cluster*M+col)
 a=tl.load(PRIOR+group*P+tl.maximum(ids,0),ids>=0,0.)
 z=tl.load(Z+tl.maximum(ids,0)[:,None]*D+dd[None,:],(ids[:,None]>=0)&(dd[None,:]<D),0.)
 moment=tl.sum(a[:,None]*z,0)
 tl.store(OUT+(group*F+cluster)*D+dd,moment,dd<D)

def moments(z,prior,labels,reps,geometry):
 g,p=prior.shape;f=len(reps);d=z.shape[-1]
 out=torch.empty((g,f,d),device=z.device)
 if geometry is None:
  for group in range(g):
   out[group].zero_();out[group].index_add_(0,labels,prior[group,:,None]*z)
 else:
  _moments[(g,f)](z,prior,geometry['members'],out,f,p,d,geometry['width'],triton.next_power_of_2(d),num_warps=4)
 return out
