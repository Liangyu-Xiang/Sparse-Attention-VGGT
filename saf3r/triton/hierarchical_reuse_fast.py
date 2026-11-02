"""Parent candidate evaluation with scalar per-candidate lanes and cached coverage."""
import torch
import triton
import triton.language as tl
from . import hierarchical_reuse_cached as cached

@triton.jit
def _bank(E,C,BASE,PARENT,TEMPLATE,TABLE,OE,OC,
 P:tl.constexpr,F:tl.constexpr,S:tl.constexpr,T:tl.constexpr,BP:tl.constexpr,BC:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1)*BP+tl.arange(0,BP);ci=tl.arange(0,BC)
 total=tl.full((BP,BC),0.,tl.float32);count=tl.full((BP,BC),0.,tl.float32)
 f0=tl.load(PARENT+parent*4,(parent<P),-1);f1=tl.load(PARENT+parent*4+1,(parent<P),-1);f2=tl.load(PARENT+parent*4+2,(parent<P),-1);f3=tl.load(PARENT+parent*4+3,(parent<P),-1)
 b0=tl.load(BASE+group*F+tl.maximum(f0,0),f0>=0,0);b1=tl.load(BASE+group*F+tl.maximum(f1,0),f1>=0,0);b2=tl.load(BASE+group*F+tl.maximum(f2,0),f2>=0,0);b3=tl.load(BASE+group*F+tl.maximum(f3,0),f3>=0,0)
 t0=tl.load(TEMPLATE+ci*4,ci<T,0);t1=tl.load(TEMPLATE+ci*4+1,ci<T,0);t2=tl.load(TEMPLATE+ci*4+2,ci<T,0);t3=tl.load(TEMPLATE+ci*4+3,ci<T,0)
 s0=tl.where(f0[:,None]>=0,tl.where(t0[None,:]<0,b0[:,None],t0[None,:]),0);s1=tl.where(f1[:,None]>=0,tl.where(t1[None,:]<0,b1[:,None],t1[None,:]),0);s2=tl.where(f2[:,None]>=0,tl.where(t2[None,:]<0,b2[:,None],t2[None,:]),0);s3=tl.where(f3[:,None]>=0,tl.where(t3[None,:]<0,b3[:,None],t3[None,:]),0)
 mask=(parent[:,None]<P)&(ci[None,:]<T)
 for child in tl.static_range(4):
  fc=tl.load(PARENT+parent*4+child,parent<P,-1)
  state=s0 if child==0 else s1 if child==1 else s2 if child==2 else s3
  native=tl.load(E+(group*F+tl.maximum(fc,0)[:,None])*S+state,mask&(fc[:,None]>=0),0.)
  cost=tl.load(C+(group*F+tl.maximum(fc,0)[:,None])*S+state,mask&(fc[:,None]>=0),0.)
  ignored=tl.load(E+(group*F+tl.maximum(fc,0))*S,(parent<P)&(fc>=0),0.)
  drop=tl.full((BP,BC),0.,tl.float32)
  for a in tl.static_range(5):
   off=((group*P+parent[:,None])*20+child*5+a)*44
   d0=tl.load(TABLE+off+s0,mask,float('inf'));d1=tl.load(TABLE+off+11+s1,mask,float('inf'));d2=tl.load(TABLE+off+22+s2,mask,float('inf'));d3=tl.load(TABLE+off+33+s3,mask,float('inf'))
   drop+=tl.minimum(tl.minimum(d0,d1),tl.minimum(d2,d3))
  total+=tl.where(state>0,native,tl.minimum(ignored[:,None],drop));count+=cost
 tl.store(OE+(group*P+parent[:,None])*T+ci[None,:],total,mask);tl.store(OC+(group*P+parent[:,None])*T+ci[None,:],count,mask)

def bank(errors,costs,baseline,prepared,table):
 g,f,s=errors.shape;p=len(prepared['parents']);t=len(prepared['template']);e=torch.empty((g,p,t),device=errors.device);c=torch.empty_like(e)
 _bank[(g,triton.cdiv(p,4))](errors,costs,baseline,prepared['parents'],prepared['template'],table,e,c,p,f,s,t,4,128,num_warps=4,enable_fp_fusion=False)
 return e,c

def allocate(errors,costs,pi,z,reps,hids,prior,baseline,prepared,cap,iterations,passes=2,geometry=None,moment_mode='exact'):
 from .hierarchical_reuse import moments,choose
 moment=moments(z,prior,prepared['labels'],reps,geometry) if moment_mode=='exact' else None
 table,table_owner=coverage(errors,pi,z,reps,hids,prior,prepared,moment)
 state=baseline;records=[]
 for _ in range(passes):
  e,c=bank(errors,costs,state,prepared,table);selected=globals().get("fast_choose",choose)(e,c,cap,iterations)
  loss=e.gather(2,selected[:,:,None]).sum((1,2));base_loss=e[:,:,0].sum(1);count=c.gather(2,selected[:,:,None]).sum((1,2))
  fallback=(count>cap)|(loss>base_loss+1e-6*pi.sum(1))|~torch.isfinite(loss)
  selected=torch.where(fallback[:,None],0,selected);state,owners=cached.route(errors,state,prepared,selected,table,table_owner)
  records.append(dict(fallback=fallback,proxy_loss=e.gather(2,selected[:,:,None]).sum((1,2)),baseline_proxy_loss=base_loss))
 return state,owners,records

@triton.jit
def _atoms(E,PI,Z,MOMENT,REP,HIDS,PRIOR,PARENT,EXTREME,F:tl.constexpr,D:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BD:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1);child=tl.arange(0,4);fine=tl.load(PARENT+parent*4+child);valid=fine>=0;safe=tl.maximum(fine,0)
 atom=tl.arange(0,32);ac=atom//5;slot=atom%5;fc=tl.load(PARENT+parent*4+ac,ac<4,-1);av=(ac<4)&(fc>=0);fc=tl.maximum(fc,0);dd=tl.arange(0,BD)
 ah=tl.load(HIDS+(group*F+fc)*4+tl.maximum(slot-1,0),av&(slot>0),-1)
 ph=tl.load(PRIOR+group*N+tl.maximum(ah,0),av&(slot>0)&(ah>=0),0.)
 hz=tl.load(Z+tl.maximum(ah,0)[:,None]*D+dd[None,:],av[:,None]&(slot[:,None]>0)&(ah[:,None]>=0)&(dd[None,:]<D),0.)
 residual=tl.load(PI+group*F+fc,av&(slot==0),0.)
 mm=tl.load(MOMENT+(group*F+fc[:,None])*D+dd[None,:],av[:,None]&(slot[:,None]==0)&(dd[None,:]<D),0.)
 known=tl.full((32,BD),0.,tl.float32)
 for h in tl.static_range(4):
  kh=tl.load(HIDS+(group*F+fc)*4+h,av&(slot==0),-1)
  mass=tl.load(PRIOR+group*N+tl.maximum(kh,0),av&(slot==0)&(kh>=0),0.)
  zz=tl.load(Z+tl.maximum(kh,0)[:,None]*D+dd[None,:],av[:,None]&(slot[:,None]==0)&(kh[:,None]>=0)&(dd[None,:]<D),0.)
  residual-=mass;known+=mass[:,None]*zz
 mass=tl.where(slot==0,tl.maximum(residual,0.),ph)
 moment=tl.where(slot[:,None]==0,mm-known,ph[:,None]*hz)
 ks=atom//6;kk=atom%6;kc=tl.load(PARENT+parent*4+ks,ks<4,-1);kv=(ks<4)&(kc>=0);kc=tl.maximum(kc,0)
 kr=tl.load(REP+kc,kv,0);kh=tl.load(HIDS+(group*F+kc)*4+tl.maximum(kk-1,0),kv&(kk>=1)&(kk<=4),-1);ke=tl.load(EXTREME+kc,kv,0)
 key=tl.where(kk==0,kr,tl.where(kk==5,ke,kh));kv=kv&(key>=0)
 zv=tl.load(Z+tl.maximum(key,0)[None,:]*D+dd[:,None],kv[None,:]&(dd[:,None]<D),0.)
 cross=tl.maximum(mass[:,None]-tl.dot(moment,zv,allow_tf32=False),0.)
 return fine,valid,safe,mass,key,kv,cross

@triton.jit
def _coverage(E,PI,Z,MOMENT,REP,HIDS,PRIOR,PARENT,EXTREME,DIST,OWNER,
 P:tl.constexpr,F:tl.constexpr,D:tl.constexpr,N:tl.constexpr,S:tl.constexpr,BD:tl.constexpr):
 group=tl.program_id(0);parent=tl.program_id(1)
 fine,valid,safe,mass,key,kvalid,cross=_atoms(E,PI,Z,MOMENT,REP,HIDS,PRIOR,PARENT,EXTREME,F,D,N,S,BD)
 atom=tl.arange(0,32);col=tl.arange(0,32);state=tl.arange(0,16)
 for child in tl.static_range(4):
  dr=tl.sum(tl.where(col[None,:]==child*6,cross,0.),1);kr=tl.sum(tl.where(col==child*6,key,0))
  vr=tl.sum(tl.where(col==child*6,kvalid.to(tl.int32),0))>0;dr=tl.where(vr,dr,float('inf'));kr=tl.where(vr,kr,2147483647)
  best=tl.full((32,),float('inf'),tl.float32);owner=tl.full((32,),2147483647,tl.int32)
  result=tl.where(state[:,None]==1,dr[None,:],float('inf'));result_owner=tl.where(state[:,None]==1,kr,2147483647)
  for h in tl.static_range(4):
   kh=tl.sum(tl.where(col==child*6+h+1,key,0));vh=tl.sum(tl.where(col==child*6+h+1,kvalid.to(tl.int32),0))>0
   dh=tl.sum(tl.where(col[None,:]==child*6+h+1,cross,0.),1);dh=tl.where(vh,dh,float('inf'));kh=tl.where(vh,kh,2147483647)
   use=(dh<best)|((dh==best)&(kh<owner));owner=tl.where(use,kh,owner);best=tl.minimum(best,dh)
   result=tl.where(state[:,None]==h+2,best[None,:],result);result_owner=tl.where(state[:,None]==h+2,owner[None,:],result_owner)
   hr_owner=tl.where((dr<best)|((dr==best)&(kr<owner)),kr,owner)
   result=tl.where(state[:,None]==h+6,tl.minimum(dr,best)[None,:],result);result_owner=tl.where(state[:,None]==h+6,hr_owner[None,:],result_owner)
  ke=tl.sum(tl.where(col==child*6+5,key,0));ve=tl.sum(tl.where(col==child*6+5,kvalid.to(tl.int32),0))>0
  de=tl.sum(tl.where(col[None,:]==child*6+5,cross,0.),1);de=tl.where(ve,de,float('inf'));ke=tl.where(ve,ke,2147483647)
  eo=tl.where((dr<de)|((dr==de)&(kr<ke)),kr,ke)
  result=tl.where(state[:,None]==10,tl.minimum(dr,de)[None,:],result);result_owner=tl.where(state[:,None]==10,eo[None,:],result_owner)
  result=tl.where(mass[None,:]>0,result,0.);result_owner=tl.where((mass[None,:]>0)&(result_owner<2147483647),result_owner,-1)
  offset=(((group*P+parent)*20+atom[None,:])*4+child)*11+state[:,None]
  tl.store(DIST+offset,result,(atom[None,:]<20)&(state[:,None]<11));tl.store(OWNER+offset,result_owner,(atom[None,:]<20)&(state[:,None]<11))

def coverage(errors,pi,z,reps,hids,prior,prepared,moment):
 if moment is None:return cached.coverage(errors,pi,z,reps,hids,prior,prepared,moment)
 g,f,s=errors.shape;p=len(prepared['parents']);dist=torch.empty((g,p,20,4,11),device=z.device);owner=torch.empty_like(dist,dtype=torch.int32)
 _coverage[(g,p)](errors,pi,z,moment,reps,hids,prior,prepared['parents'],prepared['extreme'],dist,owner,p,f,z.shape[-1],len(z),s,triton.next_power_of_2(z.shape[-1]),num_warps=4,enable_fp_fusion=False)
 return dist,owner
