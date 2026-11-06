"""Low-cost current observations and dense head-mean temporal probability cache."""
import torch,triton
import triton.language as tl

@triton.jit
def _coarse_lse(Q,K,BIAS,Z,M:tl.constexpr,N:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
 t=tl.program_id(0);h=tl.program_id(1);row=t*BM+tl.arange(0,BM);dd=tl.arange(0,BD);kk=tl.arange(0,BK);q=tl.load(Q+h*QH+row[:,None]*D+dd[None,:],(row[:,None]<M)&(dd[None,:]<D),0.)
 maximum=tl.full((BM,),-float('inf'),tl.float32);total=tl.zeros((BM,),tl.float32)
 for tile in range(tl.cdiv(N,BK)):
  key=tile*BK+kk;k=tl.load(K+h*KH+key[None,:]*D+dd[:,None],(key[None,:]<N)&(dd[:,None]<D),0.);bias=tl.load(BIAS+key,key<N,0.)
  raw=tl.where(key[None,:]<N,tl.dot(q,k,allow_tf32=False)*SCALE+bias[None,:],-float('inf'));new=tl.maximum(maximum,tl.max(raw,1));total=total*tl.exp(maximum-new)+tl.sum(tl.exp(raw-new[:,None]),1);maximum=new
 tl.store(Z+h*M+row,maximum+tl.log(total),row<M)

@triton.jit
def _coarse_mean(Q,K,BIAS,Z,OUT,G:tl.constexpr,H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,D:tl.constexpr,P:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BG:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
 g=tl.program_id(0)*BG+tl.arange(0,BG);rows=tl.program_id(0)*BG*P+tl.arange(0,BG*P);key=tl.program_id(1)*BK+tl.arange(0,BK);dd=tl.arange(0,BD);total=tl.zeros((BG*P,BK),tl.float32);bias=tl.load(BIAS+key,key<R,0.)
 for h in range(H):
  q=tl.load(Q+h*QH+rows[:,None]*D+dd[None,:],(rows[:,None]<G*P)&(dd[None,:]<D),0.);k=tl.load(K+h*KH+key[None,:]*D+dd[:,None],(key[None,:]<R)&(dd[:,None]<D),0.);z=tl.load(Z+h*G*P+rows,rows<G*P,0.)
  total+=tl.exp(tl.dot(q,k,allow_tf32=False)*SCALE+bias[None,:]-z[:,None])
 result=tl.sum(total.reshape((BG,P,BK)),1)/(H*P)
 tl.store(OUT+g[:,None]*R+key[None,:],result,(g[:,None]<G)&(key[None,:]<R))

@triton.jit
def _anchor_mean(Q,K,VALID,OUT,G:tl.constexpr,H:tl.constexpr,R:tl.constexpr,M:tl.constexpr,D:tl.constexpr,P:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BG:tl.constexpr,BR:tl.constexpr,BD:tl.constexpr):
 group=tl.program_id(0)*BG+tl.arange(0,BG);row=tl.program_id(0)*BG*P+tl.arange(0,BG*P);key=tl.program_id(1)*BR*M+tl.arange(0,BR*M);region=tl.program_id(1)*BR+tl.arange(0,BR);dd=tl.arange(0,BD);valid=tl.load(VALID+key,key<R*M,False);total=tl.zeros((BG*P,BR*M),tl.float32)
 for h in range(H):
  q=tl.load(Q+h*QH+row[:,None]*D+dd[None,:],(row[:,None]<G*P)&(dd[None,:]<D),0.);k=tl.load(K+h*KH+key[None,:]*D+dd[:,None],(key[None,:]<R*M)&(dd[:,None]<D),0.)
  raw=tl.where(valid[None,:],tl.dot(q,k,allow_tf32=False)*SCALE,-float('inf')).reshape((BG*P,BR,M));maximum=tl.max(raw,2);safe=tl.where(maximum==-float('inf'),0.,maximum);p=tl.exp(raw-safe[:,:,None]);p=p/tl.maximum(tl.sum(p,2)[:,:,None],1.e-30);total+=p.reshape((BG*P,BR*M))
 result=tl.sum(total.reshape((BG,P,BR,M)),1)/(H*P);slots=tl.arange(0,M)
 tl.store(OUT+(group[:,None,None]*R+region[None,:,None])*M+slots[None,None,:],tl.log(tl.maximum(result,1.e-30)),(group[:,None,None]<G)&(region[None,:,None]<R))

def observe(q,prepared,start,count,group_size,probes,scale):
 h,n,d=q.shape;positions=(torch.arange(start,start+count,device=q.device)[:,None]*group_size+torch.arange(probes,device=q.device)[None]*(group_size//probes)+group_size//8).clamp_max(n-1)
 qp=q[:,positions.flatten()].to(torch.float16).contiguous();bank=prepared['bank'];r=prepared['regions'];z=torch.empty((h,count*probes),device=q.device);coarse=torch.empty((count,r),device=q.device);logs=torch.empty((count,r,prepared['samples']),device=q.device)
 _coarse_lse[(triton.cdiv(count*probes,32),h)](qp,bank,prepared['bias'],z,count*probes,bank.shape[1],d,qp.stride(0),bank.stride(0),scale,32,128,triton.next_power_of_2(d),num_warps=4,num_stages=2)
 bg=16//probes
 _coarse_mean[(triton.cdiv(count,bg),triton.cdiv(r,128))](qp,bank,prepared['bias'],z,coarse,count,h,r,bank.shape[1],d,probes,qp.stride(0),bank.stride(0),scale,bg,128,triton.next_power_of_2(d),num_warps=4)
 _anchor_mean[(triton.cdiv(count,bg),triton.cdiv(r,16))](qp,prepared['anchor_keys'],prepared['valid'],logs,count,h,r,prepared['samples'],d,probes,qp.stride(0),prepared['anchor_keys'].stride(0),scale,bg,16,triton.next_power_of_2(d),num_warps=4)
 return dict(coarse=coarse,sample_logprob=logs,positions=positions)

@triton.jit
def _transport(OLD_MASS,OLD_COARSE,CUR,TARGET,BETA,G:tl.constexpr,R:tl.constexpr,BR:tl.constexpr):
 g=tl.program_id(0);r=tl.arange(0,BR);valid=r<R;old=tl.load(OLD_MASS+g*R+r,valid,0.);prev=tl.load(OLD_COARSE+g*R+r,valid,0.);cur=tl.load(CUR+g*R+r,valid,0.)
 patch=old*cur/tl.maximum(prev,1.e-30);special=tl.maximum(0.,1.-tl.sum(old,0))*tl.maximum(0.,1.-tl.sum(cur,0))/tl.maximum(1.-tl.sum(prev,0),1.e-30);den=tl.sum(patch,0)+special
 tl.store(TARGET+g*R+r,patch/tl.maximum(den,1.e-30),valid);tl.store(BETA+g*R+r,tl.minimum(1.,2*tl.sqrt(tl.maximum(cur*prev,0.))/tl.maximum(cur+prev,1.e-30)),valid)

@triton.jit
def _predict(OLD,OLD_LOG,CUR_LOG,ASSIGN,SIZE,TARGET,BETA,CUR,OUT,MIXED,N:tl.constexpr,R:tl.constexpr,M:tl.constexpr,BLOCK:tl.constexpr,BR:tl.constexpr,WEIGHT:tl.constexpr,GAIN:tl.constexpr):
 g=tl.program_id(0);r=tl.program_id(1)*BR+tl.arange(0,BR);j=r[:,None]*BLOCK+tl.arange(0,BLOCK)[None,:];valid=(r[:,None]<R)&(j<N)
 phi=tl.load(OLD+g*N+j,valid,0.);anchor=tl.load(ASSIGN+j,valid,0);oldlog=tl.load(OLD_LOG+(g*R+r[:,None])*M+anchor,valid,0.);newlog=tl.load(CUR_LOG+(g*R+r[:,None])*M+anchor,valid,0.)
 tilt=phi*tl.exp(tl.minimum(20.,tl.maximum(-20.,GAIN*(newlog-oldlog))));total=tl.sum(tilt,1);size=tl.load(SIZE+r,r<R,1.);mass=tl.load(TARGET+g*R+r,r<R,0.);beta=tl.load(BETA+g*R+r,r<R,0.);cur=tl.load(CUR+g*R+r,r<R,0.)
 cond=tl.where(total[:,None]>1.e-30,tilt/tl.maximum(total[:,None],1.e-30),1./size[:,None]);result=WEIGHT*mass[:,None]*(beta[:,None]*cond+(1.-beta[:,None])/size[:,None])+(1.-WEIGHT)*cur[:,None]/size[:,None]
 tl.store(OUT+g*N+j,result,valid);tl.store(MIXED+g*R+r,WEIGHT*mass+(1.-WEIGHT)*cur,r<R)

def predict(previous,observation,assignment,sizes,block,weight,gain):
 old=previous['dense_probability'];g,n=old.shape;r=len(sizes);target=torch.empty((g,r),device=old.device);beta=torch.empty_like(target);out=torch.empty_like(old);mixed=torch.empty_like(target)
 _transport[(g,)](previous['block_mass'],previous['coarse'],observation['coarse'],target,beta,g,r,triton.next_power_of_2(r),num_warps=4)
 _predict[(g,triton.cdiv(r,4))](old,previous['sample_logprob'],observation['sample_logprob'],assignment,sizes,target,beta,observation['coarse'],out,mixed,n,r,observation['sample_logprob'].shape[-1],block,4,weight,gain,num_warps=4)
 return out,mixed

@triton.jit
def _selected_mass(PHI,IDS,PART,N:tl.constexpr,A:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);tile=tl.program_id(1);j=tile*B+tl.arange(0,B);key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0);p=tl.load(PHI+g*N+tl.maximum(key,0),valid,0.);tl.store(PART+g*T+tile,tl.sum(p,0))

@triton.jit
def _finish_mass(PART,BLOCK_MASS,MASS,R:tl.constexpr,T:tl.constexpr,BR:tl.constexpr,BT:tl.constexpr):
 g=tl.program_id(0);r=tl.arange(0,BR);t=tl.arange(0,BT);patch=tl.sum(tl.load(BLOCK_MASS+g*R+r,r<R,0.),0);selected=tl.sum(tl.load(PART+g*T+t,t<T,0.),0)
 tl.store(MASS+g,tl.minimum(1.-1.e-7,tl.maximum(1.e-7,1.-patch+selected)))

@triton.jit
def _posterior(RAW,SHIFT,DEN,IDS,MASS,PHI,G:tl.constexpr,H:tl.constexpr,P:tl.constexpr,N:tl.constexpr,A:tl.constexpr,S:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BH:tl.constexpr,ETA:tl.constexpr):
 tile=tl.program_id(0);g=tl.program_id(1);key=tile*B+tl.arange(0,B);h=tl.arange(0,BH);p=tl.arange(0,P)
 raw=tl.load(RAW+((g*H+h[:,None,None])*P+p[None,:,None])*(S+A)+key[None,None,:],(h[:,None,None]<H)&(key[None,None,:]<S+A),-float('inf')).to(tl.float32)
 shift=tl.load(SHIFT+((g*H+h[:,None])*P+p[None,:])*T+tile,h[:,None]<H,0.);den=tl.load(DEN+(h[:,None]*G+g)*P+p[None,:],h[:,None]<H,0.)
 prob=tl.sum(tl.sum(tl.where(h[:,None,None]<H,tl.exp(raw+shift[:,:,None]-den[:,:,None]),0.),0),0)/(H*P)
 j=tl.load(IDS+g*A+key-S,(key>=S)&(key<S+A),-1);valid=(key>=S)&(key<S+A)&(j>=0);mass=tl.load(MASS+g);observed=mass*prob
 if ETA==1.:updated=observed
 elif ETA==.5:
  prior=tl.load(PHI+g*N+tl.maximum(j,0),valid,0.);updated=tl.sqrt(tl.maximum(prior,1.e-30)*tl.maximum(observed,1.e-30))
 else:
  prior=tl.load(PHI+g*N+tl.maximum(j,0),valid,0.);updated=tl.exp((1.-ETA)*tl.log(tl.maximum(prior,1.e-30))+ETA*tl.log(tl.maximum(observed,1.e-30)))
 tl.store(PHI+g*N+j,updated,valid)

@triton.jit
def _normalize(PHI,TARGET,OUT,N:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr,BR:tl.constexpr):
 g=tl.program_id(0);r=tl.program_id(1)*BR+tl.arange(0,BR);j=r[:,None]*BLOCK+tl.arange(0,BLOCK)[None,:];valid=(r[:,None]<R)&(j<N);p=tl.load(PHI+g*N+j,valid,0.);total=tl.sum(p,1);size=tl.minimum(BLOCK,N-r*BLOCK);mass=tl.load(TARGET+g*R+r,r<R,0.)
 result=tl.where(total[:,None]>1.e-30,p/tl.maximum(total[:,None],1.e-30),1./size[:,None])*mass[:,None];tl.store(OUT+g*N+j,result,valid)

def update(prior,target_mass,ids,evidence,destination,specials,block,eta):
 g,n=prior.shape;a=ids.shape[1];r=target_mass.shape[1];h,_,p=evidence['raw_lse'].shape;t=triton.cdiv(a,256);partial=torch.empty((g,t),device=prior.device);mass=torch.empty(g,device=prior.device)
 _selected_mass[(g,t)](prior,ids,partial,n,a,t,256,num_warps=4);_finish_mass[(g,)](partial,target_mass,mass,r,t,triton.next_power_of_2(r),triton.next_power_of_2(t),num_warps=4)
 tiles=triton.cdiv(specials+a,128)
 _posterior[(tiles,g)](evidence['encoded'],evidence['shifts'],evidence['raw_lse'],ids,mass,prior,g,h,p,n,a,specials,tiles,128,triton.next_power_of_2(h),eta,num_warps=4)
 _normalize[(g,triton.cdiv(r,4))](prior,target_mass,destination,n,r,block,4,num_warps=4)

@triton.jit
def _block_mass(PHI,OUT,N:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr,BR:tl.constexpr):
 g=tl.program_id(0);r=tl.program_id(1)*BR+tl.arange(0,BR);j=r[:,None]*BLOCK+tl.arange(0,BLOCK)[None,:];p=tl.load(PHI+g*N+j,(r[:,None]<R)&(j<N),0.);tl.store(OUT+g*R+r,tl.sum(p,1),r<R)

def block_mass(phi,block):
 g,n=phi.shape;r=triton.cdiv(n,block);out=torch.empty((g,r),device=phi.device);_block_mass[(g,triton.cdiv(r,4))](phi,out,n,r,block,4,num_warps=4);return out
