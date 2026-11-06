"""Head-mean evidence from existing sparse QK;temporal conditional transfer.

The per-head coarse partition is a REFERENCE,not exact Dense normalization.
After averaging these reference-normalized scores across heads,we complete the
unobserved members by pooled-Key scores and normalize. This differs from the
mean of exact per-head Dense probabilities and must be evaluated as an estimate.
No per-head historical information survives the layer boundary.
"""
import torch,triton
import triton.language as tl
EPS=1.e-30

def observe(state,q,k,scale):
 """Only one current query/group x pooled regions and special Keys."""
 h,n,d=q.shape;c=state.config;g=state.groups;r=state.regions
 pooled=torch.zeros((h,r,d),device=k.device,dtype=torch.float32)
 pooled.index_add_(1,state.grid,k[:,state.patches].float());pooled/=state.grid_sizes[None,:,None]
 positions=(torch.arange(g,device=q.device)*c.query_group_size+c.query_group_size//8).clamp_max(n-1)
 qp=q[:,positions].float()
 raw=torch.bmm(qp,pooled.transpose(1,2))*scale
 special=torch.bmm(qp,k[:,state.specials].float().transpose(1,2))*scale
 logz=torch.logaddexp((raw+state.grid_sizes.log()).logsumexp(-1),special.logsumexp(-1))
 base=(raw-logz[:,:,None]).exp().mean(0).contiguous()
 return dict(logz=logz,base=base,coarse=base*state.grid_sizes,positions=positions)

@triton.jit
def _fill(BASE,GRID,OUT,N:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);r=tl.load(GRID+j,j<N,0)
 p=tl.load(BASE+g*R+r,j<N,0.)
 tl.store(OUT+g*N+j,p,j<N)

@triton.jit
def _overwrite(IDS,COND,COARSE,BETA,SIZE,GRID,OUT,G:tl.constexpr,N:tl.constexpr,R:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0)
 r=tl.load(GRID+tl.maximum(key,0),valid,0);c=tl.load(COND+g*A+j,valid,0.).to(tl.float32)
 p=tl.load(COARSE+g*R+r,valid,0.);b=tl.load(BETA+g*R+r,valid,0.);size=tl.load(SIZE+r,valid,1.)
 tl.store(OUT+g*N+key,p*(b*c+(1.-b)/size),valid)

def prior(state,observation,start,count):
 current=observation['coarse'][start:start+count];old=state.streamed_history_cache
 if old is None:
  base=observation['base'][start:start+count].contiguous();beta=None
 else:
  previous=old['coarse'][start:start+count]
  # Smooth symmetric confidence,1 for unchanged coarse observation. It is
  # already head-mean;there is no cross-layer matching of attention heads.
  beta=(2*(current*previous).clamp_min(0).sqrt()/(current+previous).clamp_min(EPS)).clamp(0,1)
  base=(current*(beta*old['uniform'][start:start+count]+(1-beta)/state.grid_sizes)).contiguous()
 n=len(state.patches);out=torch.empty((count,n),device=current.device,dtype=torch.float32)
 _fill[(count,triton.cdiv(n,256))](base,state.grid,out,n,state.regions,256,num_warps=4)
 if old is not None:
  ids=state.history_ids[start:start+count].contiguous();conditional=old['conditional'][start:start+count].contiguous()
  _overwrite[(count,triton.cdiv(ids.shape[1],256))](ids,conditional,current.contiguous(),beta.contiguous(),state.grid_sizes,state.grid,out,count,n,state.regions,ids.shape[1],256,num_warps=4)
 return out

@triton.jit
def _summarize(IDS,SCORE,GRID,MASS,COUNT,A:tl.constexpr,S:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0)
 region=tl.load(GRID+tl.maximum(key,0),valid,0);value=tl.load(SCORE+g*(S+A)+S+j,valid,0.)
 tl.atomic_add(MASS+g*R+region,value,valid,sem='relaxed');tl.atomic_add(COUNT+g*R+region,1,valid,sem='relaxed')

@triton.jit
def _condition(IDS,SCORE,GRID,MASS,OUT,A:tl.constexpr,S:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0)
 region=tl.load(GRID+tl.maximum(key,0),valid,0);value=tl.load(SCORE+g*(S+A)+S+j,valid,0.);total=tl.load(MASS+g*R+region,valid,1.)
 tl.store(OUT+g*A+j,tl.where(valid,value/tl.maximum(total,1.e-30),0.),j<A)

def finalize(state,observation,start,ids,scores):
 """One headless evidence summary and a direct FP16 conditional write."""
 g,a=ids.shape;r=state.regions;s=len(state.specials);base=observation['base'][start:start+g]
 mass=torch.zeros((g,r),device=ids.device);count=torch.zeros((g,r),device=ids.device,dtype=torch.int32)
 _summarize[(g,triton.cdiv(a,256))](ids,scores,state.grid,mass,count,a,s,r,256,num_warps=4)
 full=(mass+(state.grid_sizes[None]-count)*base).contiguous()
 den=(full.sum(1)+scores[:,:s].sum(1)).clamp_min(EPS)
 conditional=torch.empty((g,a),device=ids.device,dtype=torch.float16)
 _condition[(g,triton.cdiv(a,256))](ids,scores,state.grid,full,conditional,a,s,r,256,num_warps=4)
 return dict(conditional=conditional,uniform=(base/full.clamp_min(EPS)).contiguous(),coarse=observation['coarse'][start:start+g].clone(),estimated_grid_probability=full/den[:,None])
