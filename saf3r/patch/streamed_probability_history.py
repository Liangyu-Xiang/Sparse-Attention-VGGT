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
 block=getattr(c,'history_key_block_size',0);patch=k[:,state.patches].float()
 if block:
  pooled=torch.empty((h,r,d),device=k.device,dtype=torch.float32);full=len(state.patches)//block
  if full:pooled[:,:full]=patch[:,:full*block].reshape(h,full,block,d).mean(2)
  if len(state.patches)%block:pooled[:,-1]=patch[:,full*block:].mean(1)
 else:
  pooled=torch.zeros((h,r,d),device=k.device,dtype=torch.float32)
  pooled.index_add_(1,state.grid,patch);pooled/=state.grid_sizes[None,:,None]
 positions=(torch.arange(g,device=q.device)*c.query_group_size+c.query_group_size//8).clamp_max(n-1)
 qp=q[:,positions].float()
 raw=torch.bmm(qp,pooled.transpose(1,2))*scale
 special=torch.bmm(qp,k[:,state.specials].float().transpose(1,2))*scale
 logz=torch.logaddexp((raw+state.grid_sizes.log()).logsumexp(-1),special.logsumexp(-1))
 base=(raw-logz[:,:,None]).exp().mean(0).contiguous()
 return dict(logz=logz,base=base,coarse=base*state.grid_sizes,positions=positions,raw_grid_logits=raw)

@triton.jit
def _fill(BASE,GRID,OUT,N:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);r=tl.load(GRID+j,j<N,0)
 p=tl.load(BASE+g*R+r,j<N,0.)
 tl.store(OUT+g*N+j,p,j<N)

@triton.jit
def _overwrite(IDS,COND,COARSE,BETA,SIZE,GRID,BASE,OUT,G:tl.constexpr,N:tl.constexpr,R:tl.constexpr,A:tl.constexpr,B:tl.constexpr,WEIGHT:tl.constexpr):
 g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0)
 r=tl.load(GRID+tl.maximum(key,0),valid,0);c=tl.load(COND+g*A+j,valid,0.).to(tl.float32)
 p=tl.load(COARSE+g*R+r,valid,0.);b=tl.load(BETA+g*R+r,valid,0.);size=tl.load(SIZE+r,valid,1.)
 current=tl.load(BASE+g*R+r,valid,0.)
 tl.store(OUT+g*N+key,WEIGHT*p*(b*c+(1.-b)/size)+(1.-WEIGHT)*current,valid)

def current_mass(state,observation,start,count,transport):
 current=observation['coarse'][start:start+count];old=state.streamed_history_cache
 if old is None or not transport:return current
 previous=old['coarse'][start:start+count];full=old['estimated_grid_probability'][start:start+count]
 patch=full*(current/previous.clamp_min(EPS))
 special=(1-full.sum(1)).clamp_min(0)*(1-current.sum(1)).clamp_min(0)/(1-previous.sum(1)).clamp_min(EPS)
 return patch/(patch.sum(1)+special).clamp_min(EPS)[:,None]

def prior(state,observation,start,count):
 coarse=observation['coarse'][start:start+count];old=state.streamed_history_cache
 current=current_mass(state,observation,start,count,getattr(getattr(state,'config',None),'streamed_history_region_transport',False))
 if old is None:
  base=observation['base'][start:start+count].contiguous();beta=None
 else:
  previous=old['coarse'][start:start+count]
  # Smooth symmetric confidence,1 for unchanged coarse observation. It is
  # already head-mean;there is no cross-layer matching of attention heads.
  beta=(2*(coarse*previous).clamp_min(0).sqrt()/(coarse+previous).clamp_min(EPS)).clamp(0,1)
  weight=getattr(getattr(state,'config',None),'streamed_history_prior_weight',1.)
  base=(weight*current*(beta*old['uniform'][start:start+count]+(1-beta)/state.grid_sizes)+(1-weight)*observation['base'][start:start+count]).contiguous()
 n=len(state.patches);out=torch.empty((count,n),device=current.device,dtype=torch.float32)
 _fill[(count,triton.cdiv(n,256))](base,state.grid,out,n,state.regions,256,num_warps=4)
 if old is not None:
  ids=state.history_ids[start:start+count].contiguous();conditional=old['conditional'][start:start+count].contiguous()
  _overwrite[(count,triton.cdiv(ids.shape[1],256))](ids,conditional,current.contiguous(),beta.contiguous(),state.grid_sizes,state.grid,observation['base'][start:start+count].contiguous(),out,count,n,state.regions,ids.shape[1],256,weight,num_warps=4)
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

@triton.jit
def _counts(IDS,GRID,COUNT,A:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0)
 region=tl.load(GRID+tl.maximum(key,0),valid,0);tl.atomic_add(COUNT+g*R+region,1,valid,sem='relaxed')

def missing_normalizer(state,observation,start,ids):
 g,a=ids.shape;r=state.regions;count=torch.zeros((g,r),device=ids.device,dtype=torch.int32)
 _counts[(g,triton.cdiv(a,256))](ids,state.grid,count,a,r,256,num_warps=4)
 remaining=state.grid_sizes-count
 logits=observation['raw_grid_logits'][:,start:start+g]
 missing=(logits+remaining.clamp_min(1).log()[None]).masked_fill(remaining[None]<=0,-torch.inf).logsumexp(-1)
 return dict(logz=missing,count=count)

@triton.jit
def _export_d2(RAW,SHIFT,DEN,IDS,GRID,OUT,MASS,SPECIAL_MASS,G:tl.constexpr,H:tl.constexpr,A:tl.constexpr,S:tl.constexpr,R:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BH:tl.constexpr,BITS:tl.constexpr):
 tile=tl.program_id(0);g=tl.program_id(1);j=tile*B+tl.arange(0,B);h=tl.arange(0,BH)
 value=tl.load(RAW+(g*H+h[:,None])*(S+A)+j[None,:],(h[:,None]<H)&(j[None,:]<S+A),255 if BITS==8 else -float('inf')).to(tl.float32)
 if BITS==8:value=tl.where(value<255.,-value/16.,-float('inf'))
 shift=tl.load(SHIFT+(g*H+h)*T+tile,h<H,0.);den=tl.load(DEN+h*G+g,h<H,0.)
 probability=tl.sum(tl.where(h[:,None]<H,tl.exp(value+shift[:,None]-den[:,None]),0.),0)/H
 key=tl.load(IDS+g*A+j-S,(j>=S)&(j<S+A),-1);valid=(j>=S)&(j<S+A)&(key>=0)
 region=tl.load(GRID+tl.maximum(key,0),valid,0)
 tl.store(OUT+g*A+j-S,tl.where(valid,probability,0.),(j>=S)&(j<S+A))
 tl.atomic_add(MASS+g*R+region,probability,valid,sem='relaxed')
 if tile*B<S:tl.atomic_add(SPECIAL_MASS+g,tl.sum(tl.where(j<S,probability,0.),0),sem='relaxed')

def finalize_d2(state,observation,start,ids,evidence,missing):
 """True per-head selected partition + pooled missing mass,THEN head mean.

 Only the within-window raw encoding has a Head axis. FP16/8-bit are explicit
 engineering formats;normalization uses unquantized selected logits. No extra
 selected QK,and no per-head data crosses to the next layer.
 """
 g,a=ids.shape;r=state.regions;s=len(state.specials);h=evidence['raw_lse'].shape[0]
 den=torch.logaddexp(evidence['raw_lse'],missing['logz']).contiguous()
 uniform=(observation['raw_grid_logits'][:,start:start+g]-den[:,:,None]).exp().mean(0).contiguous()
 selected=torch.zeros((g,r),device=ids.device);special_mass=torch.zeros(g,device=ids.device);p=torch.empty((g,a),device=ids.device)
 b=evidence['block_keys'];t=triton.cdiv(s+a,b)
 _export_d2[(t,g)](evidence['encoded'],evidence['shifts'],den,ids,state.grid,p,selected,special_mass,g,h,a,s,r,t,b,triton.next_power_of_2(h),evidence['history_bits'],num_warps=4)
 full=(selected+(state.grid_sizes-missing['count'])*uniform).contiguous()
 conditional=torch.empty((g,a),device=ids.device,dtype=torch.float16)
 _condition[(g,triton.cdiv(a,256))](ids,p,state.grid,full,conditional,a,0,r,256,num_warps=4)
 # Correct the small mass drift introduced by raw quantization before caching
 # a calibrated Dense-probability estimate. Conditional ratios are unchanged.
 normalizer=(full.sum(1)+special_mass).clamp_min(EPS)
 return dict(conditional=conditional,uniform=(uniform/full.clamp_min(EPS)).contiguous(),coarse=observation['coarse'][start:start+g].clone(),estimated_grid_probability=full/normalizer[:,None])
