"""GPU correctness checks: head permutation, explicit formula, fused grid stats."""
import sys,json,copy,time,statistics,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FORK=Path(os.environ.get('HEADMEAN_SOURCE','/tmp/zdrop_headmean_singleqk_20261102'));sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import torch
from types import SimpleNamespace as NS
from saf3r.triton.reuse_single_qk_grid import weighted_attention
from saf3r.patch import reuse_headmean_history as ready
from saf3r.triton.reuse_ready_history import decode
OUT=ROOT/'outputs'/os.environ.get('HEADMEAN_OUTPUT','headmean_single_qk_validation_20261102');OUT.mkdir(exist_ok=True)
def diff(a,b):
 d=a.float()-b.float();return dict(max_abs=float(d.abs().max()),rel_l2=float(d.norm()/a.float().norm().clamp_min(1e-30)))
def case(seed,dtype,storage):
 torch.manual_seed(seed);g,h,n,p,r,a,d=3,3,79,67,7,41,32
 grid=torch.arange(p,device='cuda')%r
 s=NS(n=n,groups=g,budget=a,regions=r,grid=grid,grid_sizes=torch.bincount(grid,minlength=r).float(),patches=torch.arange(p,device='cuda'),specials=torch.arange(p,n,device='cuda'),fine=NS(base_count=13,labels=torch.arange(p,device='cuda')%13),m5c_count_errors=[],m5c_seed_errors=[],m5c_prior_errors=[],previous_layer=-1,ready_written={},ready_cache_dtype=storage,config=NS(window_groups=64))
 q=torch.randn(h,n,d,device='cuda',dtype=dtype);k=torch.randn_like(q);v=torch.randn_like(q)
 s.history_pooled_fp32=torch.randn(h,r,d,device='cuda')*.2
 ids=torch.stack([torch.randperm(p,device='cuda')[:a] for _ in range(g)]).int();ids[0,-3:]=-1;ids[1].fill_(-1)
 bias=torch.rand(g,a,device='cuda')*2
 obs=ready.observe(s,q,k,0,g,d**-.5)
 o,summary,raw=weighted_attention(q,k,v,ids,bias,s.specials,d**-.5,grid[ids.long().clamp_min(0)],r)
 oo=torch.empty_like(q);ss=torch.zeros_like(summary);rr=torch.full_like(raw,-torch.inf)
 for group in range(g):
  first=group*32;end=min(first+32,n);valid=ids[group]>=0;part=ids[group,valid].long();allids=torch.cat((s.specials,part));lw=torch.cat((torch.zeros(len(s.specials),device='cuda'),bias[group,valid]))
  logits=q[:,first:end].float()@k[:,allids].float().transpose(-1,-2)*d**-.5+lw
  prob=logits.softmax(-1);oo[:,first:end]=(prob@v[:,allids].float()).to(dtype)
  pm=prob[:,:,len(s.specials):].mean((0,1))
  ss[group,0].scatter_add_(0,grid[part],pm)
  ss[group,1].scatter_add_(0,grid[part],pm/bias[group,valid].exp())
  positions=torch.minimum(torch.tensor([4,12,20,28],device='cuda'),torch.tensor(end-first-1,device='cuda'))+first
  rr[group,:,:,valid]=q[:,positions].float()@k[:,part].float().transpose(-1,-2)*d**-.5
 od=diff(oo,o);sd=diff(ss,summary);rd=diff(rr.masked_fill(~torch.isfinite(rr),0),raw.masked_fill(~torch.isfinite(raw),0))
 assert od['rel_l2']<(1e-5 if dtype==torch.float32 else .005),od
 assert sd['rel_l2']<1e-5,sd
 assert rd['rel_l2']<1e-5,rd
 # Explicit mean of per-head conditional probabilities, NOT mean logits or
 # ratio of mean probabilities; keep probes separate.
 valid=ids>=0;regions=grid[ids.long().clamp_min(0)];count=torch.zeros(g,r,device='cuda').scatter_add_(1,regions,valid.float());remain=s.grid_sizes-count
 missing=(obs['grid_logits']+remain.clamp_min(1).log()[:,None,None,:]).masked_fill(remain[:,None,None,:]==0,-torch.inf)
 logz=torch.logaddexp(torch.logaddexp(missing.logsumexp(-1),rr.logsumexp(-1)),obs['special_logits'].logsumexp(-1))[:,:,:,None]
 u=(obs['grid_logits']-logz).exp();prob=(rr-logz).exp()
 corrected=(u*remain[:,None,None,:]).scatter_add_(-1,regions[:,None,None,:].expand_as(prob),prob)
 cond=prob/corrected.gather(-1,regions[:,None,None,:].expand_as(prob)).clamp_min(1e-30)
 expected=cond.mean(1,keepdim=True)
 ready.ensure_cache(s,q);obs['ready_logits']=raw;ready.save_evidence(s,q,k,0,ids,ids,obs,d**-.5)
 cached=decode(s.ready_edges,a);cd=diff(expected,cached)
 assert cached.shape==(g,1,4,a) and s.ready_host is None
 assert cd['rel_l2']<(1e-5 if storage==torch.float32 else .001),cd
 assert torch.allclose(s.ready_uniform,(u/corrected.clamp_min(1e-30)).mean(1,keepdim=True),atol=1e-6,rtol=1e-5)
 # Independently permuting current-layer heads leaves the transferred prior
 # unchanged, because current grids are averaged before confidence.
 s.previous_layer=0
 current=ready.observe(s,q*.9+.05,k,0,g,d**-.5)
 pa,ca=ready.build_prior(s,0,ids,current)
 perm=torch.tensor([2,0,1],device='cuda');other={**current,'coarse_grid_probability':current['coarse_grid_probability'][:,perm]}
 pb,cb=ready.build_prior(s,0,ids,other)
 assert torch.allclose(pa,pb,atol=1e-7,rtol=1e-5)
 assert torch.allclose(ca,cb,atol=1e-7,rtol=1e-5)
 # Explicit head-mean migration reference.
 cur=current['coarse_grid_probability'].mean(1,keepdim=True);prev=obs['coarse_grid_probability'].mean(1,keepdim=True)
 ratio=(cur+1e-12)/(prev+1e-12)
 from saf3r.patch.history_dense_probability import EPS
 ratio=(cur+EPS)/(prev+EPS);conf=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
 uni=(u/corrected.clamp_min(EPS)).mean(1,keepdim=True)
 base=cur*(conf*uni+(1-conf)/s.grid_sizes)
 target=(base*current['weights'][:,None,:,None]).sum(2).mean(1)[:,grid].clone()
 ix=regions[:,None,None,:].expand(g,1,4,a)
 transferred=cur.gather(-1,ix)*(conf.gather(-1,ix)*cached+(1-conf.gather(-1,ix))/s.grid_sizes[regions][:,None,None,:])
 values=(transferred*current['weights'][:,None,:,None]).sum(2).mean(1)
 for i in range(g):target[i,ids[i,valid[i]].long()]=values[i,valid[i]]
 pd=diff(target,pa);assert pd['rel_l2']<1e-5,pd
 return dict(dtype=str(dtype),storage=str(storage),output=od,grid=sd,probe_logits=rd,conditional=cd,prior=pd,current_head_permutation_invariant=True,cache_shape=list(cached.shape))
def benchmark():
 from saf3r.triton.reuse_explore_av_ready import weighted_attention as previous
 torch.manual_seed(100);g,h,r,n,a,d,regions=8,16,256,32768,6500,64,128
 q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
 ids=torch.stack([torch.randperm(n-64,device='cuda')[:a].sort().values for _ in range(g)]).int();bias=torch.rand(g,a,device='cuda');special=torch.arange(n-64,n,device='cuda');grid=(ids//256).clamp_max(regions-1)
 res={}
 for name,fn in [('old_two_qk',lambda:previous(q,k,v,ids,bias,special,d**-.5)),('new_one_qk',lambda:weighted_attention(q,k,v,ids,bias,special,d**-.5,grid,regions))]:
  samples=[]
  for i in range(4):
   torch.cuda.synchronize();begin=time.perf_counter();x=fn();torch.cuda.synchronize()
   if i:samples.append(time.perf_counter()-begin)
   del x
  res[name]=dict(samples=samples,median_seconds=statistics.median(samples))
 return res
if __name__=='__main__':
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 results=[case(0,t,s) for t in [torch.float32,torch.bfloat16] for s in [torch.float32,torch.float16]]
 result=dict(complete=True,checks=results,microbenchmark=benchmark(),quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},reason='Numerical/kernel diagnostic; no GT quality evaluation',dense_speedup=None,dense_reason='No unchanged control rerun')
 (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
