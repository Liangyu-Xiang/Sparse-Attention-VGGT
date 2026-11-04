"""No-GT oracle for compact sampled history and own-Grid fallback."""
import json,sys,math
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.patch import reuse_sampled_history as history
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled
p=Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=False);torch.manual_seed(20261104);torch.backends.cuda.matmul.allow_tf32=False
checks=[]
for interval in [32,256]:
 h,n,d,groups,cap,regions=3,73,64,17,48,7;rows=groups*32-5;grid=torch.arange(n,device='cuda')%regions;sizes=torch.bincount(grid,minlength=regions).float();labels=torch.arange(n,device='cuda')%9
 q=torch.randn(h,rows,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n+5,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);ids=torch.stack([torch.randperm(n,device='cuda')[:cap] for _ in range(groups)]).int();ids[:,7]=-1;special=torch.arange(n,n+5,device='cuda');bias=torch.rand(groups,cap,device='cuda')*30;bias[:,3]=-torch.inf
 state=SimpleNamespace(config=SimpleNamespace(history_query_interval=interval),groups=groups,budget=cap,regions=regions,grid=grid,grid_sizes=sizes,fine=SimpleNamespace(base_count=9,labels=labels),ready_written={},previous_layer=-1,m5c_prior_errors=[],m5c_seed_errors=[],m5c_count_errors=[])
 history.ensure_cache(state,q);step=interval//32
 positions=(torch.arange(groups,device='cuda')[:,None]*32+torch.tensor([4,12,20,28],device='cuda')).clamp_max(rows-1)
 pool=torch.zeros((h,regions,d),device='cuda').index_add_(1,grid,k[:,:n].float())/sizes[None,:,None]
 def observation(start,g):
  pos=positions[start:start+g];pq=q[:,pos].permute(1,0,2,3).float();z=torch.einsum('ghpd,hrd->ghpr',pq,pool)*.125;sp=torch.einsum('ghpd,hsd->ghps',pq,k[:,special].float())*.125;den=torch.logaddexp((z+sizes.log()).logsumexp(-1),sp.logsumexp(-1))[:,:,:,None]
  return dict(positions=pos,grid_logits=z,special_logits=sp,coarse_grid_probability=(z-den).exp()*sizes,patch_probability=(z+sizes.log()).softmax(-1).mean((1,2)))
 for start,g in [(0,8),(8,9)]:
  first=start*32;last=min(rows,(start+g)*32);obs=observation(start,g)
  _,e=direct_sparse_attention_sampled(q[:,first:last],k,v,ids[start:start+g],special,.125,bias[start:start+g],(obs['positions'][:,:1]-first).int().contiguous(),probe_group_stride=step,debiased_log=True)
  history.observe_execution(state,q,k,start,ids[start:start+g],ids[start:start+g],bias[start:start+g],obs,.125,e)
  history.save_evidence(state,q,k,start,ids[start:start+g],ids[start:start+g],obs,.125)
 state.previous_layer=0;errors=[];mass_errors=[]
 for start,g in [(0,8),(8,9)]:
  obs=observation(start,g);obs['coarse_grid_probability'].mul_(.9)
  prior,pi=history.build_prior(state,start,ids[start:start+g],obs)
  current=obs['coarse_grid_probability'][:,:,0].mean(1);reference=(current/sizes)[:,grid].clone();begin=start//step
  for local in range(0,g,step):
   cache_index=begin+local//step;cur=current[local];previous=state.ready_coarse[cache_index];ratio=(cur+1e-30)/(previous+1e-30);beta=torch.where(ratio<1,ratio,ratio.pow(-2))
   reference[local]=(cur*(beta*state.ready_uniform[cache_index]+(1-beta)/sizes))[grid]
   row=ids[start+local];valid=row>=0;reg=grid[row[valid].long()];cond=state.ready_edges[cache_index,0,0,valid].float().exp()
   reference[local,row[valid].long()]=cur[reg]*(beta[reg]*cond+(1-beta[reg])/sizes[reg])
  error=float((prior-reference).abs().max());assert torch.allclose(prior,reference,atol=2e-7,rtol=2e-6),error
  unsampled=torch.arange(g,device='cuda')%step!=0
  if unsampled.any():assert torch.equal(prior[unsampled],(current/sizes)[:,grid][unsampled])
  mass_error=float((prior.sum(-1)-current.sum(-1)).abs().max());assert mass_error<.003,mass_error
  assert torch.allclose(pi.sum(-1),prior.sum(-1),atol=2e-6)
  errors.append(error);mass_errors.append(mass_error)
 # Fresh current coarse probabilities must reach the captured consumer.
 obs=observation(0,8);history.build_prior(state,0,ids[:8],obs);graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):gp,gpi=history.build_prior(state,0,ids[:8],obs)
 obs['coarse_grid_probability'].mul_(.83);graph.replay();ep,epi=history.build_prior(state,0,ids[:8],obs)
 graph_prior_error=float((gp-ep).abs().max());graph_cluster_error=float((gpi-epi).abs().max());print('GRAPH_ERRORS',interval,graph_prior_error,graph_cluster_error,flush=True)
 assert torch.equal(gp,ep) and torch.allclose(gpi,epi,atol=2e-6,rtol=2e-6)
 checks.append(dict(interval=interval,oracle_max_abs=max(errors),prior_mass_max_error=max(mass_errors),cache_groups=len(state.ready_edges),expected_cache_groups=math.ceil(groups/step),seed_normalization_max_error=float(torch.stack(state.m5c_seed_errors).max()),fresh_graph=True,graph_prior_error=graph_prior_error,graph_cluster_atomic_rounding_error=graph_cluster_error,own_grid_fallback_verified=True))
 assert len(state.ready_edges)==math.ceil(groups/step)
result=dict(complete=True,checks=checks,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Numerical integration/cache/fallback diagnostic;no geometry evaluation',dense_speedup=None,dense_reason='No paired full-forward timing in numerical diagnostic');(p/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
from scripts.quality_metrics import quality_record,quality_markdown
(p/'quality.md').write_text(quality_markdown(quality_record({},result['quality_reason'])))
