"""Synthetic operator and graph checks; never uses evaluation ground truth."""
import json,math,time
from pathlib import Path
from types import SimpleNamespace
import torch,triton
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as attention
from saf3r.triton import reuse_shared_value_states as shared
from saf3r.triton.reuse_drop_moment import prepare
from saf3r.patch.reuse_window_graph import select,eager
from scripts.validate_shared_value_loss import example
OUT=Path('outputs/query256_efficiency_20261105/validation')
OUT.mkdir(exist_ok=True,parents=True)
results={'scope':'synthetic operator diagnostic; not a complete quality evaluation','quality':{k:None for k in ('auc03','auc30','acc','comp','cd','f1')},'quality_unavailable':'Synthetic fixed tensors; pose/recon_unposed not applicable','speedup_vs_dense':None,'speedup_unavailable':'No paired Dense timing','cases':[]}
try:
 torch.manual_seed(763)
 for r in (256,519,1031):
  h,n,d=4,401,64;g=math.ceil(r/256);a=73
  q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
  ids=torch.stack([torch.randperm(n-5,device='cuda')[:a]+5 for _ in range(g)]).int();ids[-1,-4:]=-1
  specials=torch.arange(5,device='cuda');bias=torch.randn(g,a,device='cuda').masked_fill(ids<0,-torch.inf)
  positions=(torch.arange(g,device='cuda')[:,None]*256+32).clamp_max(r-1).int()
  kwargs=dict(logit_bias=bias,probe_positions=positions,probe_group_stride=1,query_group_size=256,debiased_log=True)
  ref,ev=attention(q,k,v,ids,specials,d**-.5,**kwargs)
  for bm in (64,128):
   out,e=attention(q,k,v,ids,specials,d**-.5,block_queries=bm,num_stages=1,split_export=True,**kwargs)
   row=dict(rows=r,block_queries=bm)
   for name,x,y,atol,rtol in [('output',out,ref,.016,.02),('lse',e['lse'],ev['lse'],1e-4,1e-5)]+[(s,e[s],ev[s],.025,.006) for s in ('debiased_log_patch_probability','debiased_log_special_probability')]:
    valid=torch.isfinite(x)&torch.isfinite(y)
    row[name+'_max_abs']=float((x[valid].float()-y[valid].float()).abs().max())
    assert torch.equal(torch.isfinite(x),torch.isfinite(y)),name
    torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
   assert e['debiased_log_patch_probability'].is_contiguous()
   results['cases'].append(row)
 # Population/labels/reps/geometry and all per-window values change before replay.
 z,labels,grid,reps,pop,uniform,prior,old,pi,*_=example('cuda')
 def geom(static):
  m=shared.prepare_geometry(z,labels,reps,pop,min_width=64 if static else 1)
  m['drop_bins']=[(m['width'],torch.arange(len(pop),device='cuda'))] if static else prepare(pop)
  m['fuse_state_kernel']=True;shared.prepare(z,labels,reps,grid,uniform.shape[1],m)
  return m
 state=SimpleNamespace(config=SimpleNamespace(reuse_window_graphs=True,static_shared_graph=True,graph_cache_capacity=2,graph_cache_max_mib=512,dual_iterations=16,drop_multiplier=1.,p0_loss_pack=False),budget=24,_forward_serial=1,graph_cache_hits=0,graph_cache_misses=0,graph_cache_evictions=0,graph_constant_updates=0)
 allowed=torch.ones_like(pi,dtype=torch.bool)
 for layer in range(5):
  prior=prior.roll(1,1).contiguous();uniform=uniform.roll(1,1).contiguous();z=z.roll(1,0).contiguous()
  if layer%2:labels=labels.roll(1).contiguous();reps=reps.roll(1).contiguous()
  old=old.roll(1,1).contiguous();allowed=~allowed if layer else allowed
  expected=eager(prior,pi,old,allowed,z,reps,labels,pop,geom(False),24,16,1.,uniform=uniform)
  actual=select(state,layer,prior,pi,old,allowed,z,reps,labels,pop,geom(True),uniform=uniform)
  for x,y in zip(actual[:3],expected[:3]):torch.testing.assert_close(x,y,atol=0,rtol=0)
  for row in actual[0]:
   valid=row[row>=0];assert len(valid.unique())==len(valid),'duplicate support key'
 assert state.graph_cache_misses==1 and state.graph_cache_hits==4
 results['graph']=dict(hits=state.graph_cache_hits,creations=state.graph_cache_creations,updates=state.graph_constant_updates,evictions=state.graph_cache_evictions,cache_bytes=state.graph_cache_bytes,changed_inputs_replays=5,exact_supports=True)
 # Timings cover allocation, indexed AV and evidence export, on synthetic data.
 h,n,r,d,a=16,32768,8192,64,6553;g=math.ceil(r/256)
 q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
 ids=torch.stack([torch.randperm(n-5,device='cuda')[:a]+5 for _ in range(g)]).int();sp=torch.arange(5,device='cuda')
 kw=dict(probe_group_stride=1,query_group_size=256,debiased_log=True)
 timing=[]
 for bm in (32,64,128):
  fn=lambda:attention(q,k,v,ids,sp,d**-.5,block_queries=bm,split_export=bm!=32,**kw)
  fn();torch.cuda.synchronize();samples=[]
  for _ in range(5):
   begin=time.perf_counter();o,e=fn();torch.cuda.synchronize();samples.append(time.perf_counter()-begin)
  timing.append(dict(block_queries=bm,seconds=samples,resources=e['compiler_resources'],scratch_bytes=e['scratch_bytes']))
 results['synthetic_timings']=timing;results['passed']=True
finally:
 (OUT/'operators.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
