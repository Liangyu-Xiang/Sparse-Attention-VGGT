"""Exact state arithmetic and runtime-count graph tests, no test truth."""
import importlib,json,sys,types
from pathlib import Path
from types import SimpleNamespace
import torch,triton
from scripts.quality_metrics import quality_record
from scripts.validate_shared_value_loss import example
from saf3r.triton import reuse_shared_value_states as shared
from saf3r.triton.reuse_drop_moment import prepare
from saf3r.patch.reuse_window_graph import select,eager
from saf3r.patch.reuse_inference_ops import gate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/query256_equivalent_20261105'
frozen=ROOT/'outputs/query256_efficiency_20261105/frozen_baseline/saf3r'
for name,path in [('frozen_reference',frozen),('frozen_reference.triton',frozen/'triton')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
reference=importlib.import_module('frozen_reference.triton.reuse_shared_value_states')
result=dict(quality=quality_record({},'Fixed state arithmetic / graph diagnostic; no model pose/recon_unposed or paired Dense'),layers=[],passed=False)
try:
 z,labels,grid,reps,pop,u,prior,old,pi,*_=example('cuda');allow=torch.ones_like(pi,dtype=torch.bool)
 state=SimpleNamespace(config=SimpleNamespace(reuse_window_graphs=True,static_shared_graph=False,stable_binned_graph=True,graph_cache_capacity=2,graph_cache_max_mib=512,dual_iterations=16,drop_multiplier=1.,p0_loss_pack=False),budget=24,_forward_serial=1,graph_cache_hits=0,graph_cache_misses=0,graph_cache_evictions=0,graph_constant_updates=0)
 for layer in range(6):
  f=len(reps);labels=torch.cat((torch.arange(f,device='cuda'),(torch.arange(len(z)-f,device='cuda')//(layer+1))%f));pop=torch.bincount(labels,minlength=f).float();reps=torch.stack([torch.nonzero(labels==c)[0,0] for c in range(f)])
  prior=prior.roll(1,1).contiguous();u=u.roll(1,1).contiguous();old=old.roll(1,1).contiguous();pi=torch.zeros_like(pi).scatter_add_(1,labels[None].expand(len(prior),-1),prior)
  def geometry(stable):
   g=shared.prepare_geometry(z,labels,reps,pop,min_width=64 if stable else 1);g['fuse_state_kernel']=True;g['drop_bins']=prepare(pop);shared.prepare(z,labels,reps,grid,u.shape[1],g)
   if stable:
    bins=dict(g['drop_bins']);g['drop_bins']=[(size,bins.get(size,torch.empty(0,device='cuda',dtype=torch.int32))) for size in (1,2,4,8,16,32,64,128,256,512) if size<=g['width']];g['bounded_bins']=True
   return g
  baseline=geometry(False);candidate=geometry(True)
  context=gate(prior,pi,pop,old,labels,reps)
  residual=reference.corrections(prior,old,u,labels,baseline,context)
  expected=reference.states(u,allow,pop,baseline,context,residual)
  # Fixed-capacity counts are runtime inputs, including zero-size bins.
  candidate['drop_bin_counts']=[];capacity=[]
  for size,ids in candidate['drop_bins']:
   buf=torch.empty(triton.next_power_of_2(f),device='cuda',dtype=ids.dtype);buf[:len(ids)].copy_(ids);capacity.append((size,buf));candidate['drop_bin_counts'].append(torch.tensor([len(ids)],device='cuda',dtype=torch.int32))
  candidate['drop_bins']=capacity
  actual=shared.states(u,allow,pop,candidate,context,residual)
  for x,y in zip(actual,expected):torch.testing.assert_close(x,y,atol=0,rtol=0)
  torch.testing.assert_close(candidate['objective'],baseline['objective'],atol=0,rtol=0)
  expected=eager(prior,pi,old,allow,z,reps,labels,pop,geometry(False),24,16,1.,uniform=u)
  actual=select(state,layer,prior,pi,old,allow,z,reps,labels,pop,geometry(True),uniform=u)
  for x,y in zip(actual[:3],expected[:3]):torch.testing.assert_close(x,y,atol=0,rtol=0)
  result['layers'].append(dict(layer=layer,max_population=int(pop.max()),original_bins=[(size,len(ids)) for size,ids in baseline['drop_bins']],state_errors_costs_objective_bit_equal=True,selected_supports_bit_equal=True))
 assert state.graph_cache_creations==1 and state.graph_cache_hits==5
 # Real width growth allocates a new graph; subsequent data updates reuse it.
 f=len(reps);labels=torch.cat((torch.arange(f,device='cuda'),torch.zeros(len(z)-f,device='cuda',dtype=torch.int64)));pop=torch.bincount(labels,minlength=f).float();reps=torch.stack([torch.nonzero(labels==c)[0,0] for c in range(f)]);pi=torch.zeros_like(pi).scatter_add_(1,labels[None].expand(len(prior),-1),prior)
 for layer in (6,7):
  expected=eager(prior,pi,old,allow,z,reps,labels,pop,geometry(False),24,16,1.,uniform=u)
  actual=select(state,layer,prior,pi,old,allow,z,reps,labels,pop,geometry(True),uniform=u)
  for x,y in zip(actual[:3],expected[:3]):torch.testing.assert_close(x,y,atol=0,rtol=0)
 assert state.graph_cache_creations==2 and state.graph_cache_hits==6
 result.update(width_growth=128,max_population=105,width_growth_reused=True)
 result.update(passed=True,hits=state.graph_cache_hits,creations=state.graph_cache_creations,evictions=state.graph_cache_evictions,live_bytes=state.graph_cache_bytes)
finally:(OUT/'exact_graph.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
