"""Required width growth must expand/capture, never truncate cluster cells."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from scripts.validate_shared_value_loss import example
from scripts.quality_metrics import quality_record
from saf3r.triton import reuse_shared_value_states as shared
from saf3r.triton.reuse_drop_moment import prepare
from saf3r.patch.reuse_window_graph import select,eager
out=Path('outputs/query256_efficiency_20261105/validation/graph_growth.json')
result={'quality':quality_record({},'Synthetic structural graph growth diagnostic; no model/pose/recon_unposed evaluation'),'passed':False}
try:
 z,labels,grid,reps,pop,u,prior,old,pi,*_=example('cuda')
 state=SimpleNamespace(config=SimpleNamespace(reuse_window_graphs=True,static_shared_graph=True,graph_cache_capacity=2,graph_cache_max_mib=512,dual_iterations=16,drop_multiplier=1.,p0_loss_pack=False),budget=24,_forward_serial=1,graph_cache_hits=0,graph_cache_misses=0,graph_cache_evictions=0,graph_constant_updates=0)
 allow=torch.ones_like(pi,dtype=torch.bool)
 widths=[]
 for layer in range(2):
  if layer:
   labels=torch.cat((torch.arange(len(reps),device='cuda'),torch.zeros(len(z)-len(reps),device='cuda',dtype=torch.int64)))
   pop=torch.bincount(labels,minlength=len(reps)).float()
   reps=torch.stack([torch.nonzero(labels==c)[0,0] for c in range(len(reps))])
   pi=torch.zeros_like(pi).scatter_add_(1,labels[None].expand(len(prior),-1),prior)
  def geometry(static):
   g=shared.prepare_geometry(z,labels,reps,pop,min_width=64 if static else 1)
   g['drop_bins']=[(g['width'],torch.arange(len(reps),device='cuda'))] if static else prepare(pop)
   g['fuse_state_kernel']=True;shared.prepare(z,labels,reps,grid,u.shape[1],g)
   return g
  g=geometry(True);widths.append(g['width'])
  expected=eager(prior,pi,old,allow,z,reps,labels,pop,geometry(False),24,16,1.,uniform=u)
  actual=select(state,layer,prior,pi,old,allow,z,reps,labels,pop,g,uniform=u)
  for a,b in zip(actual[:3],expected[:3]):torch.testing.assert_close(a,b,atol=0,rtol=0)
  for k in expected[3]:torch.testing.assert_close(actual[3][k],expected[3][k],atol=1e-4,rtol=1e-4)
 assert widths==[64,128] and int(pop.max())==105
 assert state._window_graph_layer==1 and len(state._window_graphs)==1
 result.update(passed=True,widths=widths,max_population=int(pop.max()),exact_supports=True,fallback_graph_created=True,no_truncation=True)
finally:out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
