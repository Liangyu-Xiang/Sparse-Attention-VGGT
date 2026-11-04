"""Synthetic fixed-state bit equality and complete states-wrapper timings."""
import importlib,json,statistics,sys,time,types
from pathlib import Path
import torch,triton
from saf3r.triton import reuse_shared_value_states as current
from saf3r.triton.reuse_drop_moment import prepare
from scripts.quality_metrics import quality_record
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/query256_deterministic_20261105'
pkg=types.ModuleType('capacity_reference');pkg.__path__=[str(OUT/'frozen_source/saf3r/triton')];sys.modules['capacity_reference']=pkg
previous=importlib.import_module('capacity_reference.reuse_shared_value_states')
torch.manual_seed(78);g,f,r,w=64,26000,512,64
pop=torch.randint(1,15,(f,),device='cuda').float();bins=dict(prepare(pop));n=int(pop.sum());u=torch.rand(g,r,device='cuda');allow=torch.rand(g,f,device='cuda')>.5
cells=torch.randint(0,r,(f,w),device='cuda',dtype=torch.int32);cells[:,3:]=-1
drop=torch.rand(f,w,device='cuda');rep=torch.rand_like(drop)
geo=dict(width=w,shared_cells=cells,shared_drop=drop,shared_rep=rep,shared_regions=r,drop_bins=[],drop_bin_counts=[],bounded_bins=True)
for size in (1,2,4,8,16,32,64):
 ids=bins.get(size,torch.empty(0,device='cuda',dtype=torch.int32));minimum=1 if size==1 else size//2+1;cap=triton.next_power_of_2(max(1,min(f,n//minimum)))
 buf=torch.empty(cap,device='cuda',dtype=torch.int32);buf[:len(ids)].copy_(ids);geo['drop_bins'].append((size,buf));geo['drop_bin_counts'].append(torch.tensor([len(ids)],device='cuda',dtype=torch.int32))
context=dict(remove_rep=torch.rand(g,f,device='cuda')>.5,eligible_count=torch.randint(0,4,(g,f),device='cuda'))
residual=tuple(torch.rand(g,f,device='cuda') for _ in range(3));a=previous.states(u,allow,pop,geo,context,residual);objective=geo['objective'].clone();b=current.states(u,allow,pop,geo,context,residual)
for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
torch.testing.assert_close(objective,geo['objective'],atol=0,rtol=0)
rows=[]
for name,module in [('fixed_capacity',previous),('persistent',current)]:
 module.states(u,allow,pop,geo,context,residual);torch.cuda.synchronize();samples=[]
 for _ in range(5):
  begin=time.perf_counter();module.states(u,allow,pop,geo,context,residual);torch.cuda.synchronize();samples.append(time.perf_counter()-begin)
 rows.append(dict(method=name,seconds=samples,median_seconds=statistics.median(samples)))
result=dict(quality=quality_record({},'Synthetic fixed-state kernel diagnostic; no model/pose/recon_unposed or paired Dense'),shape=dict(groups=g,clusters=f,regions=r,width=w),bit_equal=True,rows=rows)
(OUT/'persistent_states.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
