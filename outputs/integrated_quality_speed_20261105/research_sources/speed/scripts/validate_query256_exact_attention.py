"""Bit equality against the literal frozen uncommitted baseline, synthetic inputs."""
import importlib,json,sys,types
from pathlib import Path
import torch
from scripts.quality_metrics import quality_record
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/query256_equivalent_20261105';OUT.mkdir(exist_ok=True)
frozen=ROOT/'outputs/query256_efficiency_20261105/frozen_baseline/saf3r'
for name,path in [('frozen_reference',frozen),('frozen_reference.triton',frozen/'triton')]:
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
reference=importlib.import_module('frozen_reference.triton.reuse_direct_sampled_probability').direct_sparse_attention_sampled
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as candidate
torch.manual_seed(763);torch.set_num_threads(2)
rows=[]
for h,r,n,d,a in [(4,519,401,64,73),(16,2048,8192,64,1638)]:
 g=(r+255)//256;q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
 ids=torch.stack([torch.randperm(n-5,device='cuda')[:a]+5 for _ in range(g)]).int();ids[-1,-4:]=-1;sp=torch.arange(5,device='cuda')
 bias=torch.randn(g,a,device='cuda').masked_fill(ids<0,-torch.inf)
 kw=dict(logit_bias=bias,query_group_size=256,probe_group_stride=1,debiased_log=True)
 out,ev=reference(q,k,v,ids,sp,d**-.5,**kw)
 for bm,stages,split,execution_split in [(32,2,True,True),(32,1,True,True),(32,2,False,False)]:
  x,e=candidate(q,k,v,ids,sp,d**-.5,block_queries=bm,num_stages=stages,split_export=split,execution_split=execution_split,**kw)
  row=dict(shape=[h,r,n,d,a],block_queries=bm,stages=stages,split=split,execution_split=execution_split)
  for name,left,right in [('out',out,x)]+[(s,ev[s],e[s]) for s in ('lse','debiased_log_special_probability','debiased_log_patch_probability')]:
   valid=torch.isfinite(left)&torch.isfinite(right);row[name]=dict(equal=torch.equal(left,right),max_abs=float((left[valid].float()-right[valid].float()).abs().max()))
  rows.append(row);print(row,flush=True)
(OUT/'split_attention.json').write_text(json.dumps(dict(quality=quality_record({},'Fixed synthetic operator diagnostic; no pose/recon_unposed or paired Dense'),rows=rows),indent=2)+'\n')
