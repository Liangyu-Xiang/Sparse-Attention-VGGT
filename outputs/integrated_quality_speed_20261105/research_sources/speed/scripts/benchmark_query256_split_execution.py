"""Fixed Q/K/V/supports: full wrapper cost and exact outputs, no test truth."""
import json,statistics,time
from pathlib import Path
import torch
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as attention
from scripts.quality_metrics import quality_record
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/query256_deterministic_20261105'
torch.manual_seed(763);torch.set_num_threads(2)
h,n,r,d,a=16,131072,16384,64,26000;g=(r+255)//256
q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
ids=torch.stack([torch.randperm(n-640,device='cuda')[:a]+640 for _ in range(g)]).int();sp=torch.arange(640,device='cuda')
kw=dict(query_group_size=256,probe_group_stride=1,debiased_log=True,split_export=True)
ref,ev=attention(q,k,v,ids,sp,d**-.5,**kw);rows=[]
for split,stages in [(False,2),(True,2),(True,1)]:
 fn=lambda:attention(q,k,v,ids,sp,d**-.5,num_stages=stages,execution_split=split,**kw)
 x,e=fn();torch.cuda.synchronize();torch.testing.assert_close(x,ref,atol=0,rtol=0)
 for name in ('lse','debiased_log_special_probability','debiased_log_patch_probability'):torch.testing.assert_close(e[name],ev[name],atol=0,rtol=0)
 samples=[]
 for _ in range(5):
  begin=time.perf_counter();x,e=fn();torch.cuda.synchronize();samples.append(time.perf_counter()-begin)
 rows.append(dict(execution_split=split,stages=stages,bit_equal=True,seconds=samples,median_seconds=statistics.median(samples),resources=e['compiler_resources']))
result=dict(quality=quality_record({},'Fixed-tensor full attention-wrapper diagnostic; no model pose/recon_unposed or paired Dense'),rows=rows)
(OUT/'split_execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
