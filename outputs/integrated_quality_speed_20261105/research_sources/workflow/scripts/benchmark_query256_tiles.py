"""Fixed synthetic supports, full attention wrapper timings including exports."""
import json,statistics,time
from pathlib import Path
import torch
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as attention
torch.manual_seed(763);torch.set_num_threads(2)
h,n,r,d,a=16,131072,16384,64,26000;g=(r+255)//256
q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
ids=torch.stack([torch.randperm(n-640,device='cuda')[:a]+640 for _ in range(g)]).int();sp=torch.arange(640,device='cuda')
rows=[]
for bm in (32,64,128):
 for bk in (64,128):
  for warps in (4,8):
   for stages in (1,2):
    f=lambda:attention(q,k,v,ids,sp,d**-.5,block_queries=bm,block_keys=bk,num_warps=warps,num_stages=stages,query_group_size=256,probe_group_stride=1,debiased_log=True,split_export=True)
    try:
     o,e=f();torch.cuda.synchronize();samples=[]
     for _ in range(3):
      begin=time.perf_counter();o,e=f();torch.cuda.synchronize();samples.append(time.perf_counter()-begin)
     row=dict(bm=bm,bk=bk,warps=warps,stages=stages,median=statistics.median(samples),samples=samples,resources=e['compiler_resources'])
    except Exception as exc:row=dict(bm=bm,bk=bk,warps=warps,stages=stages,error=str(exc))
    rows.append(row);print(row,flush=True)
    Path('outputs/query256_efficiency_20261105/validation/tiles.json').write_text(json.dumps(dict(scope='synthetic diagnostic; not quality evaluation',quality={k:None for k in ('auc03','auc30','acc','comp','cd','f1')},quality_unavailable='Synthetic operator benchmark',speedup_vs_dense=None,speedup_unavailable='No paired Dense',shape=dict(h=h,n=n,r=r,d=d,a=a),rows=rows),indent=2)+'\n')
