"""Tail checks for the launch parameters chosen by the resource-only screen."""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as attention
from scripts.quality_metrics import quality_record,quality_markdown
OUT=ROOT/'outputs/source32_query_tiles_20261105/resource_tails';OUT.mkdir(exist_ok=False)
qrecord=quality_record({},'Explicit chosen-kernel tail diagnostic; no pose/recon_unposed evaluation')
result=dict(passed=False,quality=qrecord,cases=[],dense_speedup=None,dense_speedup_reason='Kernel diagnostic, no paired full forward')
(OUT/'quality.json').write_text(json.dumps(qrecord,indent=2)+'\n');(OUT/'quality.md').write_text(quality_markdown(qrecord))
winners=json.loads((OUT.parent/'resource_probe/result.json').read_text())['winners'];torch.set_num_threads(2);torch.manual_seed(20261105)
try:
 with torch.inference_mode():
  for r in (1,256,519,1031):
   h,n,d,a=4,401,64,73;g=math.ceil(r/256);q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
   ids=torch.stack([torch.randperm(n-5,device='cuda')[:a]+5 for _ in range(g)]).sort(1).values.int();ids[-1,-4:]=-1
   sp=torch.arange(5,device='cuda');bias=torch.zeros(g,a,device='cuda').masked_fill(ids<0,-torch.inf);pos=(torch.arange(g,device='cuda')[:,None]*256+32).clamp_max(r-1).int()
   kw=dict(logit_bias=bias,probe_positions=pos,probe_count=1,probe_group_stride=1,query_group_size=256,debiased_log=True,split_export=False)
   ref,e0=attention(q,k,v,ids,sp,d**-.5,block_queries=32,num_warps=4,num_stages=2,**kw)
   for bm in (64,128):
    w=winners[str(bm)];out,e=attention(q,k,v,ids,sp,d**-.5,block_queries=bm,num_warps=w['num_warps'],num_stages=w['num_stages'],**kw)
    for name,x,y,atol,rtol in [('output',out,ref,.016,.02),('lse',e['lse'],e0['lse'],1e-4,1e-5)]+[(name,e[name],e0[name],.025,.006) for name in ('debiased_log_patch_probability','debiased_log_special_probability')]:
     assert torch.equal(torch.isfinite(x),torch.isfinite(y));torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
    result['cases'].append(dict(rows=r,block_queries=bm,warps=w['num_warps'],stages=w['num_stages'],passed=True))
 result['passed']=True
finally:(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('Resource-tail cases passed:',len(result['cases']))
