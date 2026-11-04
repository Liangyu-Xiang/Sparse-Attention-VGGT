"""Bounded kernel-only launch-parameter screen after observed large-tile spills."""
import json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as attention
from scripts.quality_metrics import quality_record,quality_markdown
OUT=ROOT/'outputs/source32_query_tiles_20261105/resource_probe';OUT.mkdir(exist_ok=False,parents=True)
quality=quality_record({},'Explicit fixed-input kernel resource diagnostic; no pose/recon_unposed inference')
(OUT/'quality.json').write_text(json.dumps(quality,indent=2)+'\n');(OUT/'quality.md').write_text(quality_markdown(quality))
result=dict(scope='Same actual 128-frame K/V geometry and capacity, synthetic fixed tensors. Grid predetermined: BM64/128 x warps4/8 x stages1/2. Selection by warmed kernel median among tolerance-passing candidates, no ground truth.',query_group_size=256,source_width=32,quality=quality,cases=[],passed=False,dense_speedup=None,dense_speedup_reason='Kernel only; no paired Dense full forward')
torch.set_num_threads(2);torch.manual_seed(20261105)
try:
 with torch.inference_mode():
  h,n,r,d,a=16,133248,8192,64,26009;g=r//256
  q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
  allids=torch.arange(n,device='cuda');sp=allids[allids%1041<5];patches=allids[allids%1041>=5]
  ids=torch.stack([patches[torch.randperm(len(patches),device='cuda')[:a]] for _ in range(g)]).sort(1).values.int();bias=torch.zeros(g,a,device='cuda');pos=(torch.arange(g,device='cuda')[:,None]*256+32).int()
  kw=dict(logit_bias=bias,probe_positions=pos,probe_count=1,probe_group_stride=1,query_group_size=256,debiased_log=True,split_export=False)
  ref,refe=attention(q,k,v,ids,sp,d**-.5,block_queries=32,num_warps=4,num_stages=2,**kw)
  for bm in (64,128):
   for warps in (4,8):
    for stages in (1,2):
     fn=lambda:attention(q,k,v,ids,sp,d**-.5,block_queries=bm,num_warps=warps,num_stages=stages,**kw)
     row=dict(block_queries=bm,num_warps=warps,num_stages=stages,passed=False)
     try:
      actual,e=fn();torch.cuda.synchronize();row.update(resources=e['compiler_resources'],scratch_bytes=e['scratch_bytes'])
      for name,x,y,atol,rtol in [('output',actual,ref,.016,.02),('lse',e['lse'],refe['lse'],1e-4,1e-5)]+[(k,e[k],refe[k],.025,.006) for k in ('debiased_log_patch_probability','debiased_log_special_probability')]:
       valid=torch.isfinite(x)&torch.isfinite(y);assert torch.equal(torch.isfinite(x),torch.isfinite(y));row[name+'_max_abs']=float((x[valid].float()-y[valid].float()).abs().max());torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
      samples=[]
      for _ in range(7):
       torch.cuda.synchronize();start=time.perf_counter();o,ev=fn();torch.cuda.synchronize();samples.append(time.perf_counter()-start)
      row.update(seconds=samples,median_seconds=statistics.median(samples),passed=True)
     except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)
     result['cases'].append(row);print(json.dumps(row),flush=True)
  result['winners']={str(bm):min((c for c in result['cases'] if c['block_queries']==bm and c['passed']),key=lambda c:c['median_seconds']) for bm in (64,128)}
  result['passed']=True;result['hardware']=torch.cuda.get_device_name()
finally:(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
