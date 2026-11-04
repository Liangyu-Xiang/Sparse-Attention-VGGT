"""Fixed-input/fixed-support correctness and timing for BM32/64/128, stages=2."""
import json,math,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as attention
from scripts.quality_metrics import quality_record,quality_markdown
OUT=ROOT/'outputs/source32_query_tiles_20261105/operators';OUT.mkdir(exist_ok=False,parents=True)
qrecord=quality_record({},'Explicit fixed-input attention operator diagnostic; no camera pose or recon_unposed evaluation')
(OUT/'quality.json').write_text(json.dumps(qrecord,indent=2)+'\n');(OUT/'quality.md').write_text(quality_markdown(qrecord))
results=dict(passed=False,pipeline_stages=2,warps=4,split_export=False,query_group_size=256,cases=[],timings=[],quality=qrecord,dense_speedup=None,dense_speedup_unavailable='Operator-only timing; no paired full-forward Dense')
torch.set_num_threads(2);torch.manual_seed(20261105)
try:
 with torch.inference_mode():
  for r in (1,256,519,1031):
   h,n,d,a=4,401,64,73;g=math.ceil(r/256)
   q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
   ids=torch.stack([torch.randperm(n-5,device='cuda')[:a]+5 for _ in range(g)]).sort(1).values.int()
   ids[-1,-4:]=-1
   sp=torch.arange(5,device='cuda');bias=torch.zeros(g,a,device='cuda').masked_fill(ids<0,-torch.inf)
   pos=(torch.arange(g,device='cuda')[:,None]*256+32).clamp_max(r-1).int()
   kw=dict(logit_bias=bias,probe_positions=pos,probe_count=1,probe_group_stride=1,query_group_size=256,debiased_log=True,num_stages=2,num_warps=4,split_export=False)
   reference,refe=attention(q,k,v,ids,sp,d**-.5,block_queries=32,**kw)
   for bm in (64,128):
    actual,e=attention(q,k,v,ids,sp,d**-.5,block_queries=bm,**kw)
    row=dict(rows=r,block_queries=bm,resources=e['compiler_resources'])
    for name,x,y,atol,rtol in [('output',actual,reference,.016,.02),('lse',e['lse'],refe['lse'],1e-4,1e-5)]+[(name,e[name],refe[name],.025,.006) for name in ('debiased_log_patch_probability','debiased_log_special_probability')]:
     valid=torch.isfinite(x)&torch.isfinite(y);assert torch.equal(torch.isfinite(x),torch.isfinite(y)),name
     row[name+'_max_abs']=float((x[valid].float()-y[valid].float()).abs().max())
     torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
    results['cases'].append(row)
  # Actual 128-frame K/V geometry and actual retained-Key capacity, fixed tensors.
  h,n,r,d,a=16,133248,8192,64,26009;g=math.ceil(r/256)
  q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
  allids=torch.arange(n,device='cuda');sp=allids[allids%1041<5];patches=allids[allids%1041>=5]
  ids=torch.stack([patches[torch.randperm(len(patches),device='cuda')[:a]] for _ in range(g)]).sort(1).values.int()
  pos=(torch.arange(g,device='cuda')[:,None]*256+32).int();bias=torch.zeros(g,a,device='cuda')
  kw=dict(logit_bias=bias,probe_positions=pos,probe_count=1,probe_group_stride=1,query_group_size=256,debiased_log=True,num_stages=2,num_warps=4,split_export=False)
  ref,refe=attention(q,k,v,ids,sp,d**-.5,block_queries=32,**kw)
  for bm in (32,64,128):
   fn=lambda:attention(q,k,v,ids,sp,d**-.5,block_queries=bm,**kw)
   actual,e=fn();torch.cuda.synchronize()
   row=dict(block_queries=bm,resources=e['compiler_resources'],scratch_bytes=e['scratch_bytes'])
   for name,x,y,atol,rtol in [('output',actual,ref,.016,.02),('lse',e['lse'],refe['lse'],1e-4,1e-5)]+[(name,e[name],refe[name],.025,.006) for name in ('debiased_log_patch_probability','debiased_log_special_probability')]:
    valid=torch.isfinite(x)&torch.isfinite(y);assert torch.equal(torch.isfinite(x),torch.isfinite(y));row[name+'_max_abs']=float((x[valid].float()-y[valid].float()).abs().max());torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
   samples=[]
   for _ in range(7):
    torch.cuda.synchronize();start=time.perf_counter();o,ev=fn();torch.cuda.synchronize();samples.append(time.perf_counter()-start)
   row.update(seconds=samples,median_seconds=statistics.median(samples));results['timings'].append(row)
  results.update(passed=True,hardware=torch.cuda.get_device_name(),benchmark_scope=dict(heads=h,total_keys=n,query_rows=r,query_groups=g,retained_patch_capacity=a,special_keys=len(sp),source='Synthetic fixed tensors with actual geometry; not a full model speed comparison'))
finally:
 (OUT/'result.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
