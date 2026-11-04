"""Native FlashAttention dispatch, bias/Special/padding, and original v8 evidence."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.profiler import profile,ProfilerActivity
from saf3r.triton.reuse_torch_flash import weighted_attention
from saf3r.triton.reuse_serial_tuned import weighted_attention as old
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
torch.manual_seed(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;checks=[];start=time.perf_counter()
for dtype in [torch.float16,torch.bfloat16]:
 for rows,groups,cap in [(79,3,48),(96,3,48),(17,1,48),(79,3,1408)]:
  h,n,d,regions=3,max(93,cap+53),64,7;q=torch.randn(h,rows+9,d,device='cuda',dtype=dtype)[:,3:3+rows];k=torch.randn(h,n+5,d,device='cuda',dtype=dtype);v=torch.randn_like(k);specials=torch.arange(n,n+5,device='cuda');grid=torch.arange(n,device='cuda')%regions
  ids=torch.stack([torch.randperm(n,device='cuda')[:cap] for _ in range(groups)]).int();ids[:,-min(513,cap//3):]=-1;ids[-1,:]=-1
  bias=(torch.rand(groups,cap,device='cuda')*4).masked_fill(ids<0,-torch.inf);kg=grid[ids.long().clamp_min(0)];layout={};olayout={}
  with profile(activities=[ProfilerActivity.CPU]) as prof:out,summary,raw=weighted_attention(q,k,v,ids,bias,specials,.125,kg,regions,evidence_layout=layout,group_chunk=2)
  ref,osummary,oraw=old(q,k,v,ids,bias,specials,.125,kg,regions,evidence_layout=olayout)
  full=torch.empty_like(out,dtype=torch.float32)
  for g in range(groups):
   valid=ids[g]>=0;keys=torch.cat([specials,ids[g,valid].long()]);b=torch.cat([torch.zeros(len(specials),device='cuda'),bias[g,valid]]);pq=q[:,g*32:min((g+1)*32,rows)].float()
   full[:,g*32:min((g+1)*32,rows)]=(pq@k[:,keys].float().transpose(-1,-2)*.125+b).softmax(-1)@v[:,keys].float()
  rel=float((out.float()-full).norm()/full.norm());serial_rel=float((out.float()-ref.float()).norm()/ref.float().norm());assert rel<(.001 if dtype==torch.float16 else .005),(dtype,rel)
  assert torch.isfinite(out).all();assert torch.equal(torch.isfinite(raw),torch.isfinite(oraw));finite=torch.isfinite(raw);raw_error=float((raw[finite]-oraw[finite]).abs().max()) if finite.any() else 0.;summary_error=float((summary-osummary).abs().max());assert raw_error<1e-5 and summary_error<2e-6,(raw_error,summary_error)
  assert torch.equal(layout['selected_grid_order'],olayout['selected_grid_order']) and torch.equal(layout['selected_grid_offsets'],olayout['selected_grid_offsets'])
  ops=[x.key for x in prof.key_averages() if 'scaled_dot_product' in x.key];assert 'aten::_scaled_dot_product_flash_attention' in ops and not any('efficient_attention' in op or 'attention_math' in op for op in ops),ops
  checks.append(dict(dtype=str(dtype),query_rows=rows,fp32_relative_l2=rel,serial_relative_l2=serial_rel,raw_probe_max_abs=raw_error,grid_summary_max_abs=summary_error,dispatch=ops,specials=True,padding=True,empty_patch_support=True,tail_queries=True,bias=True))
result=dict(complete=True,checks=checks,diagnostic_wall_seconds=time.perf_counter()-start,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Small forced-backend/kernel numerical diagnostic; no pose/reconstruction evaluation',dense_speedup=None,dense_reason='Kernel diagnostic,not paired full-forward timing')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
