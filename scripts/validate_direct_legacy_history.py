"""Exact-input original v8 Grid summaries/conditional-cache oracle."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
from saf3r.triton import reuse_serial_tuned as old,reuse_grid_local_history as reference,reuse_direct_legacy_observation as new
from saf3r.triton.reuse_ready_history import decode
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
torch.manual_seed(13);torch.backends.cuda.matmul.allow_tf32=False;checks=[];start=time.perf_counter()
for dtype in [torch.float16,torch.bfloat16]:
 h,n,d,s,g,rows,cap,regions=3,93,64,5,3,79,55,7;q=torch.randn(h,rows+5,d,device='cuda',dtype=dtype)[:,:rows];k=torch.randn(h,n+s,d,device='cuda',dtype=dtype);v=torch.randn_like(k);specials=torch.arange(n,n+s,device='cuda');grid=torch.arange(n,device='cuda')%regions;sizes=torch.bincount(grid,minlength=regions).float();pooled=torch.stack([k[:,:n][:,grid==r].float().mean(1) for r in range(regions)],1)
 positions=(torch.arange(g,device='cuda')[:,None]*32+torch.arange(4,device='cuda')[None]*8+4).clamp_max(rows-1).int();pq=q[:,positions].permute(1,0,2,3).float();base=torch.einsum('ghpd,hrd->ghpr',pq,pooled)*.125;special=torch.einsum('ghpd,hsd->ghps',pq,k[:,specials].float())*.125
 ids=torch.stack([torch.randperm(n,device='cuda')[:cap].sort().values for _ in range(g)]).int();ids[:,-13:]=-1;ids[-1,:]=-1;ids[0,11]=-1;kg=grid[ids.long().clamp_min(0)].contiguous();bias=(torch.rand(g,cap,device='cuda')*3).masked_fill(ids<0,-torch.inf);layout={}
 ro,rs,raw=old.weighted_attention(q,k,v,ids,bias,specials,.125,kg,regions,evidence_layout=layout)
 out,e=direct_sparse_attention(q,k,v,ids,specials,.125,bias)
 ns,den,counts=new.observe(q,k,ids,kg,bias,positions,e['lse'],base.contiguous(),sizes,.125)
 rd,_=reference.normalizers(raw,base,layout['selected_grid_order'],layout['selected_grid_offsets'],sizes,int(sizes.max()))
 summary_error=float((rs-ns).abs().max());den_error=float((rd-den).abs().max());assert summary_error<3e-6 and den_error<4e-6,(summary_error,den_error)
 rz=reference.compatibility(rd,special);nz=reference.compatibility(den,special);old_cache=torch.empty((g,1,4,(cap*14+7)//8),device='cuda',dtype=torch.uint8);new_cache=torch.empty_like(old_cache);uniform=torch.empty((g,1,4,regions),device='cuda')
 reference.store(raw,rd,rz,ids,grid,base,old_cache,uniform,1e-30)
 new.store(q,k,ids,kg,positions,den,nz,new_cache,.125)
 avalues=decode(old_cache,cap);bvalues=decode(new_cache,cap);error=float((avalues-bvalues).abs().max());assert torch.allclose(avalues,bvalues,atol=2e-7,rtol=.002),error
 checks.append(dict(dtype=str(dtype),grid_summary_max_abs=summary_error,grid_denominator_max_abs=den_error,conditional_cache_max_abs=error,raw_head_probe_logits_materialized=False,new_grid_sort=False,includes_padding_holes=True))
result=dict(complete=True,checks=checks,diagnostic_wall_seconds=time.perf_counter()-start,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Original-v8 history-definition numerical oracle;no geometry evaluated',dense_speedup=None,dense_reason='Diagnostic only,not full-forward timing')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
