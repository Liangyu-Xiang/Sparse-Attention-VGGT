"""No-GT Triton execution tuning with BF16 output/history fidelity gates."""
import argparse,itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
 import torch
 from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as attn
 from scripts.quality_metrics import quality_record,quality_markdown
 qrec=quality_record({},'Random-shape Attention kernel-only tuning, no model/geometric evaluation; Dense speedup unavailable')
 save(a.output/'quality.json',qrec);(a.output/'quality.md').write_text(quality_markdown(qrec))
 torch.manual_seed(473);h,n,g,d=16,132608,16,64;r=g*256;s=640
 q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k);special=torch.arange(s,device='cuda',dtype=torch.int64)
 positions=(torch.arange(g,device='cuda')[:,None]*256+32).int()
 configurations=[(64,128,4,2)]+[x for x in itertools.product((32,64,128),(64,128,256),(4,8),(1,2,3)) if x!=(64,128,4,2)]
 rows=[]
 for fraction in (.1,.22):
  cap=int(n*fraction)-s
  ids=(torch.arange(cap,device='cuda')*(n-s)//cap+s).int()[None].expand(g,-1).contiguous();bias=torch.randn(g,cap,device='cuda')*.3
  target=None;target_p=None
  for bm,bk,warps,stages in configurations:
   params=dict(block_queries=bm,block_keys=bk,num_warps=warps,num_stages=stages)
   row=dict(retention=fraction,**params)
   try:
    def forward():return attn(q,k,v,ids,special,1/d**.5,bias,positions,query_group_size=256,probe_group_stride=1,debiased_log=True,split_export=True,**params)
    out,evidence=forward();torch.cuda.synchronize()
    if target is None:target=out.clone();target_p=evidence['debiased_log_patch_probability'].exp().clone()
    rel=float((out.float()-target.float()).square().sum()/target.float().square().sum().clamp_min(1e-20))**.5
    prob=evidence['debiased_log_patch_probability'].exp();prel=float((prob-target_p).abs().sum()/target_p.sum())
    events=[]
    for rep in range(3):
     a1=torch.cuda.Event(enable_timing=True);b1=torch.cuda.Event(enable_timing=True);a1.record();out,evidence=forward();b1.record();b1.synchronize();events.append(a1.elapsed_time(b1)/1000)
    row.update(seconds=sorted(events)[1],samples=events,output_relative_l2=rel,history_probability_relative_l1=prel,numerical_pass=rel<=.005 and prel<=.005,complete=True)
   except Exception as exc:row.update(complete=False,error=f'{type(exc).__name__}: {exc}')
   rows.append(row);save(a.output/'progress.json',dict(rows=rows,quality=qrec,speedup_vs_dense=None));print('KERNEL',row,flush=True)
 eligible=[]
 for bm,bk,warps,stages in configurations:
  rr=[r for r in rows if (r['block_queries'],r['block_keys'],r['num_warps'],r['num_stages'])==(bm,bk,warps,stages)]
  if len(rr)==2 and all(r.get('numerical_pass',False) for r in rr):eligible.append(dict(block_queries=bm,block_keys=bk,num_warps=warps,num_stages=stages,score=sum(r['seconds'] for r in rr)/2))
 eligible.sort(key=lambda r:r['score']);save(a.output/'result.json',dict(complete=True,quality=qrec,rows=rows,ranked=eligible,speedup_vs_dense=None,scope='One fixed random 4096-Query window,16heads,132608Keys; complete AV+sampled evidence export; not end-to-end model time'))
if __name__=='__main__':main()
