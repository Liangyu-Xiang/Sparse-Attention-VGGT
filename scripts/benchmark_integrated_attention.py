"""Fixed-input kernel parameter diagnostic, including sampled history export.

Selection uses numerical checks and runtime, never geometric test truth.
The measurements are not end-to-end model speed or geometric quality.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    import torch
    from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as attention
    from scripts.quality_metrics import quality_record,quality_markdown
    qrecord=quality_record({},'Fixed-input operator diagnostic including inline history/export; pose/recon_unposed not evaluated; no paired Dense full-forward timing')
    (a.output_dir/'quality.json').write_text(json.dumps(qrecord,indent=2)+'\n')
    (a.output_dir/'quality.md').write_text(quality_markdown(qrecord))
    torch.set_num_threads(2);torch.manual_seed(763)
    h,r,n,d,g,width=16,16384,133248,64,64,26009
    q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
    all_ids=torch.arange(n,device='cuda');special=all_ids[all_ids%1041<5]
    patches=all_ids[all_ids%1041>=5]
    ids=patches[torch.rand(g,len(patches),device='cuda').argsort(dim=1)[:,:width]].sort(1).values.int().contiguous()
    ids[-1,-11:]=-1
    bias=torch.randn(g,width,device='cuda').masked_fill(ids<0,-torch.inf)
    kw=dict(logit_bias=bias,query_group_size=256,probe_group_stride=1,debiased_log=True,split_export=True)
    reference,ev=attention(q,k,v,ids,special,d**-.5,block_queries=32,block_keys=128,num_warps=4,num_stages=2,**kw)
    cases=[(32,128,4,2,False),(32,128,4,2,True),(64,128,4,1,False),(64,128,4,1,True),
           (32,64,4,2,True),(32,256,4,1,True),(32,256,8,1,True),
           (64,64,4,1,True),(64,256,4,1,True),(64,256,8,1,True),
           (64,128,8,1,True),(128,64,8,1,True)]
    rows=[]
    for bm,bk,warps,stages,split in cases:
        row=dict(block_queries=bm,block_keys=bk,num_warps=warps,num_stages=stages,execution_split=split,quality=qrecord)
        try:
            def run():return attention(q,k,v,ids,special,d**-.5,block_queries=bm,block_keys=bk,num_warps=warps,num_stages=stages,execution_split=split,**kw)
            out,e=run();row['output_max_abs']=float((reference.float()-out.float()).abs().max())
            row['output_relative_l2']=float((reference.float()-out.float()).norm()/reference.float().norm())
            row['history_log_max_abs']=float((ev['debiased_log_patch_probability'][...,:-11]-e['debiased_log_patch_probability'][...,:-11]).abs().max())
            row['finite']=bool(torch.isfinite(out).all())
            row['numerically_valid']=row['finite'] and row['output_relative_l2']<.02 and row['history_log_max_abs']<.1
            row['compiler_resources']=e['compiler_resources']
            for _ in range(2):run()
            samples=[]
            for _ in range(7):
                torch.cuda.synchronize();start=time.perf_counter();o,e=run();torch.cuda.synchronize()
                samples.append(time.perf_counter()-start);del o,e
            import statistics
            row['median_seconds']=statistics.median(samples);row['samples_seconds']=samples
            row['error']=None
        except Exception as exc:row.update(error=repr(exc),numerically_valid=False)
        rows.append(row)
        summary=dict(shape=dict(h=h,r=r,n=n,d=d,g=g,patch_support_width=width),rows=rows,
                     scope='Synthetic same-input attention+LSE+actual sampled history export and temporary allocation; not model timing',quality=qrecord)
        (a.output_dir/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
