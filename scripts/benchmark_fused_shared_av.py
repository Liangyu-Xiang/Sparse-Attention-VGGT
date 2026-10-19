"""Fixed-support lower bound: test fusion kernel BEFORE full quality sweeps."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from saf3r.triton.fused_shared_av import shared_av


def measured(fn,repeats):
    torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();cold=time.perf_counter()-t
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();values.append(time.perf_counter()-t)
    return dict(cold_seconds=cold,seconds=float(torch.tensor(values).median()),samples=values)


def run(args):
    if args.output.exists():raise ValueError('Use fresh output file')
    torch.manual_seed(0);torch.set_num_threads(4)
    records=[]
    for frames in args.frames:
        n=frames*1041;specials=frames*5
        q=torch.randn(16,n,64,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q)
        dense=measured(lambda:F.scaled_dot_product_attention(q[None],k[None],v[None]),args.repeats)
        groups=math.ceil(n/32)
        for retention in args.retentions:
            budget=math.floor(n*retention)-specials
            for alpha in [0.,.2,.8]:
                a=round(alpha*budget);vc=budget-a;p=vc+specials
                common=torch.arange(p,device='cuda')
                pk=k[:,common].contiguous();pv=v[:,common].contiguous()
                indices=((torch.arange(a,device='cuda')[None]+torch.arange(groups,device='cuda')[:,None]*97)%(n-p)+p).int().contiguous()
                result=measured(lambda:shared_av(q,k,v,pk,pv,indices,vc),args.repeats)
                records.append(dict(frames=frames,retention=retention,attention_ratio=alpha,
                    dense=dense,fused=result,attention_speedup=dense['seconds']/result['seconds'],
                    unique_edges_per_query=p+a,padded_qk_ratio=(math.ceil(p/64)*64+math.ceil(a/64)*64)/n,
                    index_bytes=indices.numel()*indices.element_size()))
                print(json.dumps(records[-1]),flush=True)
                del indices,pk,pv
        del q,k,v
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(protocol='Fixed support only; excludes selector/state, not end-to-end speedup',
        gpu=torch.cuda.get_device_name(),records=records),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',nargs='+',type=int,default=[2,50,128]);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--retentions',nargs='+',type=float,default=[.2,.15,.1]);run(p.parse_args())
