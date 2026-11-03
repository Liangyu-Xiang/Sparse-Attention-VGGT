"""Split-K mathematical equivalence and no-GT execution microbenchmark."""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_single_qk_grid import weighted_attention as old
from saf3r.triton.reuse_split_grid_attention import weighted_attention as new
from saf3r.patch.reuse_residual_exploration import canonical_layout

def diff(a,b):
    a=a.float();b=b.float();return dict(max_abs=float((a-b).abs().max()),relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;checks=[]
    for dtype in [torch.float32,torch.bfloat16]:
        g,h,n,d,patch,b=3,3,79,64,67,41
        q=torch.randn(h,n,d,device='cuda',dtype=dtype);k=torch.randn_like(q);v=torch.randn_like(q)
        ids=torch.stack([torch.randperm(patch,device='cuda')[:b] for _ in range(g)]).int();ids[1]=-1;ids[2,-5:]=-1
        grid=(ids.clamp_min(0)%7).long();bias=torch.rand(g,b,device='cuda')*4;special=torch.arange(patch,n,device='cuda')
        x,m,z=old(q,k,v,ids,bias,special,d**-.5,grid,7)
        for parts in [1,2,4,8]:
            y,mm,zz=new(q,k,v,ids,bias,special,d**-.5,grid,7,parts=parts)
            output=diff(x,y);mass=diff(m,mm);valid=torch.isfinite(z)
            probe=diff(z.masked_fill(~valid,0),zz.masked_fill(~valid,0))
            assert output['relative_l2']<(1e-5 if dtype==torch.float32 else .005),output
            assert mass['relative_l2']<1e-5 and probe['max_abs']==0,(mass,probe)
            checks.append(dict(dtype=str(dtype),parts=parts,output=output,mass=mass,probe=probe))
    # Full real tensor dimensions; synthetic legal supports, no GT tuning.
    g,h,n,d,b=64,16,133248,64,26009;grid,r=canonical_layout(128,37,28,2,2,'cuda');p=len(grid)
    q=torch.randn(h,g*32,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
    ids=((torch.arange(b,device='cuda')[None]*5+torch.arange(g,device='cuda')[:,None]*211)%p).int();keygrid=grid[ids.long()];bias=torch.rand(g,b,device='cuda');special=torch.arange(p,n,device='cuda')
    measurements={}
    for name,fn in [('old',lambda:old(q,k,v,ids,bias,special,d**-.5,keygrid,r))]+[(f'split{s}',lambda s=s:new(q,k,v,ids,bias,special,d**-.5,keygrid,r,parts=s)) for s in [2,4,8,16]]:
        samples=[]
        for i in range(5):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();value=fn();end.record();end.synchronize();del value
            if i>=2:samples.append(start.elapsed_time(end))
        measurements[name]=dict(median_ms=statistics.median(samples),samples_ms=samples)
    result=dict(complete=True,checks=checks,benchmark=measurements,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Kernel numerical/microbenchmark diagnostic only',dense_speedup=None,dense_reason='User requested no Dense rerun; this is not complete-forward timing')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
