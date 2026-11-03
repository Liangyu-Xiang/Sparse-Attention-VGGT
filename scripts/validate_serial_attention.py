import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_single_qk_grid import weighted_attention as old
from saf3r.triton.reuse_serial_tuned import weighted_attention as new
from saf3r.patch.reuse_residual_exploration import canonical_layout

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True);torch.manual_seed(0);rows=[]
 for dtype in [torch.float32,torch.bfloat16]:
  h,n,d,g,b=3,79,64,3,41;q=torch.randn(h,n,d,device='cuda',dtype=dtype);k=torch.randn_like(q);v=torch.randn_like(q);ids=torch.stack([torch.randperm(67,device='cuda')[:b] for _ in range(g)]).int();ids[1]=-1;ids[2,-5:]=-1;grid=ids.clamp_min(0)%7;bias=torch.rand(g,b,device='cuda');special=torch.arange(67,n,device='cuda')
  x=old(q,k,v,ids,bias,special,d**-.5,grid,7);y=new(q,k,v,ids,bias,special,d**-.5,grid,7)
  assert all(torch.equal(a,b) for a,b in zip(x,y));rows.append(dict(dtype=str(dtype),output_grid_probe_bitwise_equal=True))
 g,h,n,d,b=64,16,133248,64,26009;grid,r=canonical_layout(128,37,28,2,2,'cuda');patches=len(grid);q=torch.randn(h,g*32,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k);ids=((torch.arange(b,device='cuda')[None]*5+torch.arange(g,device='cuda')[:,None]*211)%patches).int();kg=grid[ids.long()];bias=torch.rand(g,b,device='cuda');special=torch.arange(patches,n,device='cuda');bench={};reference=None
 for name,fn in [('original',old),('stage1',new)]:
  ts=[]
  for i in range(7):
   begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();value=fn(q,k,v,ids,bias,special,d**-.5,kg,r);end.record();end.synchronize()
   if i>=2:ts.append(begin.elapsed_time(end))
  if reference is None:reference=value
  else:assert all(torch.equal(a,b) for a,b in zip(reference,value))
  bench[name]=dict(median_ms=statistics.median(ts),samples_ms=ts)
 result=dict(complete=True,checks=rows,benchmark=bench,real_dimensions_bitwise_equal=True,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Serial Attention numerical and no-GT microbenchmark only',dense_speedup=None,dense_reason='No Dense rerun by user request');(a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
