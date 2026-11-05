import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton.reuse_residual_add import scaled_add
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.manual_seed(271);rows=[]
for xd in (torch.float32,torch.bfloat16,torch.float16):
 for yd in (torch.float32,torch.bfloat16,torch.float16):
  for n,d in ((17,64),(4097,1024)):
   x=torch.randn(1,n,d,device='cuda',dtype=xd);y=torch.randn_like(x,dtype=yd);scale=torch.randn(d,device='cuda')
   expected=x+y*scale;actual=scaled_add(x,y,scale)
   row=dict(xdtype=str(xd),ydtype=str(yd),rows=n,width=d,equal=torch.equal(actual,expected),max_abs=float((actual-expected).abs().max()));rows.append(row);assert row['equal'],row
result=dict(passed=True,cases=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='LayerScale/residual numerical equivalence only; no geometry',speedup_vs_dense=None,speedup_reason='Operator diagnostic only')
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),flush=True)
