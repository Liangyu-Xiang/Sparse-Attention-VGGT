"""No-GT kernel tile tuning at the real heads128 tensor dimensions."""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.patch.reuse_residual_exploration import canonical_layout
from saf3r.triton.reuse_grid_local_history import normalizers


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);g,h,a=64,16,26009;grid,r=canonical_layout(128,37,28,2,2,'cuda');sizes=torch.bincount(grid,minlength=r).float()
    ids=(torch.arange(a,device='cuda')[None]*5+torch.arange(g,device='cuda')[:,None]*211)%len(grid)
    bank=grid[ids];order=bank.argsort(dim=1,stable=True).int();counts=torch.zeros(g,r,device='cuda',dtype=torch.int32).scatter_add_(1,bank,torch.ones_like(bank,dtype=torch.int32));off=torch.nn.functional.pad(counts.cumsum(1,dtype=torch.int32),(1,0)).contiguous()
    raw=torch.randn(g,h,4,a,device='cuda');base=torch.randn(g,h,4,r,device='cuda');rows=[]
    for tile in [1,2,4,8,16]:
        values=[]
        for i in range(7):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();den,err=normalizers(raw,base,order,off,sizes,266,tile);end.record();end.synchronize()
            if i>=2:values.append(start.elapsed_time(end))
        rows.append(dict(tile=tile,median_ms=statistics.median(values),samples_ms=values))
    result=dict(rows=rows,chosen_tile=min(rows,key=lambda r:r['median_ms'])['tile'],scope='Synthetic legal support, heads128 actual dimensions, kernel timing only, no GT',quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Kernel parameter tuning only',dense_speedup=None,dense_reason='No full-forward control')
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
