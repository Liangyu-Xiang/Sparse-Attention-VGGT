import argparse,json,sys
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True);torch.manual_seed(0);checks=[]
    for g,n,f,r in [(3,173,35,7),(64,132608,26009,512)]:
        labels=torch.arange(n,device='cuda')%f;grid=torch.arange(n,device='cuda')%r;pair=labels*r+grid;unique,count=torch.unique(pair,sorted=True,return_counts=True);sizes=torch.bincount(grid,minlength=r).float()
        coo=torch.sparse_coo_tensor(torch.stack((unique//r,unique%r)),count.float()/sizes[unique%r],size=(f,r),device='cuda').coalesce();csr=coo.to_sparse_csr();demand=torch.rand(g,r,device='cuda')
        before=torch.sparse.mm(coo,demand.T).T;after=torch.sparse.mm(csr,demand.T).T
        relative=float((before-after).norm()/before.norm());assert relative<1e-6
        k=max(1,f//4);b=before.topk(k,dim=1).indices;c=after.contiguous().topk(k,dim=1).indices
        checks.append(dict(groups=g,clusters=f,csr_relative_l2=relative,csr_exact_candidate_indices_equal=bool(torch.equal(b,c))))
        for kind in ['random','ties','zeros']:
            value=torch.rand(f,g,device='cuda').T
            if kind=='ties':value=(value*3).floor()
            if kind=='zeros':value.zero_()
            x=value.topk(k,dim=1).indices;y=value.contiguous().topk(k,dim=1).indices
            assert torch.equal(x,y),kind
    result=dict(complete=True,checks=checks,contiguous_topk_tie_indices_preserved=True,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Sparse observation layout correctness diagnostic',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
