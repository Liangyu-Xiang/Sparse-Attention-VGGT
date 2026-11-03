import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.patch.reuse_window_graph import Window,eager
from saf3r.triton.reuse_compact import prepare_geometry
from saf3r.triton.reuse_drop_moment import prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);g,n,f,d,cap=3,173,35,64,70
    z=torch.randn(n,d,device='cuda');z/=z.square().sum(1).mean().sqrt();labels=torch.arange(n,device='cuda')%f;reps=torch.arange(f,device='cuda');pop=torch.bincount(labels,minlength=f).float()
    geometry=prepare_geometry(z,labels,reps,pop,l2=True);geometry['drop_bins']=prepare(pop)
    rows=[];window=None
    for seed in range(3):
        prior=torch.rand(g,n,device='cuda');prior/=prior.sum(1,keepdim=True);pi=torch.zeros(g,f,device='cuda').scatter_add_(1,labels[None].expand(g,-1),prior)
        ids=torch.stack([torch.randperm(n,device='cuda')[:cap] for _ in range(g)]).int();allowed=torch.rand(g,f,device='cuda')>.5
        x,real,w,stats=eager(prior,pi,ids,allowed,z,reps,labels,pop,geometry,cap,16,2.)
        if window is None:window=Window(prior,pi,ids,allowed,z,reps,labels,pop,geometry,cap,16,2.)
        y,real2,w2,stats2=window(prior,pi,ids,allowed)
        assert torch.equal(x,y) and torch.equal(real,real2)
        err=float((w-w2).norm()/w.norm().clamp_min(1e-30));assert err<1e-5 and not stats2['implementation_validation_error'].any()
        rows.append(dict(fresh_input_seed=seed,ids_bitwise_equal=True,weight_relative_l2=err))
    result=dict(complete=True,checks=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Window execution correctness diagnostic',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
