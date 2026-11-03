import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.patch.zdrop_history_states import gate
from saf3r.patch.zdrop_history_cleanup import states as old
from saf3r.triton.reuse_fused_states import states as new
from saf3r.triton.reuse_compact import prepare_geometry
from saf3r.triton.reuse_drop_moment import prepare,drop_norm

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True);torch.manual_seed(0);rows=[]
    for g,n,f in [(3,173,17),(8,8192,1024)]:
        z=torch.randn(n,64,device='cuda');z[f]=z[0];z/=z.square().sum(1).mean().sqrt();labels=torch.arange(n,device='cuda')%f;reps=torch.arange(f,device='cuda');pop=torch.bincount(labels,minlength=f).float()
        geometry=prepare_geometry(z,labels,reps,pop,l2=True);geometry['drop_bins']=prepare(pop)
        prior=torch.rand(g,n,device='cuda');prior/=prior.sum(1,keepdim=True);pi=torch.zeros(g,f,device='cuda').scatter_add_(1,labels[None].expand(g,-1),prior);ids=torch.stack([torch.randperm(n,device='cuda')[:n//5] for _ in range(g)]).int();allowed=torch.rand(g,f,device='cuda')>.5
        c=gate(prior,pi,pop,ids,labels,reps);d=copy.copy(c);geometry['drop_values']=drop_norm(prior,z,geometry,f,geometry['drop_bins'])
        e,cost=old(pi,z,reps,labels,pop,None,allowed,prior,2.,geometry,c);objective=geometry['objective'].clone()
        ee,cc=new(prior,z,reps,labels,pop,allowed,geometry,d);finite=torch.isfinite(e)
        assert torch.equal(finite,torch.isfinite(ee)) and torch.equal(cost,cc) and torch.equal(c['owners'],d['owners'])
        maximum=float((e[finite]-ee[finite]).abs().max());rel=float((e[finite]-ee[finite]).norm()/e[finite].norm());assert maximum<1e-6 and rel<1e-5
        good=torch.isfinite(objective);assert torch.allclose(objective[good],geometry['objective'][good],atol=1e-6,rtol=1e-5)
        rows.append(dict(groups=g,tokens=n,costs_bitwise_equal=True,owners_bitwise_equal=True,error_max_abs=maximum,error_relative_l2=rel))
    result=dict(complete=True,checks=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='State kernel numerical diagnostic',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
