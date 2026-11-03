"""Changing Value/cluster/Rep on every layer must update ALL graph inputs."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.patch.reuse_window_graph import select,eager
from saf3r.triton.reuse_compact import prepare_geometry
from saf3r.triton.reuse_drop_moment import prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True);torch.manual_seed(0)
    g,n,f,d,cap=3,173,35,64,70;state=NS(graph_cache_hits=0,graph_cache_misses=0,graph_cache_evictions=0,graph_constant_updates=0,budget=cap,_forward_serial=1,config=NS(reuse_window_graphs=True,dual_iterations=16,drop_multiplier=2.))
    rows=[]
    for layer in range(4):
        labels=(torch.arange(n,device='cuda')+layer)%f;pop=torch.bincount(labels,minlength=f).float();reps=torch.full((f,),n,device='cuda',dtype=torch.long).scatter_reduce_(0,labels,torch.arange(n,device='cuda'),reduce='amin')
        z=torch.randn(n,d,device='cuda');z/=z.square().sum(1).mean().sqrt();geometry=prepare_geometry(z,labels,reps,pop,l2=True);geometry['drop_bins']=prepare(pop);geometry['fuse_state_kernel']=True
        prior=torch.rand(g,n,device='cuda');prior/=prior.sum(1,keepdim=True);pi=torch.zeros(g,f,device='cuda').scatter_add_(1,labels[None].expand(g,-1),prior);ids=torch.stack([torch.randperm(n,device='cuda')[:cap] for _ in range(g)]).int();allowed=torch.rand(g,f,device='cuda')>.5
        x,r,w,stats=eager(prior,pi,ids,allowed,z,reps,labels,pop,geometry,cap,16,2.)
        y,rr,ww,ss=select(state,layer,prior,pi,ids,allowed,z,reps,labels,pop,geometry)
        assert torch.equal(x,y) and torch.equal(r,rr)
        error=float((w-ww).norm()/w.norm().clamp_min(1e-30));assert error<1e-5 and not ss['implementation_validation_error'].any()
        rows.append(dict(layer=layer,ids_bitwise_equal=True,weight_relative_l2=error,cached_graphs=len(state._reused_window_graphs)))
    assert len(state._reused_window_graphs)==1
    result=dict(complete=True,checks=rows,all_current_value_geometry_copied=True,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Cross-layer execution-buffer correctness only, not quality evaluation',dense_speedup=None,dense_reason='No Dense rerun by user request')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
