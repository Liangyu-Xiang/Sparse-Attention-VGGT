import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from saf3r.triton import reuse_light_states as rule
from saf3r.triton.reuse_compact import budget
from saf3r.patch.reuse_light_graph import select

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True);torch.manual_seed(0);g,n,f,d,cap=3,173,35,64,64
 s=SimpleNamespace(_forward_serial=1,budget=cap,config=SimpleNamespace(dual_iterations=16,graph_capacity=8));rows=[];saved=None;saved_copy=None
 for layer in range(1,5):
  z=torch.randn(n,d,device='cuda');labels=(torch.arange(n,device='cuda')+layer)%f;reps=torch.full((f,),n,device='cuda',dtype=torch.long).scatter_reduce_(0,labels,torch.arange(n,device='cuda'),reduce='amin');pop=torch.bincount(labels,minlength=f).float();geo=rule.prepare(z,labels,pop)
  ids=torch.stack([torch.randperm(n,device='cuda')[:cap] for _ in range(g)]).int();hp=torch.rand(g,cap,device='cuda')*.01;bg=torch.rand(g,device='cuda')*.001;allowed=(torch.rand(f,g,device='cuda')>.2).T;o0=rule.moment(ids,hp,bg,z)
  errors,costs,qual=rule.states(ids,hp,bg,z,reps,labels,pop,allowed,o0,geo);chosen,_=budget(errors,costs,errors[:,:,0].amax(1)+1e-12,cap,16);expected=rule.pack(chosen,costs,reps,pop,qual,geo,cap)
  packed,weights,counts,stats,ee,ss=select(s,layer,ids,hp,bg,z,reps,labels,pop,allowed,o0,geo)
  assert all(torch.equal(x,y) for x,y in zip(expected,(packed,weights,counts))) and torch.equal(chosen,ss)
  if saved is not None:assert torch.equal(saved,saved_copy)
  else:saved=packed;saved_copy=packed.clone()
  rows.append(dict(layer=layer,ids_weights_counts_bitwise_equal=True,selected_states_bitwise_equal=True,cached_graphs=len(s.light_graphs)))
 result=dict(complete=True,checks=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Execution graph correctness diagnostic only',dense_speedup=None,dense_reason='No Dense model rerun');(a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
