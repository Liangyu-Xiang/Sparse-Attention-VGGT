"""No-GT equivalence tests for dead-code removal and 11->4 state layout."""
import os,sys
from pathlib import Path
SNAP=Path(os.environ.get('ZDROP_SOURCE_ROOT', str(Path(__file__).resolve().parents[1])));sys.path[:0]=[str(SNAP),str(SNAP/'evaluation')]
import torch
from saf3r.patch import zdrop_history_states as reference,zdrop_history_cleanup as fast
from saf3r.triton import reuse_compact as compact

def main():
 torch.manual_seed(81);dev='cuda';pop=torch.tensor([1,2,3,4,5,6,7,4],device=dev,dtype=torch.float32);f=len(pop);n=int(pop.sum());g=3
 labels=torch.repeat_interleave(torch.arange(f,device=dev),pop.long());reps=torch.cat((torch.zeros(1,device=dev,dtype=torch.long),pop.long().cumsum(0)[:-1]))
 z=torch.randn((n,64),device=dev);z/=z.square().sum(1).mean().sqrt()
 prior=torch.randn(g,n,device=dev).softmax(-1)*.9;pi=torch.zeros(g,f,device=dev).scatter_add_(1,labels[None].expand(g,-1),prior)
 ids=torch.arange(n,device=dev,dtype=torch.int32)[None].expand(g,-1).clone();ids[0,::3]=-1;ids[2].fill_(-1)
 allowed=torch.rand(g,f,device=dev)>.25;geo=compact.prepare_geometry(z,labels,reps,pop,l2=True)
 moment=torch.zeros(g,f,64,device=dev)
 for j in range(g):moment[j].index_add_(0,labels,prior[j,:,None]*z)
 geo['drop_values']=moment.norm(dim=-1)
 c0=reference.gate(prior,pi,pop,ids,labels,reps);c1=reference.gate(prior,pi,pop,ids,labels,reps)
 hids,_=compact.top_history(ids,prior,reps,geo,4)
 e0,k0=reference.states(pi,z,reps,labels,pop,hids,allowed,prior,1.,geo,c0);o0=geo['objective'].clone()
 e1,k1=fast.states(pi,z,reps,labels,pop,None,allowed,prior,1.,geo,c1);o1=geo['objective'].clone()
 kept=torch.tensor([0,1,5,10],device=dev)
 assert torch.allclose(e0[:,:,kept],e1,atol=1e-5,rtol=1e-5)
 assert torch.allclose(o0[:,:,kept],o1,atol=1e-5,rtol=1e-5)
 assert torch.equal(k0[:,:,kept],k1) and torch.equal(c0['owners'],c1['owners'])
 print('PASS loss/objective/cost/owners match retained original slots; FP32 tolerance1e-5',flush=True)
 for cap in [0,1,2,4,8,12,20,32]:
  s0,count0=compact.budget(o0,k0,e0[:,:,0].amax(1)+1e-12,cap,16)
  ss,count1=compact.budget(o1,k1,e1[:,:,0].amax(1)+1e-12,cap,16);s1=fast.decode(ss)
  assert torch.equal(s0,s1),(cap,s0,s1)
  assert torch.equal(count0,count1)
  p0,_,_=reference.pack(s0,hids,reps,labels,pop,cap,c0);p1,_,_=fast.pack(s1,None,reps,labels,pop,cap,c1)
  assert torch.equal(p0,p1)
  w0=reference.transfer(s0,labels,reps,hids,prior,pi,p0,geo,c0);w1=fast.transfer(s1,labels,reps,None,prior,pi,p1,geo,c1)
  assert torch.equal(w0,w1)
 print('PASS identical state choices/counts/packed IDs/weights over8 budgets, including zero budget, singleton and unavailable Exact',flush=True)
 # Include exact objective ties and preserve leftmost legal-state ordering.
 for cap in [0,1,4,16]:
  err=torch.zeros((g,f,11),device=dev);err[:,:,[2,3,4,6,7,8,9]]=float('inf');cost=torch.ones_like(err);cost[:,:,0]=0
  s0,_=compact.budget(err,cost,torch.ones(g,device=dev),cap,16)
  ss,_=compact.budget(err[:,:,kept].contiguous(),cost[:,:,kept].contiguous(),torch.ones(g,device=dev),cap,16)
  assert torch.equal(s0,fast.decode(ss))
 print('PASS exact tie priority retained',flush=True)
 print('ALL CLEANUP EQUIVALENCE CHECKS PASSED',flush=True)
if __name__=='__main__':main()
