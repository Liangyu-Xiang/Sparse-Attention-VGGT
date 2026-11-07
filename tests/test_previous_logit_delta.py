import sys,types
from pathlib import Path
import torch
import saf3r
root=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('saf3r.patch');pkg.__path__=[str(root/'saf3r/patch')]
sys.modules['saf3r.patch']=pkg;saf3r.patch=pkg
from saf3r.patch.previous_logit_delta import estimate,observe

def test_equal_headwise_softmax_implies_same_block_centered_logits():
 z=torch.tensor([[1.,2.,4.,0.],[3.,1.,0.,2.]])
 shifted=z+torch.tensor([[9.],[-3.]])
 assert torch.allclose(z.softmax(-1),shifted.softmax(-1))
 delta=z-z.reshape(2,2,2).mean(-1).repeat_interleave(2,1)
 other=shifted-shifted.reshape(2,2,2).mean(-1).repeat_interleave(2,1)
 assert torch.allclose(delta,other)

def test_equal_head_mean_does_not_imply_equal_headwise_residuals():
 z=torch.tensor([[4.,0.],[0.,4.]])
 swapped=z.flip(0)
 assert torch.allclose(z.softmax(-1).mean(0),swapped.softmax(-1).mean(0))
 assert not torch.allclose(z-z.mean(1,keepdim=True),swapped-swapped.mean(1,keepdim=True))

def test_cache_only_actual_ids_and_query_head_residuals():
 mean=torch.tensor([[[[1.,3.]]]]);grid=torch.tensor([0,0,1,1]);ids=torch.tensor([[0,2,-1]]);raw=torch.tensor([[[[2.,1.,-float('inf')]]]])
 delta=observe(mean,grid,ids,raw);assert delta.shape==raw.shape and torch.equal(delta,torch.tensor([[[[1.,-2.,0.]]]]))
 C=torch.tensor([[.2,.8]]);sizes=torch.tensor([2.,2.]);eligible=torch.ones(1,1,dtype=torch.bool)
 p,_=estimate(mean,C,C,mean,ids,delta,grid,sizes,eligible,'direct')
 assert torch.allclose(p,torch.tensor([[2.,1.,1.,3.]]).softmax(-1))


def test_centered050_is_half_observed_block_residual_with_uniform_unknowns():
 mean=torch.zeros(1,1,1,2);grid=torch.tensor([0,0,0,1]);sizes=torch.tensor([3.,1.])
 ids=torch.tensor([[0,1,-1]]);delta=torch.tensor([[[[1.,3.,0.]]]])
 coarse=torch.tensor([[.75,.25]]);eligible=torch.ones(1,1,dtype=torch.bool)
 prior,_=estimate(mean,coarse,coarse,mean,ids,delta,grid,sizes,eligible,'centered050')
 assert torch.allclose(prior,torch.tensor([[-.5,.5,0.,0.]]).softmax(-1))
 # Without previous evidence, all keys fall back to their current coarse logits.
 initial,_=estimate(mean,coarse,None,None,ids,None,grid,sizes,eligible,'centered050')
 assert torch.equal(initial,torch.full((1,4),.25))
