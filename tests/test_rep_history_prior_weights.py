import pytest
import torch
from saf3r.patch.rep_history_prior_weights import reference,weights_A

def fixture(device='cpu',uniform=False):
 labels=torch.tensor([0,0,0,1,1,1,2,2],device=device);reps=torch.tensor([0,3,6],device=device)
 prior=torch.tensor([[.10,.20,.10,.07,.21,.02,.12,.18],[.03,.10,.27,.08,.12,.10,.10,.20]],device=device)
 if uniform:prior=torch.tensor([[.1]*8,[.1]*8],device=device)
 pi=torch.zeros((2,3),device=device).scatter_add_(1,labels[None].expand(2,-1),prior)
 selected=torch.tensor([[5,1,10],[5,5,0]],device=device)
 qual=torch.tensor([[True,True,False,False,False,False,False,False],[False,True,False,True,True,False,False,False]],device=device)
 ids=torch.tensor([[0,1,3,6,7],[0,1,3,4,-1]],device=device)
 return ids,labels,reps,selected,prior,pi,qual

def test_predicted_cluster_mass_conserved():
 ids,labels,reps,selected,prior,pi,qual=fixture();w=reference(ids,labels,reps,selected,prior,pi,qual)
 mass=torch.zeros_like(pi).scatter_add_(1,labels[ids.clamp_min(0)],prior.gather(1,ids.clamp_min(0))*w)
 torch.testing.assert_close(mass,pi*(selected!=0))
 # Rep cluster1 row0 total .30, representative prior .07.
 assert float(w[0,2])==pytest.approx(.30/.07)
 # Historical non-Rep keys retain unit weight.
 assert w[0,1]==1 and w[1,1]==1 and w[1,3]==1

def test_uniform_prior_recovers_population_weights():
 args=fixture(uniform=True);w=reference(*args)
 torch.testing.assert_close(w,torch.tensor([[2.,1.,3.,1.,1.],[2.,1.,2.,1.,0.]]))

def test_single_rep_history_equals_rep():
 ids,labels,reps,selected,prior,pi,qual=fixture();qual.zero_();qual[:,reps]=True
 hist=torch.full_like(selected,5);rep=torch.full_like(selected,1);ids=reps[None].expand(2,-1)
 torch.testing.assert_close(reference(ids,labels,reps,hist,prior,pi,qual),reference(ids,labels,reps,rep,prior,pi,qual))

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA numerical kernel test')
@pytest.mark.parametrize('uniform',[True,False])
def test_cuda_matches_reference(uniform):
 ids,labels,reps,selected,prior,pi,qual=fixture('cuda',uniform)
 hm=torch.zeros_like(pi).scatter_add_(1,labels[None].expand_as(prior),prior*qual);seen=qual[:,reps]
 actual=weights_A(ids.int(),labels,reps,selected,prior,pi,hm,seen,torch.empty(ids.shape,device='cuda'))
 torch.testing.assert_close(actual,reference(ids,labels,reps,selected,prior,pi,qual),rtol=1e-6,atol=1e-6)
