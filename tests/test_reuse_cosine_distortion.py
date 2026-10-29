import pytest
import torch
from saf3r.patch.reuse_cosine_distortion import normalized_values,history_prior,cosine_states,transfer_weights
from saf3r.patch.reuse_residual_exploration import pack_support,ReuseExploreConfig,ReuseExploreState,ReuseResidualExploration,reference_attention
from saf3r.models.official_vggt.layers.attention import Attention

def fixture(device='cpu'):
 torch.manual_seed(81);features=torch.randn(11,16,device=device)
 labels=torch.tensor([0,0,0,0,1,1,1,2,2,2,2],device=device);pop=torch.bincount(labels).float()
 reps=torch.tensor([0,4,7],device=device);pi=torch.tensor([[.2,.3,.5],[.6,.4,0.]],device=device)
 hids=torch.tensor([[[1,2,3,-1],[5,6,-1,-1],[8,9,10,-1]],[[2,1,3,-1],[6,5,-1,-1],[9,8,10,-1]]],device=device)
 old=torch.tensor([[1,2,5,8,-1],[1,2,3,5,6]],device=device);weight=torch.tensor([[.18,.001,.4,.2,0],[.1,.2,.3,.1,.2]],device=device)
 grid=torch.zeros(11,device=device,dtype=torch.long);beta=torch.ones(2,1,device=device)
 prior=history_prior(pi,labels,pop,old,weight,beta,grid)
 return normalized_values(features),labels,pop,reps,pi,hids,prior

def test_prior_cluster_mass_conservation_and_full_observation():
 z,labels,pop,reps,pi,hids,prior=fixture()
 total=torch.zeros_like(pi).scatter_add_(1,labels[None].expand(2,-1),prior)
 torch.testing.assert_close(total,pi)
 assert (prior[0]>0).all() and (prior[1,labels==2]==0).all()
 old=torch.arange(11)[None].repeat(2,1);w=torch.ones(2,11)*.01
 full=history_prior(pi,labels,pop,old,w,torch.ones(2,1),torch.zeros(11,dtype=torch.long))
 torch.testing.assert_close(torch.zeros_like(pi).scatter_add_(1,labels[None].expand(2,-1),full),pi)


def test_no_cancellation_mean_equals_rep_but_cosine_distortion_positive():
 x=torch.tensor([[1.,0.],[1.,1.],[1.,-1.]])
 assert torch.equal(x.mean(0),x[0])
 z=normalized_values(x);pi=torch.tensor([[.2]]);labels=torch.zeros(3,dtype=torch.long);pop=torch.tensor([3.]);reps=torch.tensor([0]);h=torch.full((1,1,4),-1,dtype=torch.long)
 a=history_prior(pi,labels,pop,None,None,None,None)
 errors,costs=cosine_states(pi,z,reps,labels,pop,h,torch.ones_like(pi,dtype=torch.bool),a)
 expected=.2/3*2*(1-2**-.5)
 assert errors[0,0,1]>0
 torch.testing.assert_close(errors[0,0,1],torch.tensor(expected))
 assert errors[0,0,0]==.4 and errors[0,0,-1]==0
 assert torch.isinf(errors[0,0,2:10]).all()


def test_history_90_percent_example():
 z=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
 labels=torch.zeros(3,dtype=torch.long);pop=torch.tensor([3.]);rep=torch.tensor([0]);pi=torch.tensor([[.1]]);h=torch.tensor([[[1,-1,-1,-1]]])
 a=history_prior(pi,labels,pop,torch.tensor([[1]]),torch.tensor([[.09]]),torch.ones(1,1),labels)
 errors,_=cosine_states(pi,z,rep,labels,pop,h,torch.ones(1,1,dtype=torch.bool),a)
 torch.testing.assert_close(errors[0,0,2],a[0,0]+a[0,2])
 torch.testing.assert_close(errors[0,0,6],2*a[0,2])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')
def test_cuda_all_losses_transfer_states_match_explicit_cpu():
 cpu=list(fixture());h=cpu[5]
 # The real top_history result excludes a sentinel slot at each group-row end.
 storage=torch.full((h.shape[0],h.shape[1]*h.shape[2]+1),-1,dtype=h.dtype)
 strided=storage[:,:-1].reshape_as(h);strided.copy_(h);assert not strided.is_contiguous()
 cpu[5]=strided;cpu=tuple(cpu)
 gpu=list(x.cuda() for x in cpu)
 h=gpu[5];storage=torch.full((h.shape[0],h.shape[1]*h.shape[2]+1),-1,dtype=h.dtype,device=h.device)
 strided=storage[:,:-1].reshape_as(h);strided.copy_(h);assert not strided.is_contiguous();gpu[5]=strided
 gpu=tuple(gpu);z,l,p,r,pi,h,a=cpu
 expected,cost=cosine_states(pi,z,r,l,p,h,torch.ones_like(pi,dtype=torch.bool),a)
 zg,lg,pg,rg,pig,hg,ag=gpu
 actual,gcost=cosine_states(pig,zg,rg,lg,pg,hg,torch.ones_like(pig,dtype=torch.bool),ag)
 torch.testing.assert_close(actual.cpu(),expected,atol=2e-6,rtol=2e-5);torch.testing.assert_close(gcost.cpu(),cost)
 for state in range(11):
  selected=torch.where(torch.isfinite(expected[:,:,state]),torch.full_like(pi,state,dtype=torch.long),torch.zeros_like(pi,dtype=torch.long))
  ids,_,_,_=pack_support(selected,h,r,l,p,11)
  w,mass=transfer_weights(selected,z,r,l,h,a,ids)
  gw,gmass=transfer_weights(selected.cuda(),zg,rg,lg,hg,ag,ids.cuda())
  torch.testing.assert_close(gw.cpu(),w,atol=2e-5,rtol=2e-5);torch.testing.assert_close(gmass.cpu(),mass,atol=1e-6,rtol=2e-5)
  transferred=(a.gather(1,ids.clamp_min(0))*w).sum(1)
  torch.testing.assert_close(transferred,(pi*(selected!=0)).sum(1),atol=1e-6,rtol=1e-5)
  assert (w[ids>=0]>=1).all() and torch.isfinite(w).all()


def test_mass_transfer_is_executed_by_weighted_softmax_under_prior_logits():
 z,l,p,r,pi,h,a=fixture();selected=torch.tensor([[2,1,7],[3,6,0]])
 ids,_,_,real=pack_support(selected,h,r,l,p,11)
 weights,mass=transfer_weights(selected,z,r,l,h,a,ids)
 # Shared K basis allows each query group to have exactly log-prior logits.
 q=a.clamp_min(1e-30).log().repeat_interleave(32,0)[None]
 k=torch.eye(11)[None];v=torch.randn(1,11,11)
 out,observed=reference_attention(q,k,v,ids,weights.log(),torch.empty(0,dtype=torch.long),1.)
 expected=mass.gather(1,ids.clamp_min(0)).masked_fill(ids<0,0)
 expected/=expected.sum(1,keepdim=True)
 torch.testing.assert_close(observed,expected,atol=1e-6,rtol=1e-5)
 assert ((observed/weights.clamp_min(1))<=observed+1e-7).all()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all24_layers_budget_and_unit_history_evidence(device):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
 torch.set_num_threads(1);torch.manual_seed(5)
 cfg=ReuseExploreConfig(state_loss='cosine_distortion',capture=True,source_width=4,window_groups=2)
 s=ReuseExploreState(cfg);s.reset(1,8,16,device)
 for layer in range(24):
  o=Attention(dim=32,num_heads=2).to(device).eval();wrapper=ReuseResidualExploration(o,s,layer).eval()
  with torch.inference_mode(),torch.autocast(device,enabled=device=='cuda',dtype=torch.bfloat16):out=wrapper(torch.randn(1,133,32,device=device))
  assert torch.isfinite(out).all()
  for wi,window in enumerate(s.captured[layer]['windows']):
   ids,pops,real,mass=(window[k] for k in ('ids','population','genuine','mass'))
   for row in range(len(ids)):
    valid=ids[row]>=0;assert len(ids[row,valid].unique())==valid.sum() and valid.sum()<=s.budget
   start=wi*cfg.window_groups;stop=start+len(ids)
   expected=(mass/pops.clamp_min(1)*real).half()
   torch.testing.assert_close(s.history_weights[start:stop],expected,atol=1e-4,rtol=.01)
