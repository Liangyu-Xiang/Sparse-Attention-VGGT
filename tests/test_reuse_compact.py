import pytest
import torch
from saf3r.patch.reuse_cosine_distortion import cosine_states,transfer_weights
from saf3r.patch.reuse_residual_exploration import top_history,budget_states,pack_support
from tests.test_reuse_cosine_distortion import fixture
from saf3r.triton import reuse_compact as compact

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')


def test_compact_geometry_losses_all_states_transfer_and_ordered_packing():
 z,labels,pop,reps,pi,hids,prior=fixture('cuda');geo=compact.prepare_geometry(z,labels,reps,pop)
 expected,costs=cosine_states(pi,z,reps,labels,pop,hids,torch.ones_like(pi,dtype=torch.bool),prior,1.)
 actual=compact.losses(labels,hids,prior,geo,len(reps))
 valid=torch.isfinite(expected);valid[:,:,0]=False;valid[:,:,-1]=False
 torch.testing.assert_close(actual[valid],expected[valid],atol=1e-7,rtol=2e-5)
 for state in range(11):
  selected=torch.where(torch.isfinite(expected[:,:,state]),torch.full_like(pi,state,dtype=torch.long),torch.zeros_like(pi,dtype=torch.long))
  ids,pops,real,record=pack_support(selected,hids,reps,labels,pop,11)
  packed,observed,fastrecord=compact.pack(selected,hids,reps,labels,pop,11)
  assert torch.equal(ids,packed.long())
  for key in record:torch.testing.assert_close(fastrecord[key].float(),record[key].float())
  reference,_=transfer_weights(selected,z,reps,labels,hids,prior,ids)
  actual=compact.transfer(selected,labels,reps,hids,prior,pi,packed,geo)
  torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)


def test_grouped_history_matches_stable_two_sort_including_equal_weights():
 z,labels,pop,reps,pi,hids,prior=fixture('cuda');geo=compact.prepare_geometry(z,labels,reps,pop)
 old=torch.tensor([[6,5,4,3,2,1,0,10,9,8,7,-1],[1,2,3,5,6,8,9,10,0,4,7,-1]],device='cuda')
 for a in (prior,torch.ones_like(prior)):
  weights=a.gather(1,old.clamp_min(0)).masked_fill(old<0,0)
  expected,_,discard=top_history(old,weights,torch.ones((2,1),device='cuda'),torch.zeros(len(labels),device='cuda',dtype=torch.long),labels,reps,4)
  actual,ad=compact.top_history(old,a,reps,geo,4)
  assert torch.equal(expected,actual.long())
  torch.testing.assert_close(ad.float(),discard.float())


def test_fused_dual_solver_matches_all_iterations_and_ties():
 torch.manual_seed(315);g,f,s=3,431,11
 costs=torch.randint(0,8,(g,f,s),device='cuda').float();costs[:,:,0]=0
 errors=torch.rand((g,f,s),device='cuda');errors[:,:,0]+=.4
 errors[:,:,4]=errors[:,:,3];costs[:,:,4]=costs[:,:,3]
 errors[:,:,9]=torch.inf
 upper=errors[:,:,0].amax(1)+1e-12
 for cap in (0,200,431,900,2500):
  a,ac=budget_states(errors,costs,cap,lambda_upper=upper)
  b,bc=compact.budget(errors,costs,upper,cap,16)
  assert torch.equal(a,b)
  torch.testing.assert_close(ac,bc,atol=0,rtol=0)


def test_fused_state_finalization_and_diagnostics_match_reference():
 z,labels,pop,reps,pi,hids,prior=fixture('cuda');geo=compact.prepare_geometry(z,labels,reps,pop)
 allowed=torch.tensor([[True,False,True],[False,True,False]],device='cuda')
 expected,costs=cosine_states(pi,z,reps,labels,pop,hids,allowed,prior,1.)
 raw=compact.losses(labels,hids,prior,geo,len(reps))
 actual,ac,objective=compact.finalize(raw,pi,pop,hids,allowed,1.)
 torch.testing.assert_close(actual,expected,atol=1e-7,rtol=2e-5)
 torch.testing.assert_close(ac,costs,atol=0,rtol=0)
 torch.testing.assert_close(objective,actual-actual[:,:,1:2],atol=0,rtol=0)
 selected,_=compact.budget(objective,ac,actual[:,:,0].amax(1)+1e-12,6,16)
 old=torch.tensor([[1,2,5,8,-1],[1,2,3,5,6]],device='cuda')
 diag=compact.diagnostics(actual,selected,hids,old,labels,reps)
 gains=actual[:,:,1:2]-actual;finite=torch.isfinite(gains)
 torch.testing.assert_close(diag['positive_gain_sum'],gains.masked_fill(~finite,0).clamp_min(0).sum((1,2)),atol=1e-7,rtol=2e-5)
 torch.testing.assert_close(diag['gain_min'],gains.masked_fill(~finite,torch.inf).amin((1,2)),atol=0,rtol=0)
 torch.testing.assert_close(diag['observed_history'],(old>=0).sum(1).float(),atol=0,rtol=0)
 for state in range(11):
  torch.testing.assert_close(diag[f'state_{state}_clusters'],(selected==state).sum(1).float(),atol=0,rtol=0)
  valid=finite[:,:,state]
  expected_gain=gains[:,:,state].masked_fill(~valid,0).sum(1)/valid.sum(1).clamp_min(1)
  torch.testing.assert_close(diag[f'gain_{state}'],expected_gain,atol=1e-7,rtol=2e-5)
