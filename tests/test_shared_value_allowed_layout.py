"""Candidate eligibility must not depend on the mask's storage layout."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bounded',[False,True])
def test_exact_candidates_respect_transposed_mask(bounded):
    from scripts.validate_shared_value_loss import example
    from saf3r.patch.reuse_inference_ops import gate
    from saf3r.triton.reuse_drop_moment import prepare as bins
    from saf3r.triton import reuse_shared_value_states as shared
    z,labels,grid,reps,population,uniform,prior,old,pi,_,_=example('cuda')
    context=gate(prior,pi,population,old,labels,reps)
    geometry=shared.prepare_geometry(z,labels,reps,population)
    shared.prepare(z,labels,reps,grid,uniform.shape[1],geometry)
    geometry['drop_bins']=bins(population);geometry['bounded_bins']=bounded
    residual=shared.corrections(prior,old,uniform,labels,geometry,context)
    row=torch.arange(prior.shape[0],device=z.device)[:,None]
    col=torch.arange(len(reps),device=z.device)[None]
    allowed=(row*7+col*3)%11<3
    transposed=allowed.T.contiguous().T
    assert not transposed.is_contiguous()
    expected_errors,expected_costs=shared.states(uniform,allowed,population,geometry,context,residual)
    actual_errors,actual_costs=shared.states(uniform,transposed,population,geometry,context,residual)
    # Every forbidden Exact state must have infinite loss, including when the
    # logical mask comes from sparse.mm(...).T as in the production selector.
    assert torch.equal(torch.isfinite(actual_errors[:,:,3]),allowed)
    torch.testing.assert_close(actual_errors,expected_errors)
    torch.testing.assert_close(actual_costs,expected_costs)
