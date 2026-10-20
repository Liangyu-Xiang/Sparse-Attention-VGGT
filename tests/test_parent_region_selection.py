import pytest
import torch
from saf3r.patch.parent_region_selection import parent_prefix


@pytest.mark.parametrize('k',[1,4,64,96,128,160])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')
def test_partial_topk_exact_prefix_with_ties_zeros_and_gpu_budget_fallback(k):
    from saf3r.triton.parent_partial_topk import partial_parent_prefix
    torch.manual_seed(9)
    scores=torch.randint(0,8,(6,197),device='cuda').float()/8
    available=torch.randint(0,12,(6,197),device='cuda',dtype=torch.long)
    missing=torch.tensor([0,1,40,350,650,900],device='cuda')
    expected,chosen,_=parent_prefix(scores,available,missing,0)
    actual,selected,stats=partial_parent_prefix(scores,available,missing,k)
    for row in range(len(scores)):
        assert torch.equal(actual[row][selected[row]],expected[row][chosen[row]])
    assert stats['partial_unfilled_groups']==0
    if k==1:assert stats['partial_fallback_groups']>0


def test_whole_parent_keeps_crossing_region_and_has_strict_population_bound():
    scores=torch.tensor([[.8,.6,.4],[.1,.3,.2]])
    population=torch.tensor([[70,50,20],[30,40,60]])
    target=torch.tensor([100,0])
    order,chosen,_=parent_prefix(scores,population,target)
    total=(population.gather(1,order)*chosen).sum(1)
    assert total.tolist()==[120,0]
    assert ((total-target)<population.max(1).values).all()
