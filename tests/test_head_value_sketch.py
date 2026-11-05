import torch
from saf3r.patch.head_value_sketch import projection,features

def test_full_orthogonal_sketch_preserves_head_concatenated_distances():
    torch.manual_seed(456)
    v=torch.randn(4,17,8);p=projection(4,8,8,'cpu');actual=features(v,p)
    expected=v.permute(1,0,2).reshape(17,-1)/2
    torch.testing.assert_close(torch.cdist(actual,actual),torch.cdist(expected,expected),atol=.002,rtol=1e-5)
    torch.testing.assert_close(actual.square().sum(-1),expected.square().sum(-1))

def test_separate_head_sketch_cannot_cancel_opposite_heads():
    v=torch.stack((torch.ones(3,8),-torch.ones(3,8)))
    assert v.mean(0).count_nonzero()==0
    assert features(v,projection(2,8,4,'cpu')).square().sum()>0

def test_projection_is_fixed_and_does_not_change_global_rng():
    torch.manual_seed(457);before=torch.random.get_rng_state()
    a=projection(4,64,8,'cpu');b=projection(4,64,8,'cpu')
    assert torch.equal(a,b) and torch.equal(before,torch.random.get_rng_state())
