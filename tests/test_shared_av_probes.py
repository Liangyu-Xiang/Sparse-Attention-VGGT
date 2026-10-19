import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_streamed_routing_keeps_per_head_softmax_then_mean():
    from saf3r.triton.shared_av_routing import routing_probabilities
    torch.manual_seed(5);h,g,c=3,5,73
    scores=torch.randn(h,g*4,c,device='cuda',dtype=torch.bfloat16)
    population=torch.rand(c,device='cuda').add(.1).log()
    expected=(scores.float()*.3+population).softmax(-1).reshape(h,g,4,c).mean((0,2))
    actual=routing_probabilities(scores,population,g,.3)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_tensorcore_terminal_matches_exact_local_probability_vote():
    from saf3r.triton.shared_av_terminal import terminal_keys_tensorcore
    torch.manual_seed(17)
    h,r,d=16,45,64
    q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(h,300,d,device='cuda',dtype=torch.bfloat16)
    patch_keys=torch.arange(5,245,device='cuda')
    members=torch.arange(240,device='cuda').reshape(40,6)
    selected=torch.stack((torch.arange(15,device='cuda'),torch.arange(20,35,device='cuda')))
    missing=torch.tensor([13,9],device='cuda')
    actual=terminal_keys_tensorcore(q,k,patch_keys,selected,members,missing,d**-.5)
    for row in range(2):
        qr=(row*32+torch.tensor([4,12,20,28],device='cuda')).clamp_max(r-1)
        for column in range(15):
            if column>=missing[row]:assert actual[row,column]==-1;continue
            ids=members[selected[row,column]]
            z=torch.einsum('hpd,hmd->hpm',q[:,qr].float(),k[:,patch_keys[ids]].float())*d**-.5
            expected=ids[z.softmax(-1).mean((0,1)).argmax()]
            assert actual[row,column]==expected
