import copy
import pytest
import torch
from saf3r.patch.hybrid_value_topk import layer_mode,exact_topk_attention,install_hybrid_value_topk


def test_schedule_zero_based():
    modes=[layer_mode(i) for i in range(24)]
    assert modes[:3]==['dense']*3
    assert modes[3:10]==['value_representatives']*7
    assert modes[10:17]==['attention_topk']*7
    assert modes[17:]==['value_representatives']*7


@pytest.mark.parametrize('chunk',[1,4,32])
def test_exact_qk_topk_matches_dense_mask_reference(chunk):
    torch.manual_seed(24)
    q=torch.randn(2,11,8);k=torch.randn(2,19,8);v=torch.randn_like(k)
    specials=torch.tensor([0,9,18]);patches=torch.tensor([i for i in range(19) if i not in specials])
    logits=q@k.transpose(1,2)*.3
    ix=patches[logits[:,:,patches].topk(5,-1).indices]
    mask=torch.zeros_like(logits,dtype=torch.bool).scatter_(-1,ix,True)
    mask[:,:,specials]=True
    expected=logits.masked_fill(~mask,-float('inf')).softmax(-1)@v
    actual=exact_topk_attention(q,k,v,patches,specials,8,.3,chunk)
    torch.testing.assert_close(actual,expected)


def test_all_layers_full_retention_recovers_dense_and_resets():
    from test_layer_value_attention import ToyAggregator
    model=torch.nn.Module();model.aggregator=ToyAggregator().eval()
    reference=copy.deepcopy(model)
    state=install_hybrid_value_topk(model,retention=1.)
    with torch.inference_mode():
        for frames in [2,1]:
            images=torch.randn(1,frames,3,28,28)
            torch.testing.assert_close(model.aggregator(images),reference.aggregator(images))
            assert len(state.records)==24
            assert all(r['value_updated']==(i>=2) for i,r in enumerate(state.records))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_existing_logits_indexed_av_no_repeated_qk(dtype):
    from saf3r.triton.indexed_weighted_value import indexed_weighted_value
    torch.manual_seed(11)
    v=torch.randn(3,217,64,device='cuda',dtype=dtype)
    ix=torch.randint(0,217,(3,13,173),device='cuda')
    p=torch.randn(3,13,173,device='cuda').softmax(-1)
    actual=indexed_weighted_value(p,ix,v)
    expected=torch.stack([(p[h,:,:,None]*v[h,ix[h]].float()).sum(1) for h in range(3)]).to(dtype)
    torch.testing.assert_close(actual,expected,atol=.002,rtol=.02)
