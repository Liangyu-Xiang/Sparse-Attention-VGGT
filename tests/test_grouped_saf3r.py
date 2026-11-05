import copy
from types import SimpleNamespace
import pytest,torch
from saf3r.patch.grouped_saf3r import install_grouped_saf3r

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('shift',[False,True])
def test_grouped_patterns_preserve_every_head_and_unique_geometry(shift):
    from saf3r.patch.patch_vggt import HeadwiseSparseAttentionWrapper
    from saf3r.models.official_vggt.layers.attention import Attention
    cfg=[dict(mode='all_to_first',head_ids=list(range(6))),
         dict(mode='stride',stride=4,shift=shift,include_as=True,head_ids=list(range(6,11))),
         dict(mode='global_to_frame',head_ids=[11,12,13]),
         dict(mode='q_probe_topk',stride=4,head_ids=[14,15])]
    module=HeadwiseSparseAttentionWrapper(Attention(dim=256,num_heads=16),config=copy.deepcopy(cfg),lazy_dino_topk=False)
    module.patch_height=8;module.patch_width=8
    model=SimpleNamespace(aggregator=SimpleNamespace(global_blocks=[SimpleNamespace(attn=module)]))
    reference=module.compute_multihead_attention;empty_cache=torch.cuda.empty_cache
    assert getattr(model.aggregator,'release_cuda_cache_each_block',True)
    install_grouped_saf3r(model)
    assert model.aggregator.release_cuda_cache_each_block is False
    assert torch.cuda.empty_cache is empty_cache
    torch.manual_seed(715)
    for frames in [4,5,4]:
        n=frames*69;q,k,v=[torch.randn(1,16,n,16,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
        expected=reference(q,k,v);actual=module.compute_multihead_attention(q,k,v)
        torch.testing.assert_close(actual,expected,rtol=.003,atol=.003)
        for key,bank in module.grouped_static_indices.items():
            length,per,stride,start,include,_=key
            ids=set(range(start,length,stride))
            if include:ids|=set(range(per))|{f*per+i for f in range(length//per) for i in range(5)}
            assert bank.cpu().tolist()==sorted(ids)
        assert module.grouped_execution_record['heads']==16
