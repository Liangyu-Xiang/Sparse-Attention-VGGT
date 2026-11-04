import json
from pathlib import Path
import pytest
import torch


@pytest.mark.parametrize('bm',[32,64,128])
@pytest.mark.parametrize('rows',[519,768])
def test_split_launch_preserves_every_tail_row_and_history(bm,rows):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled as run
    torch.manual_seed(71);h,n,d=2,401,64;g=(rows+255)//256
    q=torch.randn(h,rows,d,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(h,n,d,device='cuda',dtype=q.dtype);v=torch.randn_like(k)
    ids=torch.stack([torch.randperm(n-5,device='cuda')[:73]+5 for _ in range(g)]).int()
    ids[-1,-7:]=-1;special=torch.arange(5,device='cuda')
    bias=torch.randn(g,73,device='cuda').masked_fill(ids<0,-torch.inf)
    kw=dict(logit_bias=bias,query_group_size=256,probe_group_stride=1,debiased_log=True,
            split_export=True,block_queries=bm,num_stages=1)
    baseline,be=run(q,k,v,ids,special,d**-.5,**kw)
    actual,ae=run(q,k,v,ids,special,d**-.5,execution_split=True,**kw)
    assert torch.equal(baseline,actual)
    for name in ['lse','debiased_log_special_probability','debiased_log_patch_probability']:
        assert torch.equal(be[name],ae[name]),name


def test_live_budget_increase_and_decrease_preserve_history_and_reset():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,ReuseExploreState,ReuseResidualExploration
    from saf3r.models.official_vggt.layers.attention import Attention
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/evaluation/reuse_shared_value_loss_q256.json').read_text())
    config.update(layer_retentions=[.20,.12,.28]+[.2]*21,sampled_split_launch=True,
                  history_split_export=True,stable_binned_graph=True,dense_special_queries=1,
                  graph_cache_capacity=8,attention_block_queries=64,attention_stages=1)
    c=ReuseExploreConfig(**config);state=ReuseExploreState(c);state.reset(1,8,32,'cuda')
    x=torch.randn(1,261,64,device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for layer,rate in enumerate([.20,.12,.28]):
            wrapped=ReuseResidualExploration(Attention(dim=64,num_heads=2).cuda().eval(),state,layer).eval()
            x=wrapped(x)
            assert torch.isfinite(x).all()
            assert state.budget==int(261*rate)-5
            assert state.history_ids.shape[-1]==state.ready_edges.shape[-1]==state.budget
            assert all(k==layer for k in state.ready_written.values())
            assert state.records[-1]['dense_special_query_count']==1
    state.reset(1,8,32,'cuda')
    assert state.history_ids is None and state.previous_layer==-1
