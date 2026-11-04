"""True shared-support groups, independent of the 32-row GPU execution tile."""
import math
from pathlib import Path
import json
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('group_size', [128,256])
@pytest.mark.parametrize('tail', [0,7])
@pytest.mark.parametrize('block_queries', [32,64,128])
def test_shared_support_output_and_actual_sampled_probability(group_size,tail,block_queries):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled
    torch.manual_seed(42)
    h,n,d=2,151,32
    r=group_size*2+tail
    g=math.ceil(r/group_size)
    q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    ids=torch.stack([torch.randperm(n-5,device='cuda')[:29]+5 for _ in range(g)]).int()
    ids[-1,-3:]=-1
    bias=torch.randn(g,29,device='cuda').masked_fill(ids<0,-torch.inf)
    specials=torch.arange(5,device='cuda')
    pos=(torch.arange(g,device='cuda')[:,None]*group_size+group_size//8).clamp_max(r-1).int()
    stride=256//group_size
    out,ev=direct_sparse_attention_sampled(q,k,v,ids,specials,d**-.5,bias,
        probe_positions=pos,probe_group_stride=stride,query_group_size=group_size,debiased_log=True,block_queries=block_queries,num_stages=1,split_export=block_queries!=32)
    expected=torch.empty_like(out);scores=[]
    for group in range(g):
        start=group*group_size;stop=min(r,start+group_size)
        bank=torch.cat((specials,ids[group].long()))
        raw=q[:,start:stop].float()@k[:,bank.clamp_min(0)].float().transpose(-1,-2)*d**-.5
        logbias=torch.cat((torch.zeros(5,device='cuda'),bias[group]))
        logits=(raw+logbias).masked_fill(bank[None,None]<0,-torch.inf)
        expected[:,start:stop]=(logits.softmax(-1)@v[:,bank.clamp_min(0)].float()).to(q.dtype)
        if group%stride==0:
            probe=int(pos[group,0])-start
            val=raw[:,probe]-logits[:,probe].logsumexp(-1)[:,None]
            scores.append((val.logsumexp(0)-math.log(h)).masked_fill(bank<0,-torch.inf))
    torch.testing.assert_close(out.float(),expected.float(),atol=.018,rtol=.03)
    got=torch.cat((ev['debiased_log_special_probability'],ev['debiased_log_patch_probability']),-1)[:,0]
    torch.testing.assert_close(got,torch.stack(scores),atol=.025,rtol=.006)
    assert out.shape[1]==r and len(scores)==math.ceil(g/stride)

@pytest.mark.parametrize('group_size',[128,256])
@pytest.mark.parametrize('efficient',[False,True])
def test_full24_state_group_boundaries_and_reset(group_size,efficient):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,ReuseExploreState,ReuseResidualExploration
    from saf3r.models.official_vggt.layers.attention import Attention
    cfg=json.loads((ROOT/'configs/evaluation/reuse_shared_value_loss_256.json').read_text())
    cfg.update(query_group_size=group_size,window_groups=8,source_width=8)
    if efficient:cfg.update(attention_block_queries=64,attention_stages=1,static_shared_graph=True,history_split_export=True)
    c=ReuseExploreConfig(**cfg);s=ReuseExploreState(c)
    s.reset(1,8,32,'cuda');n=261
    assert s.groups==math.ceil(n/group_size)
    assert (s.probe_positions[:,0]//group_size==torch.arange(s.groups,device='cuda')).all()
    torch.manual_seed(3)
    for layer in range(24):
        o=Attention(dim=64,num_heads=2).cuda().eval()
        x=torch.randn(1,n,64,device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            y=ReuseResidualExploration(o,s,layer).eval()(x)
        assert y.shape==x.shape and torch.isfinite(y).all()
        assert s.history_ids.shape==(s.groups,s.budget)
        for row in s.history_ids:
            valid=row[row>=0]
            assert len(valid.unique())==len(valid)
        assert s.records[-1]['query_group_size']==group_size
        assert s.records[-1]['query_groups']==s.groups
    s.reset(1,8,32,'cuda')
    assert s.history_ids is None and s.previous_layer==-1


def test_duplicate_support_diagnostic_guard():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from saf3r.triton.reuse_direct_sampled_probability import validate_support_indices
    ids=torch.tensor([[4,5,-1,-1]],device='cuda');specials=torch.arange(4,device='cuda')
    validate_support_indices(ids,specials,10)
    for invalid in ([[4,4,-1,-1]],[[0,5,-1,-1]],[[10,5,-1,-1]],[[-2,5,-1,-1]]):
        with pytest.raises(ValueError):validate_support_indices(torch.tensor(invalid,device='cuda'),specials,10)
