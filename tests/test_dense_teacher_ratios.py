import pytest
import torch
from torch import nn

from scripts.search_dense_teacher_ratios import DenseRatioTeacher,policies,rank_policies
from saf3r.patch.value_region_av import ValueRegionAVAttention,ValueRegionAVConfig,ValueRegionAVState


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads=2;self.head_dim=16;self.scale=16**-.5
        self.qkv=nn.Linear(32,96);self.q_norm=nn.Identity();self.k_norm=nn.Identity()
        self.rope=None;self.proj=nn.Identity();self.proj_drop=nn.Identity()


def test_search_grid_and_teacher_not_gt_ranking():
    candidates=policies()
    assert len(candidates)==75
    assert len({tuple(p['stages']) for p in candidates})==75
    assert all(len(p['attention_ratios'])==24 and p['attention_ratios'][0]==0 for p in candidates)
    assert any(p['stages']==[.2,.8,.2] for p in candidates)
    layers=[dict(train_nmse=[1.,2.],validation_nmse=[9.,.1],density=[.2,.2]) for _ in range(24)]
    ranked=rank_policies(candidates[:2],layers)
    assert ranked[0]['name']==candidates[0]['name']  # held-out/GT cannot pick the winner
    with pytest.raises(ValueError):rank_policies(candidates[:2],layers[:-1])


@pytest.mark.parametrize('alpha',[0.,.2,.8,1.])
def test_teacher_replay_matches_production_support_attention_and_history(alpha):
    torch.manual_seed(7)
    policy=dict(name='test',stages=[alpha]*3,attention_ratios=[0.]+[alpha]*23)
    teacher=DenseRatioTeacher([policy],query_groups=2,capture=True)
    teacher.reset(2,8,8,torch.device('cpu'))
    config=ValueRegionAVConfig(parent_only=True,history_fraction=.8,attention_ratios=policy['attention_ratios'])
    production=ValueRegionAVState(config);production.reset(2,8,8,torch.device('cpu'))
    original=Attention().eval()
    for layer in range(3):
        # Both paths receive exactly the same Dense-forced current-layer features.
        x=torch.randn(1,production.n,32)
        with torch.inference_mode():
            teacher.step(original,x,None,layer)
            output=ValueRegionAVAttention(original,production,layer).eval()(x)[0]
        sampled=output[teacher.query_ids]
        actual=teacher.last_outputs[0].transpose(0,1).reshape(-1,32)
        torch.testing.assert_close(actual,sampled,rtol=1e-5,atol=2e-6)
        if production.history is None:assert teacher.histories[0] is None
        else:torch.testing.assert_close(teacher.histories[0],production.history[teacher.group_ids])
        torch.testing.assert_close(teacher.state.value.labels,production.value.labels)


def test_probe_group_constraints():
    with pytest.raises(ValueError):DenseRatioTeacher(query_groups=1)
    with pytest.raises(ValueError):DenseRatioTeacher(query_groups=3)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for production Tensor Core parity')
def test_gpu_teacher_replay_matches_production():
    torch.manual_seed(11)
    policy=dict(name='test',stages=[.2,.8,.2],attention_ratios=[0.]+[.2]*9+[.8]*7+[.2]*7)
    teacher=DenseRatioTeacher([policy],query_groups=2,capture=True)
    teacher.reset(2,8,8,torch.device('cuda'))
    config=ValueRegionAVConfig(parent_only=True,history_fraction=.8,attention_ratios=policy['attention_ratios'])
    production=ValueRegionAVState(config);production.reset(2,8,8,torch.device('cuda'))
    original=Attention().cuda().eval()
    for layer in range(3):
        x=torch.randn(1,production.n,32,device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            teacher.step(original,x,None,layer)
            output=ValueRegionAVAttention(original,production,layer).eval()(x)[0]
        actual=teacher.last_outputs[0].transpose(0,1).reshape(-1,32)
        torch.testing.assert_close(actual,output[teacher.query_ids],rtol=.02,atol=.02)
        if production.history is not None:
            torch.testing.assert_close(teacher.histories[0],production.history[teacher.group_ids])


def test_dense_pre_hook_preserves_keyword_rope_positions():
    class RequirePosition(nn.Module):
        def forward(self,q,pos):
            assert pos is not None
            return q
    original=Attention().eval();original.rope=RequirePosition()
    original.forward=lambda x,pos=None:x
    model=nn.Module();model.aggregator=nn.Module()
    block=nn.Module();block.attn=original
    model.aggregator.global_blocks=nn.ModuleList([block])
    policy=dict(name='test',stages=[.2,.8,.2],attention_ratios=[0.]+[.2]*9+[.8]*7+[.2]*7)
    teacher=DenseRatioTeacher([policy],query_groups=2)
    teacher.attach(model,torch.zeros(2,3,112,112))
    try:
        with torch.inference_mode():original(torch.randn(1,138,32),pos=torch.ones(1,138,2))
        assert len(teacher.layers)==1
    finally:teacher.detach()
