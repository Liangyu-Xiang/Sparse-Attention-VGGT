import pytest
import torch
from saf3r.patch.fused_shared_av import (FusedSharedAVConfig,FusedSharedAVState,
    FullCoverageValue,FusedSharedAVAttention,compact_prefix,history_from_mass)
from saf3r.models.official_vggt.layers.attention import Attention
from saf3r.patch.region_history_attention import prepare_qkv


def test_config_exact_schedule_and_validation():
    c=FusedSharedAVConfig()
    assert len(c.attention_ratios)==24 and c.attention_ratios[0]==0
    assert c.attention_ratios[1:10]==(.2,)*9 and c.attention_ratios[10:17]==(.8,)*7
    with pytest.raises(ValueError):FusedSharedAVConfig(attention_ratios=(.2,)*24)
    with pytest.raises(ValueError):FusedSharedAVConfig(retentions=(.21,)*24)
    with pytest.raises(ValueError):FusedSharedAVConfig(retentions=(.049,)*24)
    assert FusedSharedAVConfig(retentions=(.05,)*9+(.2,)*15).retentions[0]==.05
    assert FusedSharedAVConfig(history_fraction=.8).history_fraction==.8
    with pytest.raises(ValueError):FusedSharedAVConfig(history_fraction=1.01)
    with pytest.raises(ValueError):FusedSharedAVConfig(attention_ratios=(0.,))


def test_all_tokens_covered_by_real_nearest_representatives_after_merge():
    torch.manual_seed(2);v=FullCoverageValue(2,64,22,'cpu',width=4)
    for layer in range(3):
        x=torch.randn(128,8);v.update(x,layer)
        for count in [22,13,3]:
            reps,labels=v.representatives(x,count)
            assert labels.shape==(128,) and labels.min()==0 and labels.max()==count-1
            assert torch.equal(labels[reps],torch.arange(count))
            for group in range(count):
                members=torch.where(labels==group)[0];center=x[members].mean(0)
                minimum=(x[members]-center).square().sum(1).min()
                torch.testing.assert_close((x[reps[group]]-center).square().sum(),minimum)


def test_compact_prefix_keeps_history_and_fills_unique_tail():
    a=torch.tensor([[4,-1,8],[-1,5,-1]])
    b=torch.tensor([[3,6,-1],[7,8,9]])
    assert compact_prefix(a,b,4).tolist()==[[4,8,3,6],[5,7,8,9]]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all24_budgets_masks_history_and_unified_outputs(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    torch.manual_seed(7)
    c=FusedSharedAVConfig(capture=True,window_groups=2,source_width=4)
    s=FusedSharedAVState(c);s.reset(1,8,16,device)
    for layer in range(24):
        original=Attention(dim=32,num_heads=2).to(device).eval()
        wrapped=FusedSharedAVAttention(original,s,layer).eval()
        x=torch.randn(1,133,32,device=device)
        incoming=None if s.history is None else s.history.clone()
        context=torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else torch.autocast('cpu',enabled=False)
        with torch.inference_mode(),context:
            actual=wrapped(x)
            data=s.captured[layer]
            common,att=data['common'],data['attention']
            assert len(common)+att.shape[1]==int(.2*133)
            for group,ids in enumerate(att):
                full=torch.cat((common,ids.long()))
                assert full.min()>=0 and full.max()<133
                assert len(full.unique())==len(full)
                assert set(range(5)).issubset(full.tolist())
                if incoming is not None:
                    assert set(s.patches[incoming[group].long()].tolist()).issubset(full.tolist())
            q,k,v=prepare_qkv(original,x,None)
            z=(q.float()@k.float().transpose(-1,-2))*original.scale
            mask=torch.zeros(133,133,device=device,dtype=torch.bool)
            for row in range(133):mask[row,torch.cat((common,att[row//32].long()))]=True
            expected=(z.masked_fill(~mask[None],-torch.inf).softmax(-1)@v.float()).to(v.dtype)
            expected=original.proj_drop(original.proj(expected.transpose(0,1).reshape(1,133,32)))
        torch.testing.assert_close(actual.float(),expected.float(),rtol=.02 if device=='cuda' else 1e-5,atol=.01 if device=='cuda' else 1e-6)
        assert torch.isfinite(actual).all()
    assert s.records[0]['attention_edges']==0 and len(s.records)==24
    s.reset(1,8,16,device)
    assert s.history is None and s.previous_layer==-1 and not s.records


def test_history_mass_excludes_specials_and_handles_partial_tiles():
    from saf3r.patch.fused_shared_av import reference_attention
    torch.manual_seed(1)
    q=torch.randn(2,45,8);k=torch.randn(2,200,8);v=torch.randn_like(k)
    common=torch.arange(70);att=torch.arange(80,146).repeat(2,1).int()
    _,e,lse=reference_attention(q,k,v,common,att,62,8**-.5)
    history=history_from_mass(e,lse,common,att,62,100)
    for row in history:
        assert len(row.unique())==100
        assert not set(range(62,70)).intersection(row.tolist())


@pytest.mark.parametrize('mode',['value','attention','shrinking'])
def test_pure_branches_and_retention_shrink_have_no_budget_overflow(mode):
    ratios=(0.,)*24 if mode=='value' else (0.,)+(1.,)*23 if mode=='attention' else FusedSharedAVConfig().attention_ratios
    retentions=(.2,)*10+(.15,)*7+(.1,)*7 if mode=='shrinking' else (.2,)*24
    config=FusedSharedAVConfig(attention_ratios=ratios,retentions=retentions,capture=True,source_width=4)
    s=FusedSharedAVState(config);s.reset(1,8,16,'cpu')
    for layer in range(24):
        original=Attention(dim=8,num_heads=2).eval()
        with torch.inference_mode():FusedSharedAVAttention(original,s,layer).eval()(torch.randn(1,133,8))
        data=s.captured[layer]
        for group in data['attention']:
            ids=torch.cat((data['common'],group.long()))
            assert ids.min()>=0 and len(ids.unique())==int(retentions[layer]*133)
