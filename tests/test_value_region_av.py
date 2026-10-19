import math
import pytest
import torch
from types import SimpleNamespace
from saf3r.patch.value_region_av import (ValueRegionAVConfig,ValueRegionAVState,
    ValueRegionAVAttention,ranked_prefix,boundary_scores,expand_regions,ragged_history)
from saf3r.patch.fused_shared_av import reference_attention
from saf3r.patch.region_history_attention import prepare_qkv
from saf3r.models.official_vggt.layers.attention import Attention


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_prefix_stops_at_parent_or_child_and_only_refines_crossing_region():
    scores=torch.tensor([[.8,.2],[.2,.8],[.8,.2]])
    counts=torch.tensor([[4,6],[4,6],[4,6]])
    order,full,boundary,active,remaining=ranked_prefix(scores,counts,torch.tensor([4,7,0]))
    assert full.tolist()==[[True,False],[True,False],[False,False]]
    assert active.tolist()==[False,True,False]
    assert boundary[1]==0 and remaining.tolist()==[0,1,0]
    _,whole,_,_,_=ranked_prefix(scores[:1],counts[:1],torch.tensor([5]),True)
    assert whole.tolist()==[[True,True]]


def test_boundary_fanout_is_validated_before_inference():
    with pytest.raises(ValueError):ValueRegionAVConfig(region_child_width=0)
    with pytest.raises(ValueError):ValueRegionAVConfig(region_child_width=3)
    with pytest.raises(ValueError):ValueRegionAVConfig(source_width=64,region_child_width=2)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all24_whole_regions_history_budgets_unified_output_and_reset(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    torch.manual_seed(9)
    config=ValueRegionAVConfig(capture=True,source_width=4,region_child_width=2,window_groups=2)
    state=ValueRegionAVState(config);state.reset(1,8,16,device)
    for layer in range(24):
        original=Attention(dim=32,num_heads=2).to(device).eval()
        wrapper=ValueRegionAVAttention(original,state,layer).eval()
        x=torch.randn(1,133,32,device=device)
        incoming=None if state.history is None else state.history.clone()
        with torch.inference_mode(),torch.autocast(device,enabled=device=='cuda',dtype=torch.bfloat16):
            actual=wrapper(x)
            data=state.captured[layer];common=data['common'];att=data['attention']
            valid_counts=(att>=0).sum(1)
            mask=torch.zeros(133,133,device=device,dtype=torch.bool)
            for group,row in enumerate(att):
                full=torch.cat((common,row[row>=0].long()))
                assert len(full)==len(full.unique()) and full.min()>=0 and full.max()<133
                assert set(range(5)).issubset(full.tolist())
                assert len(full)>=math.floor(.2*133)
                if layer:
                    assert len(full)-math.floor(.2*133)<int(state.fine_sizes.max())
                if incoming is not None:
                    ids=incoming[group];ids=ids[ids>=0].long()
                    assert set(state.patches[ids].tolist()).issubset(full.tolist())
                mask[group*32:min(133,(group+1)*32),full]=True
            # Every selected parent/child/fine region contributes ALL its members.
            for window,(starts,lengths) in enumerate(data['regions']):
                for row in range(len(starts)):
                    group=window*config.window_groups+row
                    support=set(torch.cat((common,att[group][att[group]>=0].long())).tolist())
                    for start,length in zip(starts[row].tolist(),lengths[row].tolist()):
                        ids=data['member_bank'][start:start+length].long()
                        assert set(state.patches[ids].tolist()).issubset(support)
            q,k,v=prepare_qkv(original,x,None);q=q.to(v.dtype);k=k.to(v.dtype)
            with torch.autocast(device,enabled=False):
                z=(q.float()@k.float().transpose(-1,-2))*original.scale
                expected=(z.masked_fill(~mask[None],-torch.inf).softmax(-1)@v.float()).to(v.dtype)
            expected=original.proj_drop(original.proj(expected.transpose(0,1).reshape(1,133,32)))
        torch.testing.assert_close(actual.float(),expected.float(),rtol=.025 if device=='cuda' else 1e-5,atol=.012 if device=='cuda' else 1e-6)
        assert torch.isfinite(actual).all() and state.records[-1]['terminal_member_qk_per_head']==0
    assert len(state.records)==24 and state.records[0]['attention_edges_target']==0
    state.reset(1,8,16,device)
    assert state.history is None and state.previous_layer==-1 and not state.records


def test_dynamic_members_are_the_actual_value_clusters_not_spatial_grid():
    state=ValueRegionAVState(ValueRegionAVConfig(source_width=4,region_child_width=2))
    state.reset(2,8,8,'cpu');torch.manual_seed(7)
    for layer in range(3):
        features=torch.randn(128,8);state.value.update(features,layer)
        reps,_=state.value.representatives(features,state.kv[layer])
        state.refresh_hierarchy(torch.randn(2,state.n,8),reps)
        assert torch.isneginf(state.fine_log_population[state.fine_sizes==0]).all()
        assert torch.isneginf(state.child_log_population[state.child_sizes==0]).all()
        for cluster in range(state.value.base_count):
            start=int(state.fine_starts[cluster]);length=int(state.fine_sizes[cluster])
            members=state.member_bank[start:start+length].long()
            assert len(members)>0
            assert (state.value.labels[members]==cluster).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')
def test_boundary_tensorcore_matches_per_head_per_probe_softmax_and_is_query_specific():
    torch.manual_seed(1);q=torch.randn(16,77,64,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(16,20,64,device='cuda',dtype=torch.bfloat16)
    logpop=torch.rand(20,device='cuda');candidates=torch.tensor([[0,2,4,7],[0,2,4,7]],device='cuda')
    active=torch.tensor([True,True],device='cuda')
    got=boundary_scores(q,k,logpop,candidates,active,1,64**-.5)
    probes=(torch.arange(2,device='cuda')[:,None]+1)*32+torch.tensor([4,12,20,28],device='cuda')
    probes=probes.clamp_max(76)
    logits=torch.einsum('hgtd,hgcd->hgtc',q[:,probes].float(),k[:,candidates].float())/8
    expected=(logits+logpop[candidates][None,:,None]).softmax(-1).mean((0,2))
    torch.testing.assert_close(got,expected,atol=.001,rtol=.005)
    assert not torch.allclose(got[0],got[1])
    active[1]=False
    assert (boundary_scores(q,k,logpop,candidates,active,1,64**-.5)[1]==0).all()


def test_ragged_history_never_retains_padding_or_special_keys():
    q=torch.randn(2,45,8);k=torch.randn(2,220,8);v=torch.randn_like(k)
    common=torch.arange(70);att=torch.full((2,90),-1,dtype=torch.int32)
    att[0,:67]=torch.arange(80,147);att[1,:73]=torch.arange(90,163)
    _,e,lse=reference_attention(q,k,v,common,att,62,8**-.5)
    history=ragged_history(e,lse,common,att,torch.tensor([67,73]),62,100)
    for row in range(2):
        allowed=set(common[:62].tolist()+att[row][att[row]>=0].tolist())
        assert len(history[row].unique())==100 and set(history[row].tolist()).issubset(allowed)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')
def test_streamed_region_expansion_multiple_blocks_matches_cpu_without_token_selection():
    torch.manual_seed(5);bank=torch.randperm(8192).int();value=torch.zeros(8192,dtype=torch.bool)
    value[:100]=True
    history=bank[:1800].repeat(2,1).clone();history[:,20:25]=-1
    history=history.masked_fill(value[history.long().clamp_min(0)],-1)
    starts=torch.tensor([[0,0,1500],[1000,4000,3000]])
    lengths=torch.tensor([[1500,0,2500],[2000,0,1000]])
    cpu=SimpleNamespace(capacity=64,current_value_ids=torch.arange(100),member_bank=bank)
    gpu=SimpleNamespace(capacity=64,current_value_ids=torch.arange(100,device='cuda'),member_bank=bank.cuda())
    expected,counts=expand_regions(history,starts,lengths,cpu,value,4000)
    actual,gpu_counts=expand_regions(history.cuda(),starts.cuda(),lengths.cuda(),gpu,value.cuda(),4000)
    assert torch.equal(actual.cpu(),expected) and torch.equal(gpu_counts.cpu(),counts)


@pytest.mark.parametrize('mode',['value','attention','shrinking'])
def test_ablation_and_budget_shrink(mode):
    ratios=(0.,)*24 if mode=='value' else (0.,)+(1.,)*23 if mode=='attention' else ValueRegionAVConfig().attention_ratios
    retentions=(.2,)*10+(.15,)*7+(.1,)*7 if mode=='shrinking' else (.2,)*24
    config=ValueRegionAVConfig(attention_ratios=ratios,retentions=retentions,capture=True,source_width=4,region_child_width=2)
    state=ValueRegionAVState(config);state.reset(1,8,16,'cpu')
    for layer in range(24):
        original=Attention(dim=8,num_heads=2).eval()
        with torch.inference_mode():ValueRegionAVAttention(original,state,layer).eval()(torch.randn(1,133,8))
        data=state.captured[layer]
        for row in data['attention']:
            ids=torch.cat((data['common'],row[row>=0].long()))
            assert len(ids.unique())==len(ids) and len(ids)>=math.floor(retentions[layer]*133)
            if state.ka[layer]:assert len(ids)-math.floor(retentions[layer]*133)<int(state.fine_sizes.max())
