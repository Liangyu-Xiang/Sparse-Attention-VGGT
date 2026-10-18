from functools import partial
import pytest
import torch
from saf3r.patch.region_history_attention import (
    spatial_search_layout,table_from_labels,select_regions,RegionHistoryState,RegionHistoryAttention)
from saf3r.patch.region_history_corrections import (
    group_probabilities,select_regions_corrected,materialize_cells,influence_scores,
    install_corrected_region_history_attention)


def test_history_is_hard_preserved_even_outside_coarse_gate():
    layout=spatial_search_layout(1,4,6,'cpu')
    args=(torch.ones(1,1,2),torch.randn(1,24,2),layout,
        torch.tensor([[.9,.1]]),torch.ones(24,dtype=torch.long),torch.tensor([[0,1]]),4,'B')
    legacy=select_regions(*args)
    fixed=select_regions_corrected(*args,group_scoring='mean',terminal='representative')
    assert legacy[1]['history_retained'].item()==0
    assert fixed[1]['history_retained'].item()==2
    assert {0,1}.issubset(fixed[0][0].tolist())


def test_corrected_history_overflow_expands_budget_without_exploration():
    layout=spatial_search_layout(1,4,6,'cpu')
    ids,record=select_regions_corrected(torch.ones(1,1,2),torch.randn(1,24,2),layout,
        torch.tensor([[.9,.1]]),torch.ones(24,dtype=torch.long),torch.tensor([[0,1,6]]),2,'A')
    assert ids.tolist()==[[0,1,6]]
    assert record['explored'].item()==0 and record['terminal_member_qk'].item()==0


def test_population_bias_uses_real_tokens_not_number_of_fine_cells():
    q=torch.zeros(1,1,1);keys=torch.zeros(1,2,1)
    labels=torch.tensor([0,1,1,1]);parent=torch.tensor([0,1]);table=table_from_labels(parent)
    eligible=torch.ones(1,2,dtype=torch.bool)
    mean,_=group_probabilities(q,keys,table,eligible,labels,parent,'mean')
    size,_=group_probabilities(q,keys,table,eligible,labels,parent,'size_bias')
    exact,_=group_probabilities(q,keys,table,eligible,labels,parent,'exact_mass',torch.zeros(1,1,4))
    torch.testing.assert_close(mean,torch.tensor([[.5,.5]]))
    torch.testing.assert_close(size,torch.tensor([[.25,.75]]))
    torch.testing.assert_close(size,exact)


def test_exact_mass_preserves_variance_information_and_excludes_covered_cells():
    q=torch.ones(2,2,1);keys=torch.zeros(2,2,1)
    labels=torch.tensor([0,0,1,1]);parent=torch.tensor([0,1]);table=table_from_labels(parent)
    logits=torch.tensor([[[-4.,4.,0.,0.],[-4.,4.,0.,0.]],
                         [[1.,1.,2.,2.],[1.,1.,2.,2.]]])
    eligible=torch.tensor([[True,True],[False,True]])
    exact,count=group_probabilities(q,keys,table,eligible,labels,parent,'exact_mass',logits)
    expected=torch.stack([torch.stack([torch.logsumexp(head[:2],0),torch.logsumexp(head[2:],0)]).softmax(0)
                          for head in logits[:,0]]).mean(0)
    torch.testing.assert_close(exact[0],expected)
    torch.testing.assert_close(exact[1],torch.tensor([0.,1.]))
    assert count.tolist()==[[1,1],[0,1]]
    with pytest.raises(ValueError):group_probabilities(q,keys,table,eligible,labels,parent,'exact_mass')


def test_query_dependent_local_terminal_without_dense_oracle():
    q=torch.tensor([[[1.],[-1.]]]);keys=torch.tensor([[[-2.],[3.],[1.],[0.]]])
    labels=torch.tensor([0,0,1,1]);reps=torch.tensor([0,2])
    selected=torch.tensor([[True,False],[True,False]])
    final,probes=materialize_cells(q,keys,labels,reps,selected,torch.zeros(2,4,dtype=torch.bool),'local_top1')
    assert final.nonzero().tolist()==[[0,1],[1,0]]
    assert probes.tolist()==[2,2]


def test_dense_best_member_is_explicit_and_keeps_region_fixed():
    labels=torch.tensor([0,0,1,1]);reps=torch.tensor([0,2]);selected=torch.tensor([[True,False]])
    args=(torch.ones(1,1,1),torch.zeros(1,4,1),labels,reps,selected,torch.zeros(1,4,dtype=torch.bool))
    with pytest.raises(ValueError):materialize_cells(*args,'best_member')
    final,_=materialize_cells(*args,'best_member',torch.tensor([[.1,.6,.2,.1]]))
    assert final.nonzero().tolist()==[[0,1]]
    assert selected.tolist()==[[True,False]]


def test_single_deletion_influence_matches_actual_renormalized_output_change():
    torch.manual_seed(19)
    a=torch.randn(2,3,5).softmax(-1);v=torch.randn(2,5,4);output=a@v
    scores=influence_scores(a,v,output)
    for j in range(5):
        without=(output-a[:,:,j,None]*v[:,None,j])/(1-a[:,:,j,None])
        expected=(without-output).norm(dim=-1).mean(0)
        torch.testing.assert_close(scores[:,j],expected,rtol=1e-5,atol=1e-6)


def test_legacy_selector_replays_exactly_when_requested():
    torch.manual_seed(51)
    layout=spatial_search_layout(1,8,16,'cpu')
    args=(torch.randn(2,5,4),torch.randn(2,128,4),layout,torch.rand(5,3).softmax(-1),
        torch.arange(128)%3,torch.tensor([[0,1],[2,3],[4,5],[6,7],[8,9]]),12,'B')
    old=select_regions(*args)
    new=select_regions_corrected(*args,history_policy='gate',group_scoring='mean',terminal='representative')
    assert torch.equal(old[0],new[0])


@pytest.mark.parametrize('retention',[.1,.2])
def test_corrected_all_layer_integration_dense_unchanged_budget_and_unique_keys(retention):
    from saf3r.models.official_vggt.layers.attention import Attention
    state=RegionHistoryState(variant='A',retention=retention,capture=True,query_chunk=32)
    state.selector=partial(select_regions_corrected,history_policy='hard',group_scoring='size_bias',terminal='local_top1')
    state.coarse_size_bias=True;state.reset(1,8,16,'cpu')
    for layer in range(24):
        original=Attention(dim=8,num_heads=2).eval();x=torch.randn(1,133,8)
        with torch.inference_mode():
            result=RegionHistoryAttention(original,state,layer).eval()(x)
            assert torch.isfinite(result).all()
            if not 10<=layer<=16:torch.testing.assert_close(result,original(x),atol=1e-6,rtol=1e-5)
    for data in state.captured.values():
        ids=data['indices'];assert ids.shape==(133,int(retention*133))
        assert all(len(row.unique())==len(row) for row in ids)
    assert all(record['terminal_member_qk']>0 for record in state.records if record['mode']=='sparse')


def test_deployable_factory_rejects_dense_oracle_terminal():
    with pytest.raises(ValueError):install_corrected_region_history_attention(None,terminal='best_member')


def test_capture_disabled_preserves_all_outputs_history_and_density():
    from saf3r.models.official_vggt.layers.attention import Attention
    states=[RegionHistoryState(variant='B',retention=.1,capture=capture,query_chunk=32)
            for capture in [False,True]]
    for state in states:
        state.selector=partial(select_regions_corrected,history_policy='hard',group_scoring='size_bias',terminal='local_top1')
        state.coarse_size_bias=True;state.reset(1,8,16,'cpu')
    for layer in range(24):
        original=Attention(dim=8,num_heads=2).eval();x=torch.randn(1,133,8)
        with torch.inference_mode():
            outputs=[RegionHistoryAttention(original,state,layer).eval()(x) for state in states]
        assert torch.equal(*outputs)
        if layer>=9:
            assert torch.equal(states[0].history,states[1].history)
            assert states[0].records==states[1].records
    assert not states[0].captured and len(states[1].captured)==7
