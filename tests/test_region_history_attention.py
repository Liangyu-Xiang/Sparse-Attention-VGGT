import numpy as np
import pytest
import torch
from scipy import ndimage
from saf3r.patch.region_history_attention import (
    spatial_search_layout,select_regions,pooled_group_probabilities,
    table_from_labels,transfer_history,concentration,RegionHistoryState,RegionHistoryAttention)


def test_spatial_fine_cells_are_frame_local_connected():
    labels,reps,parent,child=spatial_search_layout(2,28,37,'cpu')
    images=labels.numpy().reshape(2,28,37)
    assert len(reps)==504
    for label in np.unique(images):
        masks=images==label
        assert sum(plane.any() for plane in masks)==1
        assert sum(ndimage.label(plane)[1] for plane in masks)==1
    assert all(len(torch.unique(parent[child==c]))==1 for c in child.unique())


@pytest.mark.parametrize('variant',['A','B'])
def test_multiple_history_edges_in_one_fine_cell_survive_and_exclude_exploration(variant):
    layout=spatial_search_layout(1,4,6,'cpu')
    labels=layout[0]
    assert len(labels[torch.tensor([0,1,6])].unique())==1
    ids,record=select_regions(torch.ones(1,1,2),torch.randn(1,24,2),layout,
        torch.tensor([[.9,.1]]),torch.zeros(24,dtype=torch.long),torch.tensor([[0,1,6]]),4,variant)
    assert set([0,1,6]).issubset(ids[0].tolist())
    new=set(ids[0].tolist())-{0,1,6}
    assert all(labels[j]!=labels[0] for j in new)
    assert record['history_retained'].item()==3 and record['explored'].item()==1


def test_history_overflow_expands_budget_without_dropping_edges():
    layout=spatial_search_layout(1,4,6,'cpu')
    ids,record=select_regions(torch.ones(1,1,2),torch.randn(1,24,2),layout,
        torch.tensor([[.9,.1]]),torch.zeros(24,dtype=torch.long),torch.tensor([[0,1,6]]),2)
    assert ids.tolist()==[[0,1,6]]
    assert record['intermediate_probes'].item()==0


def test_pool_keys_before_group_attention_and_remove_history_covered_cells():
    q=torch.tensor([[[1.]],[[2.]]]);k=torch.tensor([[[1.],[3.],[9.],[11.]],[[2.],[6.],[4.],[8.]]])
    table=table_from_labels(torch.tensor([0,0,1,1]))
    p,count=pooled_group_probabilities(q,k,table,torch.tensor([[True,False,True,True]]))
    expected=torch.stack([torch.tensor([1.,10.]).softmax(0),torch.tensor([4.,12.]).softmax(0)]).mean(0)
    torch.testing.assert_close(p[0],expected)
    assert count.tolist()==[[1,2]]


def test_concentration_uses_global_coarse_mass_not_sparse_entropy():
    p=torch.tensor([[.25]*4,[.9,.04,.03,.03]])
    torch.testing.assert_close(concentration(p),torch.tensor([0.,.75]))
    ids=torch.tensor([[3,1,5,7],[3,1,5,7]])
    history=transfer_history(torch.full((2,4),.25),ids,p,4,.5,adaptive=True)
    assert (history[0]>=0).sum()==0 and (history[1]>=0).sum()==3


def test_dense_layers_match_unmodified_module():
    from saf3r.models.official_vggt.layers.attention import Attention
    state=RegionHistoryState()
    state.reset(1,4,6,'cpu')
    original=Attention(dim=8,num_heads=2).eval()
    wrapped=RegionHistoryAttention(original,state,0).eval()
    x=torch.randn(1,29,8)
    with torch.inference_mode():
        torch.testing.assert_close(wrapped(x),original(x),atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('variant',['A','B'])
@pytest.mark.parametrize('retention',[.1,.2])
def test_all24_layer_modes_dense9_handoff_and_exact_unique_budgets(variant,retention):
    from saf3r.models.official_vggt.layers.attention import Attention
    state=RegionHistoryState(variant=variant,retention=retention,capture=True,query_chunk=32)
    state.reset(1,8,16,'cpu')
    for layer in range(24):
        original=Attention(dim=8,num_heads=2).eval()
        x=torch.randn(1,133,8)
        with torch.inference_mode():
            output=RegionHistoryAttention(original,state,layer).eval()(x)
            assert torch.isfinite(output).all()
            if not 10<=layer<=16:
                torch.testing.assert_close(output,original(x),atol=1e-6,rtol=1e-5)
        if layer==9:
            assert state.history is not None and state.history.shape[0]==133
    assert set(state.captured)==set(range(10,17))
    for data in state.captured.values():
        ids=data['indices']
        assert ids.shape==(133,int(retention*133))
        assert all(len(row.unique())==len(row) for row in ids)
        assert bool((ids[:,-5:]==torch.arange(5)).all())
