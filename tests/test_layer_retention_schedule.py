"""Budget invariants for schedules that change both up and down across layers."""
from types import SimpleNamespace
import pytest
from saf3r.patch.layer_retention_schedule import (
    PROFILES, layer_budget, patch_budget, profile_summary, validate_schedule,
    output_patch_budget,
)


@pytest.mark.parametrize('n,special,dense',[(1003,5,0),(1003,5,1),(1003,5,5),(133248,640,640),(35340,190,190)])
def test_final_output_budget_counts_corrected_queries_and_special_keys(n,special,dense):
    rates=[.1*5/7]*6+[.18*5/7]*12+[.1*5/7]*6
    total=0
    for r in rates:
        b=output_patch_budget(n,special,dense,r)
        cap=(n-dense)*(b+special)+dense*n
        assert cap<=r*n*n+1e-5
        assert r*n*n-cap<n-dense+1e-5
        total+=cap
    assert total<=.1*24*n*n+1e-4
    if dense==0:assert output_patch_budget(n,special,0,.1)==patch_budget(n,special,.1)


def test_final_output_budget_rejects_infeasible_dense_floor():
    with pytest.raises(ValueError):output_patch_budget(100,10,10,.1)


def test_midlayer_increase_does_not_change_cluster_initialization_budget():
    cfg=SimpleNamespace(retention=.2,layer_retentions=PROFILES['middle28'])
    n,special=133248,640
    assert patch_budget(n,special,cfg.retention)==26009
    budgets=[layer_budget(cfg,l,n,special) for l in range(24)]
    assert budgets[0]==budgets[23]<26009<budgets[6]==budgets[17]
    assert sum(budgets)+24*special<=int(.2*n*24)
    assert .2-1/n<profile_summary(cfg.layer_retentions,n,special)['mean_total_key_density_cap']<=.2


def test_uniform_schedule_preserves_original_budget():
    cfg=SimpleNamespace(retention=.2,layer_retentions=None)
    assert all(layer_budget(cfg,l,1003,5)==195 for l in range(24))


@pytest.mark.parametrize('layers',[[.2]*23,[.2]*25,[float('nan')]*24,[.01]*24,[.6]*24])
def test_invalid_schedules_rejected(layers):
    with pytest.raises(ValueError):validate_schedule(.2,layers)


def test_special_keys_are_included_in_budget_cap():
    assert patch_budget(1000,25,.2)==175
    with pytest.raises(ValueError):patch_budget(100,20,.2)


def test_cache_resize_preserves_previous_layer_and_contiguous_group_strides():
    import torch
    from saf3r.patch.layer_retention_schedule import begin_history_layer,finish_history_layer
    old=torch.arange(3*11,dtype=torch.float16).reshape(3,1,1,11)
    state=SimpleNamespace(ready_edges=old,budget=19)
    begin_history_layer(state,old)
    assert state.ready_edges is old
    assert state.ready_next_edges.is_contiguous()
    assert state.ready_next_edges.stride(0)==19
    state.ready_next_edges.fill_(7)
    assert torch.equal(old.flatten(),torch.arange(33,dtype=torch.float16))
    finish_history_layer(state)
    assert state.ready_edges.shape==(3,1,1,19) and state.ready_next_edges is None
    assert bool((state.ready_edges==7).all())
    state.budget=5
    begin_history_layer(state,old)
    assert state.ready_edges.shape[-1]==19
    finish_history_layer(state)
    assert state.ready_edges.stride(0)==5


def test_equal_width_cache_reuses_consumed_window_storage():
    import torch
    from saf3r.patch.layer_retention_schedule import begin_history_layer,finish_history_layer
    state=SimpleNamespace(ready_edges=torch.ones(3,1,1,11),budget=11)
    original=state.ready_edges
    begin_history_layer(state,original)
    assert state.ready_next_edges is original
    finish_history_layer(state)
    assert state.ready_edges is original
