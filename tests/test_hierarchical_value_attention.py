import pytest
import torch
import copy
from pathlib import Path

from saf3r.patch.hierarchical_value_attention import (
    allocate_budget, initial_hierarchy, lightweight_clusters, sibling_update,
    reliability, select_edges, sizes, sparse_reference,
    HierarchicalValueState, HierarchicalValueAttention,
    validate_history,
)
from saf3r.models.official_vggt.layers.attention import Attention


@pytest.mark.parametrize('budget', [0, 1, 4, 10, 16])
def test_capacity_constrained_largest_remainder(budget):
    p = torch.tensor([[.999, .0005, .0005], [0., 1., 0.], [.2, .3, .5]])
    cap = torch.tensor([1, 5, 10])
    b = allocate_budget(p, budget, cap)
    assert (b >= 0).all() and (b <= cap).all()
    assert (b.sum(-1) == budget).all()
    if budget == 4:
        # Unconstrained Hamilton on last row: [.8,1.2,2] -> [1,1,2].
        torch.testing.assert_close(b[-1], torch.tensor([1, 1, 2]))


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        allocate_budget(torch.ones(2, 3), 10, torch.tensor([1, 2, 3]))


def hierarchy(device='cpu'):
    torch.manual_seed(1)
    x = torch.randn(200, 8, device=device)
    z, r = initial_hierarchy(x, .3, 32, 2, 2)
    m = lightweight_clusters(x[r], 7, 2)
    return x, (z, r, m, sizes(z, len(r)))


def test_hierarchy_membership_anchors_and_original_coarse_constraint():
    x, (z, r, m, count) = hierarchy()
    assert len(r) == 60 and (count > 0).all()
    torch.testing.assert_close(z[r], torch.arange(len(r)))
    new_z, new_r = sibling_update(x+torch.randn_like(x)*.7, z, r, m, 2)
    torch.testing.assert_close(m[new_z], m[z])
    torch.testing.assert_close(new_z[r], torch.arange(len(r)))
    torch.testing.assert_close(new_z[new_r], torch.arange(len(r)))


@pytest.mark.parametrize('ka', [0, 1, 5, 20])
@pytest.mark.parametrize('use_history', [False, True])
def test_exact_unique_budget_exploration_and_no_duplicate_value_coverage(ka, use_history):
    _, h = hierarchy()
    p = torch.randn(12, 7).softmax(-1)
    history = None
    if use_history:
        # Deliberately several historical tokens in the same fine cluster.
        history = torch.stack([torch.randperm(200)[:27] for _ in range(12)])
    indices, origin = select_edges(p, 20, ka, h, history)
    assert (indices >= 0).all() and (indices < 200).all()
    assert (origin.sum(-1) == ka).all()
    z, reps, mapping, count = h
    quotas = allocate_budget(p, ka, sizes(mapping[z], 7))
    for row in range(len(p)):
        assert len(indices[row].unique()) == 20
        actual = sizes(mapping[z[indices[row, :ka]]], 7)
        torch.testing.assert_close(actual, quotas[row])
        covered = z[indices[row, :ka]]
        value = z[indices[row, ka:]]
        assert len(value.unique()) == len(value)
        assert not torch.isin(value, covered).any()


def test_exploration_exhausted_representatives_uses_unique_members_in_same_group():
    labels = torch.arange(20)//5
    reps = torch.tensor([0, 5, 10, 15])
    mapping = torch.tensor([0, 0, 1, 1])
    h = (labels, reps, mapping, sizes(labels, 4))
    p = torch.tensor([[1., 0.]])
    selected, origin = select_edges(p, 4, 4, h)
    assert (selected < 10).all() and len(selected[0].unique()) == 4
    assert set([0, 5]).issubset(set(selected[0].tolist()))


def test_reliability_uniform_and_query_independent_concentration():
    for p in (torch.ones(10, 4)/4, torch.tensor([[.97, .01, .01, .01]]).repeat(10, 1)):
        a, c, s = reliability(p)
        assert abs(float(a)) < 1e-6 and abs(float(s)) < 1e-6
    a, _, _ = reliability(torch.eye(4))
    assert float(a) == pytest.approx(1.)
    assert float(reliability(torch.ones(3, 1))[0]) == 0


def test_validation_discards_bad_attention_candidate_and_promotes_value_edge():
    indices = torch.tensor([[3, 7, 9, 11]])
    origin = torch.tensor([[True, True, False, False]])
    logits = torch.tensor([[[.1, 3., 5., .2]], [[.3, 2., 4., .4]]])
    history, promoted = validate_history(indices, origin, logits, 2)
    assert history.tolist() == [[9, 7]]
    assert int(promoted) == 1
    assert history.ndim == 2 and history.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_edge_selection_matches_cpu_with_history_and_fallback():
    _, h = hierarchy()
    p = torch.randn(9, 7).softmax(-1)
    history = torch.stack([torch.randperm(200)[:30] for _ in range(9)])
    expected, _ = select_edges(p, 50, 40, h, history)
    actual, _ = select_edges(p.cuda(), 50, 40, tuple(t.cuda() for t in h), history.cuda())
    torch.testing.assert_close(actual.cpu(), expected)


def test_shared_full_retention_recovers_dense_and_reset_drops_history():
    torch.manual_seed(2)
    original = Attention(dim=8, num_heads=2, qk_norm=True).eval()
    state = HierarchicalValueState(retention=1., fine_ratio=1., coarse_size=5, query_chunk=7)
    state.reset(2, 2, 2)
    wrapper = HierarchicalValueAttention(original, state, 2).eval()
    x = torch.randn(1, 18, 8)
    with torch.no_grad():
        expected = original(x)
        actual = wrapper(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert state.records[0]['connection_retention'] == 1
    assert state.history.ndim == 2  # no head dimension
    state.reset(2, 2, 2)
    assert state.history is None and state.hierarchy is None


def test_zero_attention_budget_uses_identical_shared_value_support(monkeypatch):
    import saf3r.patch.hierarchical_value_attention as module
    monkeypatch.setattr(module, 'reliability', lambda p: (p.new_zeros(()),)*3)
    torch.manual_seed(4)
    original = Attention(dim=8, num_heads=2).eval()
    state = HierarchicalValueState(retention=.5, fine_ratio=.7, coarse_size=5, initialization='lightweight')
    state.reset(2, 3, 3)
    wrapper = HierarchicalValueAttention(original, state, 2).eval()
    x = torch.randn(1, 28, 8)
    with torch.no_grad():
        actual = wrapper(x)
        q, k, v = original.qkv(x).reshape(1, 28, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
        patches, specials = module.token_layout(2, 9, 5, x.device)
        z, reps, mapping, count = state.hierarchy
        selected = torch.cat((patches[reps[count.argsort(descending=True, stable=True)[:4]]], specials))
        expected = torch.nn.functional.scaled_dot_product_attention(q, k[:, :, selected], v[:, :, selected])
        expected = original.proj(expected.transpose(1, 2).reshape(1, 28, 8))
    torch.testing.assert_close(actual, expected)
    assert state.history.shape == (28, 0)
    assert state.records[0]['total_keys'] == 14
    assert state.records[0]['execution'] == 'shared_sdpa_zero_attention_budget'


def test_framework_config_and_only_requested_layers_and_sequence_reset():
    from addict import Dict
    from hydra import initialize_config_dir, compose
    from saf3r.patch.patch_vggt import patch_model, update_args_per_forward
    from test_layer_value_attention import ToyAggregator
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs/evaluation'), version_base=None):
        cfg = compose(config_name='eval_hierarchical_value_vggt')
    assert cfg.model.patch_module.retention == .1
    assert cfg.datasets['7scenes'].max_frames == 128
    model = torch.nn.Module()
    model.aggregator = ToyAggregator().eval()
    reference = copy.deepcopy(model)
    original = [b.attn for b in model.aggregator.global_blocks]
    patch_model(model, Dict(patch_module=dict(type='hierarchical_value', retention=1., fine_ratio=1., coarse_size=5,
                                           sparse_start=3, sparse_end=9, initialization='lightweight')))
    update_args_per_forward(model, 28, 28)
    assert model.aggregator.global_blocks[1].attn is original[1]
    assert model.aggregator.global_blocks[9].attn is original[9]
    assert all(isinstance(model.aggregator.global_blocks[i].attn, HierarchicalValueAttention) for i in range(2, 9))
    with torch.inference_mode():
        for frames in [2, 1]:
            images = torch.randn(1, frames, 3, 28, 28)
            torch.testing.assert_close(model.aggregator(images), reference.aggregator(images))
            state = model.hierarchical_value_state
            assert [r['layer'] for r in state.records] == list(range(3, 10))
            assert state.history.shape[0] == 9*frames
    with pytest.raises(ValueError):
        patch_model(model, Dict(patch_module=dict(type='hierarchical_value')))


def test_all_24_layers_patched_and_uniform_layer_zero_is_value_only(monkeypatch):
    import saf3r.patch.hierarchical_value_attention as module
    from addict import Dict
    from saf3r.patch.patch_vggt import patch_model
    from test_layer_value_attention import ToyAggregator
    model = torch.nn.Module()
    model.aggregator = ToyAggregator().eval()
    reference = copy.deepcopy(model)
    patch_model(model, Dict(patch_module=dict(type='hierarchical_value', retention=1., fine_ratio=1., coarse_size=5)))
    assert all(isinstance(b.attn, HierarchicalValueAttention) for b in model.aggregator.global_blocks)
    with torch.inference_mode():
        images = torch.randn(1, 2, 3, 28, 28)
        torch.testing.assert_close(model.aggregator(images), reference.aggregator(images))
    state = model.hierarchical_value_state
    assert [r['layer_zero_based'] for r in state.records] == list(range(24))
    assert state.records[0]['uniform_value_only_start']
    assert state.records[0]['attention_budget'] == 0
    assert not any(r['uniform_value_only_start'] for r in state.records[1:])
    x = torch.randn(200, 8)
    labels, reps = module.uniform_hierarchy(x, .15)
    torch.testing.assert_close(labels, torch.arange(200)*30//200)
    torch.testing.assert_close(labels[reps], torch.arange(30))
    assert int(sizes(labels, 30).max()-sizes(labels, 30).min()) <= 1


def test_uniform_bootstrap_skips_coarse_attention_and_spreads_representatives(monkeypatch):
    import saf3r.patch.hierarchical_value_attention as module
    def forbidden(p):
        raise AssertionError('Uniform first layer must not compute reliability')
    monkeypatch.setattr(module, 'reliability', forbidden)
    state = HierarchicalValueState(retention=.5, fine_ratio=.7, coarse_size=5)
    state.reset(2, 3, 3)
    original = Attention(dim=8, num_heads=2).eval()
    wrapper = HierarchicalValueAttention(original, state, 0).eval()
    x = torch.randn(1, 28, 8)
    with torch.inference_mode():
        out = wrapper(x)
        q, k, v = original.qkv(x).reshape(1, 28, 3, 2, 4).permute(2, 0, 3, 1, 4).unbind(0)
        patches, specials = module.token_layout(2, 9, 5, x.device)
        z, reps, m, counts = state.hierarchy
        priority = torch.arange(4)*len(reps)//4
        selected = torch.cat((patches[reps[priority]], specials))
        expected = torch.nn.functional.scaled_dot_product_attention(q, k[:, :, selected], v[:, :, selected])
        expected = original.proj(expected.transpose(1, 2).reshape(1, 28, 8))
        torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_single_sparse_qk_kernel_and_returned_logits(dtype):
    from saf3r.triton.hierarchy_attention import indexed_attention_with_logits
    torch.manual_seed(2)
    q = torch.randn(3, 13, 16, device='cuda', dtype=dtype)[:, 2:11]
    k = torch.randn(3, 233, 16, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    indices = torch.stack([torch.randperm(226, device='cuda')[:151] for _ in range(9)]).int()
    specials = torch.arange(226, 233, device='cuda')
    actual, logits = indexed_attention_with_logits(q, k, v, indices, specials, .25)
    expected, expected_logits = sparse_reference(q, k, v, indices.long(), specials, .25)
    torch.testing.assert_close(logits, expected_logits, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(actual, expected, rtol=.01 if dtype == torch.bfloat16 else 1e-5,
                               atol=.005 if dtype == torch.bfloat16 else 1e-6)
