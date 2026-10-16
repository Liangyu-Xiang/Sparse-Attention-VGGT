import json
import copy
import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from saf3r.models.official_vggt.layers.attention import Attention
from saf3r.patch.value_evolving_attention import (
    LayerValueAttention, LayerValueState, install_layer_value_attention,
    representative_attention, token_layout,
)


def test_all_queries_use_shared_representative_key_and_value_indices():
    torch.manual_seed(5)
    q, k, v = [torch.randn(1, 3, 10, 4) for _ in range(3)]
    patches, specials = token_layout(2, 3, 2, "cpu")
    medoids = np.array([1, 4])
    selected = torch.cat((specials, patches[medoids])).sort().values
    mask = torch.zeros((10, 10), dtype=torch.bool)
    mask[:, selected] = True
    actual = representative_attention(q, k, v, patches, specials, medoids)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    dropped = torch.tensor([i for i in patches.tolist() if i not in selected.tolist()])
    changed = v.clone()
    changed[:, :, dropped] += 1000
    torch.testing.assert_close(representative_attention(q, k, changed, patches, specials, medoids), actual)


def test_full_retention_recovers_dense_for_every_query_including_specials():
    torch.manual_seed(6)
    q, k, v = [torch.randn(1, 2, 18, 4) for _ in range(3)]
    patches, specials = token_layout(2, 4, 5, "cpu")
    torch.testing.assert_close(representative_attention(q, k, v, patches, specials, np.arange(8)),
                               F.scaled_dot_product_attention(q, k, v))


def test_dense_seed_then_recursive_layer_average_not_head_matching(tmp_path):
    torch.manual_seed(8)
    state = LayerValueState(retention=.5, neighbors=1, seed_backend="device", output_dir=tmp_path)
    state.reset(2, 2, 2)
    modules = [Attention(dim=8, num_heads=2, qk_norm=True).eval() for _ in range(3)]
    wrappers = [LayerValueAttention(m, state=state, layer_idx=i+1).eval() for i, m in enumerate(modules)]
    x = torch.randn(1, 18, 8)
    with torch.inference_mode():
        torch.testing.assert_close(wrappers[0](x), modules[0](x))
        previous_medoids = state.medoids.copy()
        x2 = torch.randn_like(x)
        actual = wrappers[1](x2)
        qkv = modules[1].qkv(x2).reshape(1, 18, 3, 2, 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        patches, specials = token_layout(2, 4, 5, "cpu")
        np.testing.assert_array_equal(state.previous_values, v[0, :, patches].float().mean(0).numpy())
        expected = representative_attention(modules[1].q_norm(q), modules[1].k_norm(k), v,
                                             patches, specials, state.medoids, modules[1].scale)
        expected = modules[1].proj(expected.transpose(1, 2).reshape(1, 18, 8))
        torch.testing.assert_close(actual, expected)
        np.testing.assert_array_equal(state.labels[previous_medoids], np.arange(4))
        np.testing.assert_array_equal(np.load(tmp_path/"forward_0001/layer_03/clusters.npz")["anchors"], previous_medoids)
        next_anchors = state.medoids.copy()
        wrappers[2](torch.randn_like(x))
        np.testing.assert_array_equal(np.load(tmp_path/"forward_0001/layer_04/clusters.npz")["anchors"], next_anchors)
    assert [r["source_layer_1based"] for r in state.records] == [None, 2, 3]
    assert state.medoids.shape == (4,)  # One set, not [heads, K].
    assert state.records[1]["iterations"] == 2
    assert json.loads((tmp_path/"forward_0001/layer_03/clustering.json").read_text())["attention_patch_keys"] == 4


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention(dim=8, num_heads=2)


class ToyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_blocks = nn.ModuleList([ToyBlock() for _ in range(24)])

    def forward(self, images):
        frames = images.shape[1]
        patches = (images.shape[-2]//14)*(images.shape[-1]//14)
        n = frames*(patches+5)
        x = torch.arange(n*8, dtype=torch.float32).reshape(1, n, 8)/100 + images.mean()
        for block in self.global_blocks:
            x = x + block.attn(x)
        return x


def test_installer_limits_sparse_layers_and_resets_every_input():
    torch.manual_seed(12)
    aggregator = ToyAggregator().eval()
    model = nn.Module()
    model.aggregator = aggregator
    originals = [b.attn for b in aggregator.global_blocks]
    state = install_layer_value_attention(model, retention=1., neighbors=1, seed_backend="device")
    assert aggregator.global_blocks[0].attn is originals[0]
    assert all(aggregator.global_blocks[i].attn is originals[i] for i in range(9, 24))
    with torch.inference_mode():
        aggregator(torch.randn(1, 2, 3, 28, 28))
        assert len(state.records) == 8
        aggregator(torch.randn(1, 1, 3, 28, 42))
    assert len(state.records) == 8 and state.forward_id == 2
    assert state.frames == 1 and len(state.previous_values) == 6
    assert [r["global_layer_1based"] for r in state.records] == list(range(2, 10))


def test_pam_stays_fp32_under_autocast(monkeypatch):
    import saf3r.patch.value_evolving_attention as implementation
    state = LayerValueState(retention=.5, neighbors=1, seed_backend="device")
    state.reset(2, 2, 2)
    module = LayerValueAttention(Attention(dim=8, num_heads=2).eval(), state=state, layer_idx=1).eval()
    base = implementation.blocked_distances
    observed = []

    def capture(values):
        result = base(values)
        observed.append((values.dtype, result.dtype))
        return result

    monkeypatch.setattr(implementation, "blocked_distances", capture)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = module(torch.randn(1, 18, 8))
    assert observed == [(torch.float32, torch.float32)]
    assert output.dtype == torch.bfloat16


def test_local_layer_without_dense_seed_fails():
    state = LayerValueState(retention=.5, neighbors=1)
    state.reset(2, 2, 2)
    module = LayerValueAttention(Attention(dim=8, num_heads=2).eval(), state=state, layer_idx=2).eval()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="immediately previous"):
        module(torch.randn(1, 18, 8))


def test_value_patch_and_reset_hook_can_be_removed_for_matched_method_comparison():
    torch.manual_seed(43)
    model = nn.Module()
    model.aggregator = ToyAggregator().eval()
    originals = [block.attn for block in model.aggregator.global_blocks]
    reference = copy.deepcopy(model)
    state = install_layer_value_attention(model, retention=1., neighbors=1, local_backend="gpu")
    state.reset_hook_handle.remove()
    for block, original in zip(model.aggregator.global_blocks, originals):
        block.attn = original
    delattr(model, "layer_value_state")
    with torch.inference_mode():
        images = torch.randn(1, 2, 3, 28, 28)
        torch.testing.assert_close(model.aggregator(images), reference.aggregator(images))
    assert state.records == []


@pytest.mark.parametrize("backend,updater", [("cpu", "exact"), ("gpu", "exact"),
                                             ("gpu", "centroid"), ("gpu", "previous"), ("gpu", "sampled")])
def test_framework_patch_configuration_and_full_retention_model_equivalence(backend, updater):
    from addict import Dict
    from saf3r.patch.patch_vggt import patch_model, update_args_per_forward
    torch.manual_seed(23)
    model = nn.Module()
    model.aggregator = ToyAggregator().eval()
    reference = copy.deepcopy(model)
    config = Dict(patch_module=dict(type="layer_value_local", retention=1., neighbors=1,
                                   seed_backend="device", local_backend=backend,
                                   representative_update=updater))
    patch_model(model, config)
    update_args_per_forward(model, 28, 28)
    images = torch.randn(1, 2, 3, 28, 28)
    with torch.inference_mode():
        expected = reference.aggregator(images)
        actual = model.aggregator(images)
    torch.testing.assert_close(actual, expected)
    assert model.layer_value_state.previous_layer == 9
