import json
import torch
from saf3r.patch.value_diagnostics import ValueDiagnostics
from saf3r.patch.hierarchical_value_attention import HierarchicalValueState, HierarchicalValueAttention
from saf3r.models.official_vggt.layers.attention import Attention


def test_raw_diagnostics_reconstruct_mask_lifecycle_and_oracles(tmp_path, monkeypatch):
    import saf3r.patch.hierarchical_value_attention as module
    monkeypatch.setattr(module, 'reliability', lambda p: (p.new_tensor(.5), p.new_tensor(.5), p.new_tensor(1.)))
    collector = ValueDiagnostics(tmp_path, raw=True, dense_output=True, dense_oracle=True)
    state = HierarchicalValueState(retention=.5, fine_ratio=.7, coarse_size=5, diagnostics=collector, query_chunk=9)
    state.reset(2, 3, 3)
    torch.manual_seed(1)
    x = torch.randn(1, 28, 8)
    with torch.inference_mode():
        for layer in range(3):
            wrapper = HierarchicalValueAttention(Attention(dim=8, num_heads=2).eval(), state, layer).eval()
            x = x+.1*wrapper(x)
    collector.save({'test': True})
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert len(summary['layers']) == 3 and len(summary['per_head']) == 6
    assert summary['layers'][0]['history_survival_rate'] is None
    for layer in range(3):
        raw = torch.load(tmp_path/f'layer_{layer:02d}.pt', weights_only=False)
        z = raw['fine_cluster_id'].long()
        torch.testing.assert_close(z[raw['fine_rep_index'].long()], torch.arange(len(raw['fine_rep_index'])))
        indices = raw['final_sparse_indices']
        assert indices.shape == raw['provenance'].shape == raw['selection_origin'].shape == (28, 14)
        assert all(len(row.unique()) == 14 for row in indices)
        assert raw['exact_sparse_logits'].shape == (2, 28, 14)
        assert raw['dense_topk_indices'].shape == (2, 28, 14)
        assert ((raw['retained_dense_mass'] >= 0) & (raw['retained_dense_mass'] <= 1+1e-6)).all()
        assert ((raw['dense_topk_recall'] >= 0) & (raw['dense_topk_recall'] <= 1)).all()
        row = summary['layers'][layer]
        assert int((raw['provenance'] == 3).sum()) == row['promoted_value']
        if layer == 0:
            assert raw['coarse_attention'] is None
            assert row['attention_budget'] == 0
        else:
            assert raw['coarse_attention'] is not None
            assert (raw['group_budget'].sum(-1) == 2).all()
            assert (raw['validated_edges'].shape == (28, 2))
            assert row['reused']+row['explored'] == 56
            assert row['validated'] == 56
    for name in ['runtime_breakdown.csv', 'mask_composition.csv', 'history_quality.csv', 'accuracy_per_layer.csv', 'accuracy_per_head.csv']:
        assert (tmp_path/name).is_file()


def test_mixed_norm_qk_and_autocast_v_dense_reference(tmp_path):
    if not torch.cuda.is_available():
        return
    collector = ValueDiagnostics(tmp_path, raw=False)
    state = HierarchicalValueState(retention=.5, fine_ratio=.7, coarse_size=5, diagnostics=collector)
    state.reset(2, 3, 3)
    wrapper = HierarchicalValueAttention(Attention(dim=8, num_heads=2, qk_norm=True).cuda().eval(), state, 0).eval()
    class FloatCast(torch.nn.Module):
        def forward(self, x):
            return x.float()
    wrapper.original.q_norm = torch.nn.Sequential(wrapper.original.q_norm, FloatCast())
    wrapper.original.k_norm = torch.nn.Sequential(wrapper.original.k_norm, FloatCast())
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = wrapper(torch.randn(1, 28, 8, device='cuda'))
    assert torch.isfinite(out).all()
    assert len(collector.head_rows) == 2
