import numpy as np
import torch
import torch.nn.functional as F

from scripts.compare_value_medoids_uniform import PHASES, fidelity, joint_uniform, uniform_cells


def test_uniform_sampling_is_unique_in_range_and_keeps_exact_frame_budgets():
    for counts in ([103, 104], [118, 89], [71, 136]):
        for phase in PHASES:
            indices = joint_uniform(28, 37, counts, phase)
            assert len(indices) == len(np.unique(indices)) == 207
            assert indices.min() >= 0 and indices.max() < 2072
            assert [(indices // 1036 == f).sum() for f in (0, 1)] == counts
    assert len(uniform_cells(28, 37, 0)) == 0
    assert len(np.unique(uniform_cells(28, 37, 1036))) == 1036


def test_output_nmse_matches_sdpa_with_same_selected_keys_and_softmax():
    torch.manual_seed(4)
    q, k, v = torch.randn(9, 4), torch.randn(13, 4), torch.randn(13, 4)
    logits = q @ k.T / 2
    probabilities = logits.softmax(-1)
    target = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None])[0, 0]
    selected = torch.tensor([0, 1, 5, 8, 11])
    sparse = F.scaled_dot_product_attention(q[None, None], k[selected][None, None], v[selected][None, None])[0, 0]
    expected_nmse = (sparse - target).square().double().sum() / target.square().double().sum()
    actual = fidelity(logits, v, selected, target, probabilities)
    assert abs(actual["attention_output_nmse"] - expected_nmse.item()) < 1e-6
    assert abs(actual["dense_attention_mass_retained"] - probabilities[:, selected].sum(-1).mean().item()) < 1e-7
    full = fidelity(logits, v, torch.arange(13), target, probabilities)
    assert full["attention_output_nmse"] < 1e-12
