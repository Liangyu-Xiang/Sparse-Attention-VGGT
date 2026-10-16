import pytest
import torch
from saf3r.patch.coarse_sensing import pool_keys, coarse_attention, reliability_statistics


def test_opposite_heads_cancel_before_qk_but_not_in_matched_head_relevance():
    q = torch.tensor([[[9., 0.], [0., 9.]], [[-9., 0.], [0., -9.]]])
    k = torch.tensor([[[1., 0.], [0., 1.]], [[-1., 0.], [0., -1.]]])
    groups = torch.tensor([0, 1])
    old, old_stats, _ = coarse_attention(q, k, groups, 2, mode='legacy_mean_qk')
    new, stats, _ = coarse_attention(q, k, groups, 2)
    torch.testing.assert_close(old, torch.full_like(old, .5))
    assert float(old_stats['specificity'][0]) == 0
    assert (stats['specificity'] > .95).all()
    assert (new[:, 0, 0] > .99).all() and (new[:, 1, 1] > .99).all()


def test_population_bias_does_not_enter_default_relevance_statistics():
    q = torch.zeros(2, 7, 3)
    k = torch.zeros(2, 5, 3)
    groups = torch.tensor([0, 1, 1, 1, 1])
    p, stats, capacity = coarse_attention(q, k, groups, 2)
    biased, other, _ = coarse_attention(q, k, groups, 2, population_bias=True)
    torch.testing.assert_close(p, torch.full_like(p, .5))
    torch.testing.assert_close(biased[0, 0], torch.tensor([.2, .8]))
    torch.testing.assert_close(capacity, torch.tensor([1., 4.]))
    assert (stats['concentration'] == 0).all()
    assert (other['concentration'] > .2).all()
    assert (other['specificity'] < 1e-6).all()  # prior alone is not specificity


def test_pooling_is_per_head_and_preserves_token_population():
    keys = torch.arange(24.).reshape(2, 4, 3)
    groups = torch.tensor([0, 0, 1, 1])
    pooled, count = pool_keys(keys, groups, 2)
    torch.testing.assert_close(pooled[:, 0], keys[:, :2].mean(1))
    torch.testing.assert_close(pooled[:, 1], keys[:, 2:].mean(1))
    torch.testing.assert_close(count, torch.tensor([2., 2.]))


def test_reliability_never_averages_query_and_head_axes_together():
    p = torch.stack((torch.eye(4), torch.full((4, 4), .25)))
    r = reliability_statistics(p)
    torch.testing.assert_close(r['specificity'], torch.tensor([1., 0.]))
    torch.testing.assert_close(r['concentration'], torch.tensor([1., 0.]))
    assert reliability_statistics(torch.ones(2, 5, 1))['cs'].shape == (2,)


def test_legacy_control_matches_mean_q_mean_k_with_size_bias():
    torch.manual_seed(3)
    q, k = torch.randn(3, 11, 8), torch.randn(3, 6, 8)
    groups = torch.tensor([0, 1, 1, 2, 2, 2])
    p, _, population = coarse_attention(q, k, groups, 3, mode='legacy_mean_qk', population_bias=True)
    pooled = torch.stack([k[:, groups == c].mean(1) for c in range(3)], 1)
    expected = (q.mean(0)@pooled.mean(0).T/8**.5+population.log()).softmax(-1)
    torch.testing.assert_close(p[0], expected)


def test_empty_coarse_group_is_rejected():
    with pytest.raises(ValueError):
        pool_keys(torch.randn(2, 4, 3), torch.zeros(4, dtype=torch.long), 2)


def test_dense_mass_is_not_mislabeled_topk_recall():
    from scripts.evaluate_coarse_sensing import localization_metrics
    p = torch.tensor([[[.9, .1]]])
    mass = torch.tensor([[[.8, .2]]])
    dense_patch_mass = torch.ones(1, 1)
    # Two oracle top-K keys, one in either region: recall=.5 but mass=.8.
    top_groups = torch.tensor([[[0, 1]]])
    valid = torch.ones_like(top_groups, dtype=torch.bool)
    metrics = localization_metrics(p, mass, dense_patch_mass, top_groups, valid, torch.tensor([1., 1.]), .5)
    assert float(metrics['dense_patch_mass_fraction'][0]) == pytest.approx(.8)
    assert float(metrics['dense_patch_topk_recall'][0]) == pytest.approx(.5)


def test_masked_operator_matches_explicit_selected_softmax():
    from scripts.evaluate_coarse_sensing import masked_output
    torch.manual_seed(4)
    logits, values = torch.randn(2, 7, 11), torch.randn(2, 11, 3)
    indices = torch.tensor([2, 4, 8])
    actual = masked_output(logits, values, indices, chunk=3)
    expected = logits[..., indices].softmax(-1)@values[:, indices]
    torch.testing.assert_close(actual, expected)
    full = torch.arange(logits.shape[-1])
    torch.testing.assert_close(masked_output(logits, values, full), logits.softmax(-1)@values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_headwise_sensing_stays_fp32_under_outer_autocast():
    torch.manual_seed(8)
    q, k = torch.randn(2, 9, 8, device='cuda'), torch.randn(2, 12, 8, device='cuda')
    groups = torch.arange(12, device='cuda')//3
    expected, _, _ = coarse_attention(q, k, groups, 4)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual, _, _ = coarse_attention(q, k, groups, 4)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
