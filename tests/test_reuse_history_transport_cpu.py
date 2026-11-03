"""CPU invariants for direct Sparse History -> current dense-proxy transport."""
import torch

from saf3r.patch.reuse_history_transport import direct_sparse_history_transport


def test_observed_unobserved_share_each_grid_probability():
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    sizes = torch.tensor([3., 4.])
    ids = torch.tensor([[0, 2, 3, -1], [1, 4, -1, -1]])
    p_sparse = torch.tensor([[.64, .20, .06, 0], [.55, .28, 0, 0]])
    prev = torch.tensor([[.7, .3], [.6, .4]])
    curr = torch.tensor([[.4, .6], [.2, .8]])
    observed, unseen, beta = direct_sparse_history_transport(
        ids, p_sparse, labels, sizes, prev, curr,
        eta_down=.5, eta_up=2.0,
    )
    valid = ids >= 0
    region = labels[ids.clamp_min(0)]
    obs_per_grid = torch.zeros_like(curr).scatter_add_(1, region, observed)
    n_observed = torch.zeros_like(curr).scatter_add_(1, region, valid.float())
    total = obs_per_grid + (sizes[None] - n_observed) * unseen
    torch.testing.assert_close(total, curr, rtol=1e-6, atol=1e-7)
    assert (observed.masked_select(~valid) == 0).all()
    assert ((beta >= 0) & (beta <= 1)).all()


def test_zero_confidence_uses_uniform_prior():
    labels = torch.tensor([0, 0, 1, 1])
    ids = torch.tensor([[0, 2]])
    # Force a near-zero previous-grid mass to make confidence zero.
    old = torch.tensor([[0., 0.]])
    current = torch.tensor([[.3, .7]])
    observed, remaining, beta = direct_sparse_history_transport(
        ids, torch.tensor([[.8, .1]]), labels,
        torch.tensor([2., 2.]), old, current,
    )
    torch.testing.assert_close(beta, torch.zeros_like(beta))
    torch.testing.assert_close(observed, torch.tensor([[.15, .35]]))
    torch.testing.assert_close(remaining, torch.tensor([[.15, .35]]))


def test_no_history_and_all_observed_zero_scores():
    labels = torch.tensor([0, 0, 1, 1])
    sizes = torch.tensor([2., 2.])
    prev = torch.tensor([[.5, .5]])
    current = torch.tensor([[.25, .75]])
    empty = torch.empty((1, 0), dtype=torch.long)
    result, remaining, _ = direct_sparse_history_transport(
        empty, torch.empty((1, 0)), labels, sizes, prev, current,
    )
    assert result.shape == (1, 0)
    torch.testing.assert_close(remaining, torch.tensor([[.125, .375]]))

    # All keys observed with zero inherited scores. The fallback is uniform.
    ids = torch.tensor([[0, 1, 2, 3]])
    # beta = 1 when R == 1, zero history scores would otherwise divide by 0.
    p, unseen, _ = direct_sparse_history_transport(
        ids, torch.zeros((1, 4)), labels, sizes, prev, prev,
    )
    torch.testing.assert_close(p, torch.tensor([[.25, .25, .25, .25]]))
    assert torch.isfinite(unseen).all()


def test_constant_history_falls_back_to_original_sparse_prior():
    labels = torch.tensor([0, 0, 0])
    sizes = torch.tensor([3.])
    ids = torch.tensor([[0, 2]])
    old = torch.tensor([[1.]])
    p = torch.tensor([[.8, .2]])
    observed, unseen, beta = direct_sparse_history_transport(
        ids, p, labels, sizes, old, old, eta_down=2, eta_up=3,
    )
    torch.testing.assert_close(beta, torch.ones_like(beta))
    expected = torch.tensor([.8, .2, 1/3])
    expected /= expected.sum()
    torch.testing.assert_close(observed[0], expected[:2])
    torch.testing.assert_close(unseen[0, 0], expected[2])
