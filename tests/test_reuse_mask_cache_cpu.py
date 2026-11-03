"""CPU regression checks for allocation-preserving Mask-builder optimizations."""
import torch

from saf3r.patch.reuse_residual_exploration import (
    ReuseExploreConfig,
    ReuseExploreState,
    count_history_rep_overlap,
)


def _original_dense_overlap(ids, valid, reps, num_patches):
    hits = torch.zeros((ids.shape[0], num_patches), dtype=torch.int32)
    hits.scatter_add_(1, ids.long().clamp_min(0), valid.to(torch.int32))
    return (hits[:, reps] > 0).sum(1)


def test_compact_rep_overlap_matches_dense_diagnostic():
    # Includes duplicate keys, invalid padding, and a completely empty bank.
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    reps = torch.tensor([1, 3, 4], dtype=torch.long)
    for ids in (
        torch.tensor([[1, 5, -1, 3, 3], [0, 2, 4, -1, -1]]),
        torch.full((2, 4), -1, dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
    ):
        valid = ids >= 0
        expected = _original_dense_overlap(ids, valid, reps, len(labels))
        actual = count_history_rep_overlap(ids, valid, labels, reps)
        torch.testing.assert_close(actual, expected)


def test_probe_cache_and_grid_log_match_original_expression():
    state = ReuseExploreState(ReuseExploreConfig())
    state.reset(frames=3, height=14, width=14, device=torch.device("cpu"))
    reference = (
        torch.arange(state.groups)[:, None] * 32
        + torch.tensor([4, 12, 20, 28])
    ).clamp_max(state.n - 1)
    torch.testing.assert_close(state.probe_indices, reference)
    torch.testing.assert_close(state.grid_log_sizes, state.grid_sizes.log())
    for start in (0, 1, max(0, state.groups - 2)):
        g = min(2, state.groups - start)
        torch.testing.assert_close(
            state.probe_indices[start : start + g], reference[start : start + g]
        )
