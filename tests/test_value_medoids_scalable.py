import numpy as np
import torch
import pytest

from scripts.value_medoids_scalable import blocked_distances, eager_pam, nearest_two, swap_costs
from scripts.visualize_vggt_value_medoids import best_swap


def test_blocked_distances_and_candidate_swap_costs():
    torch.manual_seed(42)
    x = torch.randn(19, 5)
    distance = blocked_distances(x, block_size=4)
    torch.testing.assert_close(distance, torch.cdist(x, x), atol=2e-6, rtol=2e-6)
    medoids = torch.tensor([0, 6, 12])
    nearest, positions = nearest_two(distance, medoids)
    costs = swap_costs(distance, nearest[:, 0], nearest[:, 1], positions[:, 0], 3)
    objective = distance[medoids].amin(0).sum()
    for c in range(19):
        for removed in range(3):
            proposed = medoids.clone()
            proposed[removed] = c
            expected = distance[proposed].amin(0).sum() - objective
            assert abs(float(costs[c, removed]-expected)) < 1e-5


def test_bounded_initial_nearest_two_matches_full_gather():
    torch.manual_seed(17)
    distance = torch.rand(1100, 1100)
    medoids = torch.tensor([0, 123, 1099])
    expected = nearest_two(distance, medoids)
    actual = nearest_two(distance, medoids, max_entries=1)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_eager_pam_finishes_at_exhaustive_single_swap_optimum():
    for seed in range(3):
        torch.manual_seed(seed)
        distance = blocked_distances(torch.randn(37, 7), block_size=8)
        result = eager_pam(distance, 5, seed=seed, candidate_batch=4)
        assert result["converged"]
        delta, _, _ = best_swap(distance, torch.tensor(result["medoids"]))
        assert delta >= -result["tolerance"]
        assert len(np.unique(result["medoids"])) == 5
        assert np.all(np.diff(result["objective_history"]) < 0)


def test_compiled_cpu_pam_certificate_matches_independent_best_swap():
    pytest.importorskip("kmedoids")
    from scripts.value_medoids_scalable import rust_pam
    torch.manual_seed(31)
    distance = blocked_distances(torch.randn(43, 6))
    result = rust_pam(distance, 6, seed=7, cpu_threads=2)
    assert result["converged"]
    assert result["certificate_swaps"] == 0
    delta, _, _ = best_swap(distance, torch.tensor(result["medoids"]))
    assert delta >= -result["tolerance"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for hybrid path")
def test_hybrid_incremental_top_two_and_final_swap_certificate():
    from scripts.value_medoids_scalable import host_distances
    torch.manual_seed(51)
    values = torch.randn(59, 8)
    distance = host_distances(values, block_size=7)
    torch.testing.assert_close(distance, torch.cdist(values, values, compute_mode="donot_use_mm_for_euclid_dist"), atol=2e-6, rtol=2e-6)
    fit = eager_pam(distance, 7, seed=5, candidate_batch=8, compute_device="cuda")
    assert fit["converged"]
    delta, _, _ = best_swap(distance, torch.tensor(fit["medoids"]))
    assert delta >= -fit["tolerance"]
    chosen = distance[:, torch.tensor(fit["medoids"])].min(1).values.numpy()
    np.testing.assert_array_equal(fit["distances"], chosen)
    resident = eager_pam(distance.cuda(), 7, seed=5, candidate_batch=8)
    assert resident["converged"]
    delta, _, _ = best_swap(distance, torch.tensor(resident["medoids"]))
    assert delta >= -resident["tolerance"]
    np.testing.assert_array_equal(resident["distances"], distance[:, torch.tensor(resident["medoids"])].min(1).values.numpy())
