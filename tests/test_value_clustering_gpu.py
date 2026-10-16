import numpy as np
import pytest
import torch

from saf3r.patch.value_clustering import source_neighborhoods, local_refine, recenter_partition
from saf3r.patch.value_clustering_gpu import source_neighborhoods_gpu, local_refine_gpu, recenter_partition_gpu
from saf3r.patch.value_evolving_attention import LayerValueState


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_neighbors_match_cpu_with_duplicate_vectors(device):
    values = np.arange(24, dtype=float).reshape(12, 2)
    values[4] = values[3]
    medoids = np.arange(12)
    for q in (0, 4, 8, 11):
        actual = source_neighborhoods_gpu(torch.tensor(values, device=device), torch.tensor(medoids, device=device), q, block_size=3)
        np.testing.assert_array_equal(actual.cpu(), source_neighborhoods(values, medoids, q, block_size=3))


@pytest.mark.parametrize("device", DEVICES)
def test_topk_boundary_ties_choose_smallest_labels_not_arbitrary_topk_ids(device):
    values = np.zeros((31, 4))
    values[-1] = 1
    medoids = np.arange(len(values))
    for q in (0, 1, 8, 30):
        actual = source_neighborhoods_gpu(torch.tensor(values, device=device), torch.tensor(medoids, device=device), q, block_size=7)
        np.testing.assert_array_equal(actual.cpu(), source_neighborhoods(values, medoids, q))


@pytest.mark.parametrize("device", DEVICES)
def test_ragged_exact_medoids_and_empty_pair_chunk_boundaries(device):
    values = np.array([[0.], [1.], [2.], [4.], [100.], [9.], [8.], [8.]])
    labels = np.array([0, 0, 0, 0, 0, 1, 2, 2])
    old = np.array([4, 5, 7])
    medoids, pairs = recenter_partition_gpu(torch.tensor(values, device=device), torch.tensor(labels, device=device),
                                           torch.tensor(old, device=device), pair_block=3)
    np.testing.assert_array_equal(medoids.cpu(), recenter_partition(values, labels, old))
    assert int(pairs) == 30
    assert int(medoids[0]) == 2  # Exact medoid, not nearest-to-mean token 3.


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", [3, 9, 41])
def test_two_iterations_match_cpu_fixed_neighborhoods_and_anchors(device, seed):
    rng = np.random.default_rng(seed)
    previous = rng.normal(size=(56, 5))
    current = previous+rng.normal(size=previous.shape)*.8
    labels = np.repeat(np.arange(7), 8)
    medoids = np.arange(0, 56, 8)
    graph = source_neighborhoods(previous, medoids, 3)
    reference = local_refine(current, labels, medoids, graph, max_iterations=2, anchors=medoids,
                             save_iterations=(1, 2), stop_when_converged=False)
    actual = local_refine_gpu(torch.tensor(current, device=device), torch.tensor(labels, device=device),
                              torch.tensor(medoids, device=device), torch.tensor(graph, device=device), 2)
    np.testing.assert_array_equal(actual["assignments"].cpu(), reference["assignments"])
    np.testing.assert_array_equal(actual["medoids"].cpu(), reference["medoids"])
    np.testing.assert_allclose(actual["objective"].cpu(), reference["objective"], rtol=1e-12)
    for i, (z, m) in enumerate(actual["snapshots"], 1):
        np.testing.assert_array_equal(z.cpu(), reference["snapshots"][i]["assignments"])
        np.testing.assert_array_equal(m.cpu(), reference["snapshots"][i]["medoids"])


@pytest.mark.parametrize("device", DEVICES)
def test_state_resets_and_uses_cached_seed_only_at_layer_two(device):
    values = torch.arange(36, device=device, dtype=torch.float32).reshape(18, 2)
    labels = torch.arange(6, device=device).repeat_interleave(3)
    medoids = torch.arange(0, 18, 3, device=device)
    state = LayerValueState(retention=1/3, neighbors=2, local_backend="gpu",
                            seed_cache={"assignments": labels, "medoids": medoids})
    state.select(values, 2)
    state.select(values.flip(0), 3)
    state.select(values, 4)
    assert state.previous_layer == 4 and state.labels.device == values.device
    assert [r["source_layer_1based"] for r in state.records] == [None, 2, 3]
    state.reset(1, 2, 9)
    assert state.seed_values is None and state.selection_events == []
    torch.testing.assert_close(state.select(values, 2), medoids)
