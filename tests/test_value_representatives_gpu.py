import numpy as np
import pytest
import torch

from saf3r.patch.value_clustering_gpu import (
    centroid_nearest_gpu, sampled_medoid_gpu, local_refine_gpu,
    source_neighborhoods_gpu, recenter_partition_gpu,
)
from saf3r.patch.value_evolving_attention import LayerValueState

DEVICES = ["cpu"]+(["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_centroid_selects_real_member_and_is_exact_for_squared_not_unsquared_objective(device):
    x = torch.tensor([[0.], [1.], [2.], [4.], [100.]], device=device, dtype=torch.float64)
    z, old = torch.zeros(5, device=device, dtype=torch.long), torch.tensor([4], device=device)
    chosen = centroid_nearest_gpu(x, z, old)
    exact, _ = recenter_partition_gpu(x, z, old)
    assert int(chosen[0]) == 3 and int(exact[0]) == 2
    squared_costs = (x[:, None]-x[None]).square().sum((1, 2))
    assert squared_costs[chosen[0]] == squared_costs.min()


@pytest.mark.parametrize("device", DEVICES)
def test_centroid_and_sampled_preserve_previous_on_ties_and_choose_cluster_members(device):
    x = torch.zeros((13, 3), device=device, dtype=torch.float64)
    z = torch.tensor([0]*8+[1]*4+[2], device=device)
    old = torch.tensor([5, 10, 12], device=device)
    torch.testing.assert_close(centroid_nearest_gpu(x, z, old), old)
    torch.testing.assert_close(sampled_medoid_gpu(x, z, old, 4), old)


@pytest.mark.parametrize("device", DEVICES)
def test_sampled_equals_exact_when_candidates_cover_all_members(device):
    torch.manual_seed(92)
    x = torch.randn((15, 4), device=device, dtype=torch.float64)
    z = torch.arange(3, device=device).repeat_interleave(5)
    old = torch.tensor([0, 5, 10], device=device)
    expected, _ = recenter_partition_gpu(x, z, old)
    torch.testing.assert_close(sampled_medoid_gpu(x, z, old, 8, block_size=3), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("strategy", ["previous", "centroid", "sampled"])
def test_approximate_updates_keep_fixed_neighborhoods_anchors_and_k_without_exact_kernel(device, strategy, monkeypatch):
    import saf3r.patch.value_clustering_gpu as implementation
    torch.manual_seed(11)
    x = torch.randn(72, 4, dtype=torch.float64, device=device)
    z = torch.arange(9, device=device).repeat_interleave(8)
    initial = torch.arange(0, 72, 8, device=device)
    graph = source_neighborhoods_gpu(x, initial, 3)
    def forbidden(*args, **kwargs):
        raise AssertionError("Approximate update called exhaustive member-pair kernel")
    monkeypatch.setattr(implementation, "recenter_partition_gpu", forbidden)
    fit = local_refine_gpu(x, z, initial, graph, representative_update=strategy)
    for labels, medoids in fit["snapshots"]:
        torch.testing.assert_close(labels[initial], torch.arange(9, device=device))
        torch.testing.assert_close(labels[medoids], torch.arange(9, device=device))
        assert bool((graph[z] == labels[:, None]).any(1).all())
        assert len(medoids.unique()) == 9
    assert fit["member_pair_counts"] == []
    if strategy == "previous":
        torch.testing.assert_close(fit["medoids"], initial)
    expected = (x-x[fit["medoids"][fit["assignments"]]]).norm(dim=1).sum()
    torch.testing.assert_close(fit["objective"], expected)


def test_approximate_backend_requires_explicit_gpu_and_fast_config_composes():
    with pytest.raises(ValueError, match="GPU backend"):
        LayerValueState(representative_update="centroid")
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/evaluation"):
        cfg = compose(config_name="eval_layer_value_local_vggt_fast")
    assert cfg.model.name == "VGGT"
    assert cfg.model.patch_module.representative_update == "centroid"
    assert cfg.model.patch_module.local_backend == "gpu"
    assert cfg.model.patch_module.retention == .035
    assert cfg.model.patch_module.iterations == 2
