"""CUDA regression: Cluster-state direct support is an unsorted permutation of old packing.

Run: python -m pytest -q tests/test_direct_compact_pack.py
"""
import pytest
import torch

pytest.importorskip("triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton required")
def test_cluster_direct_pack_matches_original_key_sets_and_counts():
    from saf3r.patch.reuse_inference_ops import pack as old_pack
    from saf3r.triton.reuse_direct_pack import pack as new_pack

    device = "cuda"
    groups, clusters, members_per_cluster, width = 3, 12, 5, 8
    n = clusters * members_per_cluster
    labels = (torch.arange(n, device=device) // members_per_cluster).long()
    reps = torch.arange(0, n, members_per_cluster, device=device, dtype=torch.int32)
    population = torch.full((clusters,), members_per_cluster, device=device, dtype=torch.float32)
    table = torch.full((clusters, width), -1, device=device, dtype=torch.int32)
    table[:, :members_per_cluster] = torch.arange(n, device=device, dtype=torch.int32).reshape(
        clusters, members_per_cluster)
    geometry = {"members": table, "width": width}
    selected = torch.tensor([
        [0, 1, 5, 10, 1, 0, 5, 10, 10, 1, 5, 0],
        [10, 5, 1, 0, 0, 10, 5, 1, 5, 10, 0, 1],
        [5, 0, 10, 1, 10, 5, 0, 1, 1, 5, 10, 0],
    ], dtype=torch.int64, device=device)
    qualified = (torch.arange(n, device=device)[None, :] % 3 == 0).expand(groups, -1).contiguous()
    qualified[1] = torch.arange(n, device=device) % 2 == 0
    qualified[2] = torch.arange(n, device=device) % 4 == 0
    amount = qualified.reshape(groups, clusters, members_per_cluster).sum(-1).int()
    context = {
        "qualified": qualified,
        "eligible_count": amount,
        "remove_rep": torch.zeros(groups, clusters, device=device, dtype=torch.bool),
    }
    old_context = {k: v.clone() for k, v in context.items()}
    new_context = {k: v.clone() for k, v in context.items()}
    cap = n

    old_ids, _, old_record = old_pack(
        selected, None, reps, labels, population, cap, old_context)
    new_ids, _, new_record = new_pack(
        selected, reps, labels, population, cap, geometry, new_context)

    for i in range(groups):
        a = old_ids[i][old_ids[i] >= 0].sort().values
        b = new_ids[i][new_ids[i] >= 0].sort().values
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert len(torch.unique(b)) == len(b)

    torch.testing.assert_close(old_record["counts"], new_record["counts"], atol=0, rtol=0)
    torch.testing.assert_close(old_context["pack_error"], new_context["pack_error"], atol=0, rtol=0)
    assert not bool(new_context["pack_error"].any())
    assert int(new_record["native_cost_count_error"].abs().max()) == 0
