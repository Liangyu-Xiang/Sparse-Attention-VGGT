import torch
import pytest
from saf3r.patch.layer_shared_av import (
    spatial_hierarchy, sample_queries, branch_statistics, choose_edges, LayerSharedAVState)


def test_spatial_centers_and_fine_state():
    labels, reps, groups = spatial_hierarchy(2, 28, 37, .15, 128, 'cpu')
    assert torch.equal(labels[reps], torch.arange(len(reps)))
    assert len(groups) == len(reps)
    assert labels[:1036].max() < labels[1036:].min()
    a, b = LayerSharedAVState(), LayerSharedAVState()
    a.reset(2, 28, 37); b.reset(2, 28, 37)
    x = torch.randn(2072, 4)
    ha, hb = a.update(x, 0), b.update(x*100+50, 0)
    assert all(torch.equal(u, v) for u, v in zip(ha, hb))
    assert len(a.fine) == 2  # no recurrent coarse state


def test_sampling_and_statistics():
    ids = sample_queries(133248, .01, 'cpu')
    assert len(ids) == 1333 and len(ids.unique()) == len(ids)
    assert branch_statistics(torch.ones(9, 7)/7)['alpha'] == 0
    # Concentration with a common sink must not imply Attention reliability.
    assert branch_statistics(torch.tensor([[1., 0.]]).repeat(10, 1))['alpha_raw'] == 0
    diverse = torch.eye(100)
    assert branch_statistics(diverse, .8)['alpha'] == 1


@pytest.mark.parametrize('ka', [0, 1, 4, 7])
def test_selection_exact_budget_history_and_coverage(ka):
    labels = torch.arange(30)//3
    reps = torch.arange(10)*3+1
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    hierarchy = labels, reps, groups, torch.bincount(labels)
    p = torch.tensor([[.999, .0005, .0005], [.1, .2, .7]])
    history = torch.tensor([[1, 2, 3, 9, 15], [24, 25, 0, 29, 4]])
    indices, reused = choose_edges(p, 7, ka, hierarchy, history)
    assert (indices >= 0).all()
    for row in indices:
        assert len(row.unique()) == 7
        assert len(labels[row[ka:]].unique()) == 7-ka
        assert not torch.isin(labels[row[ka:]], labels[row[:ka]]).any()
    assert reused <= 2*ka
    # Capacity includes extra distinct validated tokens in covered fine cells.
    from saf3r.patch.hierarchical_value_attention import allocate_budget
    cap = torch.bincount(groups).expand(2, -1).clone()
    if ka:
        for r in range(2):
            hf = labels[history[r].unique()]
            cap[r] += torch.bincount(groups[hf], minlength=3)-torch.bincount(groups[hf.unique()], minlength=3)
    expected = allocate_budget(p, ka, cap)
    for row in range(len(p)):
        assert torch.equal(torch.bincount(groups[labels[indices[row, :ka]]], minlength=3), expected[row])


def test_history_validation_order_and_no_old_coarse_state():
    # All tokens have separate fine clusters; same group history is used in
    # current validation rank rather than token-index or population order.
    labels = torch.arange(12)
    h = labels, labels, labels//4, torch.ones(12, dtype=torch.long)
    result, reused = choose_edges(torch.tensor([[1., 0., 0.]]), 5, 2, h,
        torch.tensor([[3, 1, 0, 2]]))
    assert result[0, :2].tolist() == [3, 1]
    assert int(reused) == 2
    s = LayerSharedAVState(coarse_size=8, fine_ratio=.3)
    s.reset(1, 8, 8)
    x = torch.randn(64, 4)
    s.update(x, 0)
    s.update(x.flip(0), 1)
    assert len(s.fine) == 2
    assert torch.equal(s.fine[0][s.fine[1]], torch.arange(len(s.fine[1])))
    with pytest.raises(ValueError): s.update(x, 3)
    s.history = torch.ones(64, 3)
    s.reset(2, 8, 8)
    assert s.fine is None and s.history is None and s.records == []


def test_query_chunks_do_not_share_masks_or_change_selection():
    torch.manual_seed(90)
    labels = torch.arange(120)//3
    reps = torch.arange(40)*3
    groups = torch.arange(40)//8
    h = labels, reps, groups, torch.bincount(labels)
    p = torch.rand(17, 5).softmax(-1)
    history = torch.stack([torch.randperm(120)[:15] for _ in range(17)])
    all_rows, _ = choose_edges(p, 20, 13, h, history)
    chunks = [choose_edges(p[i:i+3], 20, 13, h, history[i:i+3])[0] for i in range(0, 17, 3)]
    assert torch.equal(all_rows, torch.cat(chunks))
    assert not torch.equal(all_rows[0], all_rows[-1])


def test_distinct_validated_edges_in_same_fine_cluster_must_survive():
    labels = torch.arange(30)//3
    reps = torch.arange(10)*3
    groups = torch.arange(10)//5
    h = labels, reps, groups, torch.bincount(labels)
    history = torch.tensor([[2, 1, 0, 4, 2]])  # duplicate token2, not distinct edges
    indices, reused = choose_edges(torch.tensor([[1., 0.]]), 7, 4, h, history)
    assert indices[0, :4].tolist() == [2, 1, 0, 4]
    assert int(reused) == 4
    assert len(indices[0].unique()) == 7
    assert not torch.isin(labels[indices[0, 4:]], labels[indices[0, :4]]).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_normalized_weight_history_reuses_qk():
    from saf3r.triton.hierarchy_attention import indexed_attention_with_logits
    torch.manual_seed(13)
    q = torch.randn(3, 4, 32, device='cuda', dtype=torch.float16)
    k = torch.randn(3, 25, 32, device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    ids = torch.tensor([[1, 4, 6, 8]]*4, device='cuda', dtype=torch.int32)
    specials = torch.tensor([20, 23], device='cuda')
    out, logits, lse = indexed_attention_with_logits(q, k, v, ids, specials, .2, return_lse=True)
    all_ids = torch.cat((ids.long(), specials.expand(4, -1)), -1)
    exact = (q[:, :, None].float()*k[:, all_ids].float()).sum(-1)*.2
    weights = exact.softmax(-1)
    expected = (weights[..., None]*v[:, all_ids].float()).sum(-2)
    torch.testing.assert_close(out.float(), expected, atol=.002, rtol=.002)
    torch.testing.assert_close((logits-lse[..., None]).exp().mean(0), weights[..., :4].mean(0), atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_canonical_coarse_qk_chunk_invariance_cuda():
    from saf3r.patch.layer_shared_av import coarse_probabilities
    torch.manual_seed(50)
    q = torch.randn(1024, 64, device='cuda')
    k = torch.randn(100, 64, device='cuda')
    whole = coarse_probabilities(q, k, 4.)
    chunked = torch.cat([coarse_probabilities(q[i:i+128], k, 4.) for i in range(0, 1024, 128)])
    assert torch.equal(whole, chunked)
