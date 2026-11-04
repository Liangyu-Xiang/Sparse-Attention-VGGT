"""Numerical oracle for the optional four-Probe, rather than 32-Query, Grid mass summary.

Run on a CUDA host with PyTorch and Triton:
    python -m pytest -q tests/test_direct_probe_grid_summary.py

This does NOT test geometry quality or model inference speed.
"""
import pytest
import torch

pytest.importorskip("triton")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/Triton required")
def test_grid_summary_sampling_preserves_conditional_denominators():
    from saf3r.triton.reuse_direct_legacy_observation import observe

    torch.manual_seed(20261104)
    device = "cuda"
    heads, groups, rows, dim, regions, grid_size = 2, 2, 32, 32, 8, 16
    queries = groups * rows
    scale = dim ** -0.5
    q = (torch.randn(heads, queries, dim, device=device) * .4).to(torch.float16)
    k = (torch.randn(heads, 144, dim, device=device) * .4).to(torch.float16)

    ids = torch.tensor([
        [2, 5, 7, 16, 18, 20, 35, 38, 40, 56, 62, 65, 80, 96, 97, 99, 112, 117, -1, -1],
        [1, 4, 9, 11, 28, 29, 30, 32, 33, 34, 49, 50, 70, 72, 90, 91, 110, 122, -1, -1],
    ], dtype=torch.int32, device=device)
    num_keys = ids.shape[1]
    valid = ids >= 0
    safe = ids.clamp_min(0).long()
    grid = (safe // grid_size).int()
    sizes = torch.full((regions,), grid_size, dtype=torch.float32, device=device)
    bias = torch.randn(groups, num_keys, device=device) * .2
    positions = torch.tensor([[4, 12, 20, 28], [36, 44, 52, 60]], dtype=torch.int32, device=device)
    weights = torch.full((groups, 4), .25, device=device)

    q4 = q[:, positions.long()]
    means = k[:, :regions * grid_size].reshape(heads, regions, grid_size, dim).float().mean(2)
    base = torch.einsum("hgpd,hrd->ghpr", q4.float().permute(1, 0, 2, 3), means) * scale

    # Reference final full-support LSE: all selected patches (with bias) and Specials.
    query = q.reshape(heads, groups, rows, dim).float()
    selected = k[:, safe].float()
    raw = torch.einsum("hgqd,hgad->hgqa", query, selected) * scale
    biased = (raw + bias[None, :, None, :]).masked_fill(~valid[None, :, None, :], -torch.inf)
    specials = torch.einsum("hgqd,hsd->hgqs", query, k[:, 128:132].float()) * scale
    lse = torch.logsumexp(torch.cat((specials, biased), -1), -1).reshape(heads, queries)

    default, den_default, count_default = observe(
        q, k, ids, grid, bias, positions, lse, base, sizes, scale,
        probe_q=q4,
    )
    sampled, den_sampled, count_sampled = observe(
        q, k, ids, grid, bias, positions, lse, base, sizes, scale,
        probe_q=q4, probe_only_summary=True, probe_weights=weights,
    )
    # The per-Grid v8 conditional denominator is based on the same four Probe
    # logits in BOTH modes. Only the full-query mass summary is approximated.
    torch.testing.assert_close(den_sampled, den_default, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(count_sampled, count_default, atol=0, rtol=0)

    patch_p = torch.exp(biased - lse.reshape(heads, groups, rows, 1))
    patch_p = patch_p.masked_fill(~valid[None, :, None, :], 0.)
    unit_p = patch_p * torch.exp(-bias)[None, :, None, :]

    def grid_aggregate(p, probe_only):
        if probe_only:
            probe_idx = (positions.long() % rows)[None, :, :, None].expand(heads, groups, 4, num_keys)
            p = p.gather(2, probe_idx)
            averaged = (p * weights[None, :, :, None]).sum(2).mean(0)
        else:
            averaged = p.mean((0, 2))
        out = torch.zeros((groups, regions), device=device, dtype=torch.float32)
        return out.scatter_add_(1, grid.long(), averaged)

    expected_full = torch.stack(
        (grid_aggregate(patch_p, False), grid_aggregate(unit_p, False)), dim=1)
    expected_probe = torch.stack(
        (grid_aggregate(patch_p, True), grid_aggregate(unit_p, True)), dim=1)

    torch.testing.assert_close(default, expected_full, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(sampled, expected_probe, atol=3e-3, rtol=3e-3)
    assert torch.isfinite(sampled).all()
    # The approximation is intentional; the test must NOT assert equal summaries.
