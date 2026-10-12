import torch

from saf3r.patch.progressive_grid_attention import (
    ProgressiveGridState,
    progressive_grid_attention,
)


def test_uniform_selection_enforces_one_patch_per_2x2_grid():
    state = ProgressiveGridState(patch_key_budget=10)
    height, width, images = 5, 7, 2
    count = height * width * images
    selected = state.select("scanning", count, height, width, head_rank=0, device=torch.device("cpu"))
    grids = state._grid_ids(selected, height, width)
    assert selected.shape == (count, 10)
    assert torch.all(grids.sort(dim=1).values[:, 1:] != grids.sort(dim=1).values[:, :-1])


def test_history_stays_grid_deduplicated_without_a_layer_span_cutoff():
    state = ProgressiveGridState(patch_key_budget=4, exploitation_ratio=0.75)
    height, width, images = 4, 4, 1
    count = height * width * images
    first = state.select("correspondence", count, height, width, head_rank=0, device=torch.device("cpu"))
    scores = torch.arange(count * 4, dtype=torch.float32).view(count, 4)

    # Simulate updates farther apart than the old eight-layer constraint.  The
    # same group keeps a persistent, bounded state rather than being reset.
    for _ in range(12):
        state.update("correspondence", [(first, scores)], height, width)
    second = state.select("correspondence", count, height, width, head_rank=0, device=torch.device("cpu"))
    grids = state._grid_ids(second, height, width)

    assert "correspondence" in state._memory
    assert state._memory["correspondence"].token_idx.shape == (count, 4)
    assert torch.all(grids.sort(dim=1).values[:, 1:] != grids.sort(dim=1).values[:, :-1])


def test_cpu_reference_keeps_special_tokens_and_returns_finite_output():
    height, width, images, special, dim = 2, 2, 2, 2, 4
    tokens_per_image = height * width + special
    tokens = images * tokens_per_image
    q = torch.randn(1, 1, tokens, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    state = ProgressiveGridState(patch_key_budget=2)
    selected = state.select(
        "position", images * height * width, height, width, head_rank=0, device=torch.device("cpu")
    )
    output = progressive_grid_attention(q, k, v, selected, height, width, special, chunk_size=3)

    assert output.shape == v.shape
    assert torch.isfinite(output).all()
