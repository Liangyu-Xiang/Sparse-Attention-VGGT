import torch

from saf3r.patch.progressive_grid_attention import (
    ProgressiveGridState,
    progressive_grid_attention,
)


def test_uniform_selection_keeps_one_key_from_every_2x2_grid_at_ratio_one():
    state = ProgressiveGridState(grid_keep_ratio=1.0)
    height, width, images = 5, 7, 2
    count = height * width * images
    selected = state.select(
        "scanning", count, height, width, head_rank=0, device=torch.device("cpu")
    )
    grids = state._grid_ids(selected, height, width)
    expected_grids = images * ((height + 1) // 2) * ((width + 1) // 2)

    assert selected.shape == (count, expected_grids)
    assert selected.stride(0) == 0
    assert torch.all(grids.sort(dim=1).values[:, 1:] != grids.sort(dim=1).values[:, :-1])


def test_same_layer_heads_stagger_grids_before_corners():
    state = ProgressiveGridState(grid_keep_ratio=0.25)
    height, width, images = 4, 4, 1
    count = height * width * images
    selected = [
        state.select(
            "scanning",
            count,
            height,
            width,
            head_rank=rank,
            num_group_heads=4,
            device=torch.device("cpu"),
        )
        for rank in range(4)
    ]
    grid_sets = [set(state._grid_ids(tokens[0], height, width).tolist()) for tokens in selected]

    assert all(len(grid_set) == 1 for grid_set in grid_sets)
    assert len(set.union(*grid_sets)) == 4


def test_exploitation_warms_up_without_a_layer_span_cutoff():
    state = ProgressiveGridState(
        grid_keep_ratio=0.5,
        exploitation_ratio=0.75,
        exploitation_step=0.25,
    )
    height, width, images = 4, 4, 1
    count = height * width * images
    first = state.select("correspondence", count, height, width, head_rank=0, device=torch.device("cpu"))
    scores = torch.arange(count * first.shape[1], dtype=torch.float32).view_as(first)

    assert state.exploitation_ratio_for("correspondence") == 0.0
    expected_schedule = [0.25, 0.50, 0.75, 0.75]
    for expected in expected_schedule:
        state.update("correspondence", [(first, scores)], height, width)
        assert state.exploitation_ratio_for("correspondence") == expected

    # State persists across layers; no maximum-layer-span reset is introduced.
    for _ in range(8):
        state.update("correspondence", [(first, scores)], height, width)
    second = state.select("correspondence", count, height, width, head_rank=0, device=torch.device("cpu"))
    grids = state._grid_ids(second, height, width)

    assert "correspondence" in state._memory
    assert state._depth["correspondence"] == 12
    assert torch.all(grids.sort(dim=1).values[:, 1:] != grids.sort(dim=1).values[:, :-1])


def test_cpu_reference_excludes_special_keys_from_patch_query_softmax():
    height, width, images, special, dim = 2, 2, 2, 2, 4
    tokens_per_image = height * width + special
    tokens = images * tokens_per_image
    q = torch.zeros(1, 1, tokens, dim)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    for image in range(images):
        start = image * tokens_per_image
        v[:, :, start:start + special] = 17.0

    state = ProgressiveGridState(grid_keep_ratio=1.0)
    selected = state.select(
        "position", images * height * width, height, width, head_rank=0, device=torch.device("cpu")
    )
    output = progressive_grid_attention(q, k, v, selected, height, width, special, chunk_size=3)

    image_indices = torch.tensor([2, 3, 4, 5, 8, 9, 10, 11])
    assert output.shape == v.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output[0, 0, image_indices], torch.zeros(len(image_indices), dim))
