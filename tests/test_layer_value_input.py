from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_layer_value_local import load_input, install_saf3r_layer39


def test_saf3r_only_wraps_natural_global_layers_3_through_9():
    import torch
    from saf3r.models.official_vggt.layers.attention import Attention
    from saf3r.patch.patch_vggt import HeadwiseSparseAttentionWrapper, update_args_per_forward
    originals = [Attention(dim=32, num_heads=16).eval() for _ in range(24)]
    model = SimpleNamespace(aggregator=SimpleNamespace(global_blocks=[SimpleNamespace(attn=a) for a in originals]))
    info = install_saf3r_layer39(model, topk=128)
    update_args_per_forward(model, 28, 28)
    assert info["configured_topk"] == 128 and not info["topk_effective"]
    assert set(info["profile"]) == {str(i) for i in range(2, 9)}
    for i, block in enumerate(model.aggregator.global_blocks):
        if 2 <= i <= 8:
            assert isinstance(block.attn, HeadwiseSparseAttentionWrapper)
            assert block.attn.orig_module is originals[i]
            assert block.attn.dino_topk == 128
            assert len(block.attn.head_cfg) == 16
            assert block.attn.patch_height == block.attn.patch_width == 2
        else:
            assert block.attn is originals[i]
    # A real tiny 16-head forward exercises the calibrated L3 pattern.
    x = torch.randn(1, 18, 32)
    with torch.inference_mode():
        actual = model.aggregator.global_blocks[2].attn(x)
    assert actual.shape == x.shape and torch.isfinite(actual).all()


def test_eth3d_keeps_calibration_order_and_world_to_camera(monkeypatch):
    gt = np.repeat(np.eye(4)[None], 5, axis=0)
    gt[:, 0, 3] = np.arange(5)

    class Reader:
        def get_data(self, scene):
            assert self.data_root == "/test/eth3d"
            assert scene == "pipes"
            return SimpleNamespace(image_files=[f"/test/{i}.JPG" for i in range(5)], extrinsics=gt)

    monkeypatch.setitem(sys.modules, "benchmarks.da3.bench.datasets.eth3d", SimpleNamespace(ETH3D=Reader))
    paths, actual, indices, available, sampling = load_input(
        "eth3d", Path("/test/eth3d"), "pipes", "ignored", 3, 10)
    assert indices == [0, 2, 4]
    assert paths == [Path(f"/test/{i}.JPG") for i in indices]
    np.testing.assert_array_equal(actual, gt[indices])
    assert (available, sampling) == (5, "uniform")
    all_paths, _, ids, _, _ = load_input("eth3d", Path("/test/eth3d"), "pipes", "", 100, 10)
    assert len(all_paths) == 5 and ids == list(range(5))


def test_sevenscenes_stride_and_pose_inversion_unchanged(tmp_path):
    directory = tmp_path / "chess" / "seq-01"
    directory.mkdir(parents=True)
    gt = np.eye(4)
    gt[0, 3] = 2
    for i in (0, 10):
        (directory / f"frame-{i:06d}.color.png").touch()
        np.savetxt(directory / f"frame-{i:06d}.pose.txt", gt)
    paths, actual, indices, available, sampling = load_input(
        "7scenes", tmp_path, "chess", "seq-01", 2, 10)
    assert len(paths) == 2 and indices == [0, 10]
    np.testing.assert_array_equal(actual, np.repeat(np.linalg.inv(gt)[None], 2, axis=0))
    assert available is None and sampling == "stride"


@pytest.mark.parametrize("frames,stride", [(1, 10), (2, 0)])
def test_invalid_input_settings(frames, stride):
    with pytest.raises(ValueError):
        load_input("eth3d", Path("/unused"), "pipes", "", frames, stride)
