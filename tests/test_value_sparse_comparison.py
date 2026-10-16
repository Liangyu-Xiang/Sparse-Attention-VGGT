import numpy as np
import pytest

from scripts.benchmark_value_sparse_baselines import make_manifest
from scripts.report_value_sparse_comparison import matched_input, complete_scene_means


def test_manifest_matches_uniform_sampling_and_stairs_sequence(tmp_path):
    for scene, sequence in (("chess", "seq-01"), ("stairs", "seq-02")):
        directory = tmp_path/scene/sequence
        directory.mkdir(parents=True)
        for i in range(9):
            (directory/f"frame-{i:06d}.color.png").touch()
            (directory/f"frame-{i:06d}.pose.txt").touch()
    manifest = make_manifest(tmp_path, ["chess", "stairs"], 4)
    for scene in manifest["scenes"].values():
        assert [int(p.split("frame-")[-1][:6]) for p in scene["paths"]] == np.linspace(0,8,4,dtype=np.int64).tolist()
        assert len(scene["path_digest_sha256"]) == 64
    with pytest.raises(ValueError, match="insufficient"):
        make_manifest(tmp_path, ["chess"], 10)


def test_input_check_rejects_different_order_or_resolution(tmp_path):
    a = {"paths": [str(tmp_path/"a"), str(tmp_path/"b")], "shape": [2,3,392,518]}
    assert matched_input(a, a)
    assert not matched_input(a, {**a,"paths":a["paths"][::-1]})
    assert not matched_input(a, {**a,"shape":[2,3,518,518]})


def test_do_not_average_partial_scenes_or_missing_methods():
    row = {key: 1. for key in ("auc03","auc05","auc15","auc30","latency_seconds","peak_allocated_gib")}
    rows = [{"method":"dense","scene":scene,**row} for scene in ("chess","fire","heads")]
    rows.append({"method":"value_pam","scene":"chess",**row})
    assert list(complete_scene_means(rows,["chess","fire","heads"])) == ["dense"]


def test_report_uses_selected_frame_count_and_output_roots(tmp_path,monkeypatch):
    import json
    import sys
    import scripts.report_value_sparse_comparison as report
    root = tmp_path/"comparison_10"
    root.mkdir()
    scenes = {scene:{"frames":10,"paths":[str(tmp_path/f"{scene}_{i}.png") for i in range(10)]}
              for scene in ("chess","fire","heads")}
    (root/"inputs.json").write_text(json.dumps({"frames":10,"scenes":scenes}))
    monkeypatch.setattr(sys,"argv",["report","--root",str(root),"--value-root",str(tmp_path/"value_10")])
    report.main()
    assert "# 7Scenes：10帧" in (root/"report.md").read_text()
    assert not (tmp_path/"comparison_50").exists()
