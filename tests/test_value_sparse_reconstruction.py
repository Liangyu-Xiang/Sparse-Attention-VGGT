import json
import numpy as np
import pytest
import torch

from scripts.evaluate_value_sparse_reconstruction import (
    ReplayMedoidAttention,validate_support,check_replay_poses,strict_sample,
)
from scripts.queue_value_sparse_reconstruction import value_ready
from scripts.report_value_sparse_comparison import complete_reconstruction_means


def test_support_rejects_partial_duplicate_and_out_of_range():
    torch.testing.assert_close(validate_support(np.array([1,5]),8,2),torch.tensor([1,5]))
    for bad in (np.array([1]),np.array([1,1]),np.array([1,8]),np.array([1.,5.])):
        with pytest.raises(ValueError):
            validate_support(bad,8,2)


def test_replay_exactly_matches_original_online_pam_output(tmp_path):
    from saf3r.models.official_vggt.layers.attention import Attention
    from scripts.run_value_medoid_sparse import MeanValueMedoidAttention
    torch.manual_seed(6)
    original = Attention(dim=8,num_heads=2,qk_norm=True).eval()
    online = MeanValueMedoidAttention(original,2,4,0,tmp_path).eval()
    x = torch.randn(1,18,8)
    with torch.inference_mode(),torch.autocast("cpu",dtype=torch.bfloat16):
        expected = online(x)
        with np.load(tmp_path/"layer_00/clusters.npz") as fit:
            support = validate_support(fit["medoids"],8,2)
        replay = ReplayMedoidAttention(original,2,4,support).eval()
        actual = replay(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_pose_replay_check_accepts_batch_dimension_and_rejects_drift(tmp_path):
    ext = np.repeat(np.eye(4,dtype=np.float32)[None,:3],2,axis=0)
    intr = np.repeat(np.eye(3,dtype=np.float32)[None],2,axis=0)
    path = tmp_path/"poses.npz"
    np.savez(path,extrinsics=ext[None],intrinsics=intr[None])
    assert check_replay_poses(path,ext,intr)["extrinsics_max_abs"]==0
    with pytest.raises(AssertionError):
        check_replay_poses(path,ext+.1,intr)


def test_strict_sampling_never_returns_random_cube_for_empty_mesh():
    import open3d as o3d
    with pytest.raises(ValueError,match="Empty"):
        strict_sample(o3d.geometry.TriangleMesh(),100)
    mesh = o3d.geometry.TriangleMesh.create_box()
    first,second = strict_sample(mesh,100),strict_sample(mesh,100)
    np.testing.assert_array_equal(np.asarray(first.points),np.asarray(second.points))


def test_reconstruction_distance_units_and_five_cm_threshold():
    from benchmarks.da3.bench.utils import evaluate_3d_reconstruction
    cloud = np.array([[0.,0,0],[1.,0,0],[2.,0,0]])
    metrics = evaluate_3d_reconstruction(cloud,cloud+np.array([0,.02,0]),threshold=.05)
    assert metrics["acc"]==pytest.approx(.02)
    assert metrics["comp"]==pytest.approx(.02)
    assert metrics["overall"]==pytest.approx(.02)
    assert metrics["fscore"]==1


def test_reconstruction_needs_all_scenes_and_all_nine_supports(tmp_path):
    row = {key:1. for key in ("acc","comp","overall","precision","recall","fscore")}
    assert complete_reconstruction_means([{"method":"dense","scene":"chess",**row}],["chess","fire","heads"])=={}
    directory = tmp_path/"chess"
    directory.mkdir()
    (directory/"result.json").write_text(json.dumps({"scene":"chess"}))
    assert not value_ready(tmp_path,"chess")
    for layer in range(9):
        (directory/f"layer_{layer:02d}").mkdir()
        (directory/f"layer_{layer:02d}/clusters.npz").touch()
    assert value_ready(tmp_path,"chess")


def test_reconstruction_queue_uses_only_available_gpu_and_cpu_scoring(tmp_path,monkeypatch):
    import sys
    import scripts.queue_value_sparse_reconstruction as queue
    root = tmp_path/"comparison"
    root.mkdir()
    (root/"inputs.json").write_text(json.dumps({"scenes":{"chess":{}},"frames":50}))
    for method in ("dense","saf3r"):
        original = root/method/"chess"
        original.mkdir(parents=True)
        (original/"result.json").write_text("{}")
    monkeypatch.setattr(sys,"argv",["queue","--gpus","2","--comparison",str(root),"--value-root",str(tmp_path/"value")])
    monkeypatch.setattr(queue,"value_ready",lambda *a:True)
    monkeypatch.setattr(queue,"idle_gpus",lambda requested:sorted(requested))
    monkeypatch.setattr(queue.subprocess,"run",lambda *a,**kw:None)
    monkeypatch.setattr(queue.time,"sleep",lambda _:None)
    launched = []

    class Worker:
        def __init__(self,cmd,**kwargs):
            stage,method = cmd[cmd.index("--stage")+1],cmd[cmd.index("--method")+1]
            directory = root/"reconstruction"/method/"chess"
            if stage=="capture":
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"]=="2"
                (directory/"capture.json").write_text("{}")
                (directory/"prediction.npz").touch()
            else:
                assert kwargs["env"]["CUDA_VISIBLE_DEVICES"]==""
                (directory/"result.json").write_text("{}")
            launched.append((stage,method))
            self.pid = 100+len(launched)

        def poll(self):
            return 0

    monkeypatch.setattr(queue.subprocess,"Popen",Worker)
    queue.main()
    assert len(launched)==6 and len(set(launched))==6
    status = json.loads((root/"reconstruction/queue_status.json").read_text())
    assert status["status"]=="completed"


def test_queue_marks_failed_value_dependency_instead_of_rerunning_pam(tmp_path,monkeypatch):
    import sys
    import scripts.queue_value_sparse_reconstruction as queue
    root,value = tmp_path/"comparison",tmp_path/"value"
    root.mkdir();value.mkdir()
    (root/"inputs.json").write_text(json.dumps({"scenes":{"chess":{}},"frames":50}))
    (value/"parallel_status.json").write_text(json.dumps({"jobs":{"chess":{"state":"stopped_oom"}}}))
    for method in ("dense","saf3r"):
        directory = root/"reconstruction"/method/"chess"
        directory.mkdir(parents=True)
        (directory/"result.json").write_text("{}")
    monkeypatch.setattr(sys,"argv",["queue","--comparison",str(root),"--value-root",str(value)])
    monkeypatch.setattr(queue.subprocess,"run",lambda *a,**kw:None)
    queue.main()
    status = json.loads((root/"reconstruction/queue_status.json").read_text())
    assert status["jobs"]["value_pam/chess"]["state"]=="blocked_missing_online_value_result"
    assert status["status"]=="finished_with_incomplete_results"
