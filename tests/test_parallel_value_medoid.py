import json
import signal
import sys

import pytest

import scripts.parallel_value_medoid_sparse as scheduler


def result(root, scene):
    directory = root/scene
    directory.mkdir(exist_ok=True)
    data = {"scene":scene,"frames":50,"pose":{key:.5 for key in ("auc03","auc05","auc15","auc30")}}
    (directory/"result.json").write_text(json.dumps(data))
    return data


def test_completion_requires_full_valid_result(tmp_path):
    assert scheduler.completed_result(tmp_path,"chess",50) is None
    expected = result(tmp_path,"chess")
    assert scheduler.completed_result(tmp_path,"chess",50) == expected
    assert scheduler.completed_result(tmp_path,"chess",128) is None
    (tmp_path/"chess/result.json").write_text('{"scene":')
    assert scheduler.completed_result(tmp_path,"chess",50) is None


def test_idle_gpu_filter_respects_both_memory_and_utilization(monkeypatch):
    monkeypatch.setattr(scheduler.subprocess,"check_output",lambda *a,**kw:"0, 1, 0\n1, 500, 0\n2, 1, 98\n3, 1, 0\n")
    assert scheduler.idle_gpus({1,2,3}) == [3]


def test_validate_process_refuses_unrelated_pid():
    with pytest.raises(ValueError):
        scheduler.validate_legacy({"cmdline":[b"python",b"another_task.py"]})


def test_dispatch_preserves_chess_until_result_and_does_not_duplicate(tmp_path,monkeypatch):
    launched, signals = [], []
    alive = [True]
    clock = [100.]
    identity = {"start_ticks":"123","cmdline":[b"python",b"scripts/run_value_medoid_sparse.py"]}
    monkeypatch.setattr(sys,"argv",["scheduler","--gpus","2","3","--legacy-pid","987","--output",str(tmp_path),"--scenes",*scheduler.SCENES])
    monkeypatch.setattr(scheduler,"process_identity",lambda pid:identity if alive[0] else None)
    monkeypatch.setattr(scheduler,"idle_gpus",lambda requested:sorted(requested))
    monkeypatch.setattr(scheduler.subprocess,"run",lambda *a,**kw:None)
    monkeypatch.setattr(scheduler.time,"monotonic",lambda:clock[0])

    def sleep(_):
        clock[0] += 5
        result(tmp_path,"chess")

    def kill(pid, sig):
        assert scheduler.completed_result(tmp_path,"chess",50)
        assert (pid,sig)==(987,signal.SIGTERM)
        signals.append(pid)
        alive[0] = False

    class Worker:
        def __init__(self,command,**kwargs):
            scene = command[command.index("--scenes")+1]
            launched.append(scene)
            result(tmp_path,scene)
            self.pid = 900+len(launched)

        def poll(self):
            return 0

    monkeypatch.setattr(scheduler.time,"sleep",sleep)
    monkeypatch.setattr(scheduler.os,"kill",kill)
    monkeypatch.setattr(scheduler.subprocess,"Popen",Worker)
    scheduler.main()
    assert launched == scheduler.SCENES[1:]
    assert signals == [987]
    status = json.loads((tmp_path/"parallel_status.json").read_text())
    assert status["status"] == "completed"
    assert set(status["jobs"]) == set(scheduler.SCENES)
    assert json.loads((tmp_path/"status.json").read_text())["scenes"] == ["chess"]


def test_adopt_only_exact_single_scene_worker():
    identity = {"start_ticks":"1","cmdline":[b"python",b"scripts/run_value_medoid_sparse.py",b"--scenes",b"fire",b"--frames",b"50"]}
    scheduler.validate_scene_worker(identity,"fire")
    with pytest.raises(ValueError,match="does not match"):
        scheduler.validate_scene_worker(identity,"heads")
    with pytest.raises(ValueError,match="single-scene"):
        scheduler.validate_scene_worker({**identity,"cmdline":identity["cmdline"][:4]+[b"heads"]},"fire")


def test_existing_worker_observation_does_not_restart_or_signal(monkeypatch):
    identity = {"start_ticks":"1","cmdline":[b"python"]}
    monkeypatch.setattr(scheduler,"process_identity",lambda pid:identity)
    worker = scheduler.ExistingWorkerProcess(123,identity)
    assert worker.poll() is None
    monkeypatch.setattr(scheduler,"process_identity",lambda pid:None)
    assert worker.poll()==-1
