"""Distribute independent scene processes, preserving the legacy chess run.

Retire our old serial process ONLY after its complete chess result exists.
Locks, independent status files and one process per GPU prevent duplication.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def completed_result(root, scene, frames):
    data = read_json(Path(root) / scene / "result.json")
    if (data and data.get("scene") == scene and data.get("frames") == frames
            and all(key in data.get("pose", {}) for key in ("auc03", "auc05", "auc15", "auc30"))):
        return data
    return None


def process_identity(pid):
    try:
        proc = Path("/proc") / str(pid)
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"start_ticks": stat[19], "cmdline": (proc / "cmdline").read_bytes().split(b"\0")[:-1]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def validate_legacy(identity):
    if not identity or not any(arg.endswith(b"scripts/run_value_medoid_sparse.py") for arg in identity["cmdline"]):
        raise ValueError("Legacy PID is not the expected Value-PAM experiment")


def idle_gpus(requested):
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    result = []
    for row in raw.splitlines():
        gpu, used, util = map(int, row.split(","))
        if gpu in requested and used < 256 and util == 0:
            result.append(gpu)
    return result


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


class ExistingWorkerProcess:
    """Observe an already-running scene without restarting or reparenting it."""
    def __init__(self, pid, identity):
        self.pid, self.identity = pid, identity

    def poll(self):
        return None if process_identity(self.pid) == self.identity else -1


def validate_scene_worker(identity, scene):
    validate_legacy(identity)
    argv = identity["cmdline"]
    if b"--scenes" not in argv:
        raise ValueError("Cannot adopt a worker without an explicit scene")
    index = argv.index(b"--scenes")
    if index+1 >= len(argv) or argv[index+1] != scene.encode():
        raise ValueError("Scene worker PID does not match requested scene")
    if index+2 < len(argv) and not argv[index+2].startswith(b"--"):
        raise ValueError("Adoption only supports single-scene workers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--legacy-pid", type=int, required=True)
    parser.add_argument("--legacy-gpu", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/value_medoid_sparse_50")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--comparison-root",type=Path,default=ROOT/"outputs/compare_value_sparse_50")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES[:3])
    parser.add_argument("--adopt-status", type=Path, help="Adopt selected live workers from a stopped dispatcher's status")
    args = parser.parse_args()
    if "chess" not in args.scenes or len(set(args.scenes)) != len(args.scenes):
        parser.error("Selected scenes must include legacy chess, without duplicates")
    if args.legacy_gpu in args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("Worker GPUs must be unique and exclude the legacy GPU")
    args.output.mkdir(parents=True, exist_ok=True)
    workers = args.output / "workers"
    workers.mkdir(exist_ok=True)
    controller_lock = (args.output / ".parallel.lock").open("a")
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = process_identity(args.legacy_pid)
    validate_legacy(identity)
    # Save the exact process identity; do not signal a reused PID or another job.
    status = {"status": "running", "frames": args.frames, "controller_pid": os.getpid(),
              "legacy": {"pid": args.legacy_pid, "gpu": args.legacy_gpu, "scene": "chess",
                         "state": "running_preserved", "retire_after": "complete chess result"},
              "jobs": {}, "pending": [], "selected_scenes":args.scenes,
              "updated_unix": time.time()}
    pending = [scene for scene in args.scenes if scene != "chess" and not completed_result(args.output, scene, args.frames)]
    for scene in args.scenes:
        if completed_result(args.output, scene, args.frames):
            status["jobs"][scene] = {"state": "completed_existing"}
    active = {}
    if args.adopt_status:
        previous = read_json(args.adopt_status)
        if not previous:
            raise ValueError("Missing adoption status")
        status["cancelled_scenes"] = previous.get("cancelled_scenes",[])
        for scene in pending.copy():
            job = previous.get("jobs",{}).get(scene,{})
            worker_identity = process_identity(job.get("pid",-1))
            if not worker_identity:
                continue
            validate_scene_worker(worker_identity,scene)
            if job["gpu"] not in args.gpus:
                raise ValueError("Adopted worker GPU is outside selected worker GPUs")
            active[scene] = {"process":ExistingWorkerProcess(job["pid"],worker_identity),
                "gpu":job["gpu"],"status_path":Path(job["status_file"]),
                "log":None,"lock":None,"adopted":True}
            status["jobs"][scene] = {**job,"state":"running","adopted_without_restart":True}
            pending.remove(scene)
            print(f"保留并接管{scene}：GPU {job['gpu']}，PID {job['pid']}，不重启",flush=True)
    legacy_done, retirement_sent = False, False
    last_write, last_report_signature, last_gpu_query = 0., None, -10.
    while pending or active or not legacy_done:
        now = time.monotonic()
        same_legacy = process_identity(args.legacy_pid) == identity
        chess = completed_result(args.output, "chess", args.frames)
        if not legacy_done and chess and same_legacy and not retirement_sent:
            # chess/result.json is written after full model forward and pose
            # scoring. No chess checkpoint/work is discarded by this signal.
            os.kill(args.legacy_pid, signal.SIGTERM)
            retirement_sent = True
            print("chess完整结果已落盘：结束旧串行队列，避免其他序列重复运行", flush=True)
        if not legacy_done and not same_legacy:
            legacy_done = True
            legacy_status = read_json(args.output / "status.json") or {}
            status["legacy"]["state"] = "completed_retired" if chess else "stopped_without_result"
            status["jobs"]["chess"] = {"state": "completed" if chess else "failed_or_stopped", "gpu": args.legacy_gpu}
            if chess and retirement_sent:
                atomic_json(args.output / "legacy_status_at_retirement.json", legacy_status)
                atomic_json(args.output / "status.json", {**legacy_status,
                    "status": "completed_legacy_chess_only", "scenes": ["chess"],
                    "previous_serial_scenes": legacy_status.get("scenes"),
                    "completed_scenes": [chess], "parallel_status": str(args.output / "parallel_status.json")})
            print("旧进程已结束：" + status["legacy"]["state"], flush=True)
        for scene, item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            worker_status = read_json(item["status_path"]) or {}
            state = "completed" if completed_result(args.output, scene, args.frames) else worker_status.get("status", "failed")
            if state in ("running","initializing"):
                state = "failed_process_exited"
            status["jobs"][scene].update(state=state, returncode=None if item.get("adopted") else code,
                                         reason=worker_status.get("reason"))
            if item["log"] is not None:
                item["log"].close()
            if item["lock"] is not None:
                item["lock"].close()
            del active[scene]
            print(f"{scene} / GPU {item['gpu']}：{state}", flush=True)
        occupied = {item["gpu"] for item in active.values()}
        eligible = set(args.gpus) | ({args.legacy_gpu} if legacy_done else set())
        free_gpus = []
        if pending and now-last_gpu_query >= 5:
            free_gpus = idle_gpus(eligible - occupied)
            last_gpu_query = now
        for gpu in free_gpus:
            if not pending:
                break
            scene = pending.pop(0)
            directory = args.output / scene
            directory.mkdir(exist_ok=True)
            scene_lock = (directory / "worker.lock").open("a")
            fcntl.flock(scene_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status_path = workers / f"{scene}_gpu{gpu}.json"
            log_path = workers / f"{scene}_gpu{gpu}.log"
            log = log_path.open("a", buffering=1)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT),
                   "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            command = [sys.executable, str(ROOT / "scripts/run_value_medoid_sparse.py"),
                       "--frames", str(args.frames), "--scenes", scene,
                       "--output", str(args.output), "--status-file", str(status_path)]
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    pass_fds=(scene_lock.fileno(),))
            active[scene] = {"process": proc, "gpu": gpu, "status_path": status_path,
                             "log": log, "lock": scene_lock}
            status["jobs"][scene] = {"state": "running", "gpu": gpu, "pid": proc.pid,
                                      "status_file": str(status_path), "log": str(log_path)}
            print(f"已启动{scene}：GPU {gpu}，PID {proc.pid}", flush=True)
        if now-last_write >= 2:
            status.update(pending=pending.copy(), updated_unix=time.time())
            atomic_json(args.output / "parallel_status.json", status)
            last_write = now
            signature = [scene for scene in args.scenes if completed_result(args.output, scene, args.frames)]
            if signature != last_report_signature:
                subprocess.run([sys.executable, str(ROOT / "scripts/report_value_sparse_comparison.py"),
                                "--root",str(args.comparison_root),"--value-root",str(args.output)], cwd=ROOT, check=False)
                last_report_signature = signature
        time.sleep(.2)
    status.update(status="completed" if all(completed_result(args.output, scene, args.frames) for scene in args.scenes)
                  else "finished_with_incomplete_scenes", pending=[], updated_unix=time.time())
    atomic_json(args.output / "parallel_status.json", status)


if __name__ == "__main__":
    main()
