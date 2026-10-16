"""Queue GPU depth captures and independent CPU reconstruction scoring."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.parallel_value_medoid_sparse import read_json,atomic_json,idle_gpus,process_identity


def value_ready(root,scene):
    return ((root/scene/"result.json").is_file()
            and all((root/scene/f"layer_{layer:02d}/clusters.npz").is_file() for layer in range(9)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus",type=int,nargs="+",default=[1,2,3,4,5,6])
    parser.add_argument("--comparison",type=Path,default=ROOT/"outputs/compare_value_sparse_50")
    parser.add_argument("--value-root",type=Path,default=ROOT/"outputs/value_medoid_sparse_50")
    args = parser.parse_args()
    root = args.comparison/"reconstruction"
    root.mkdir(parents=True,exist_ok=True)
    lock = (root/"queue.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest = json.loads((args.comparison/"inputs.json").read_text())
    jobs = {}
    for method in ("dense","saf3r","value_pam"):
        for scene in manifest["scenes"]:
            directory = root/method/scene
            directory.mkdir(parents=True,exist_ok=True)
            state = "waiting_value_result" if method=="value_pam" else "waiting_baseline_result"
            if (directory/"result.json").is_file():
                state = "completed"
            elif (directory/"capture.json").is_file() and (directory/"prediction.npz").is_file():
                state = "waiting_cpu_score"
            jobs[f"{method}/{scene}"] = {"method":method,"scene":scene,"state":state}
    active = {}
    status = {"status":"running","queue_pid":os.getpid(),"jobs":jobs,
              "policy":"no preemption; no use of occupied GPUs; no PAM recomputation; preserve original timing"}
    previous_signature = None
    while True:
        for key,item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            job = jobs[key]
            success = code==0
            if item["stage"]=="capture":
                success &= (root/key/"capture.json").is_file()
                job["state"] = "waiting_cpu_score" if success else "failed_capture"
            else:
                success &= (root/key/"result.json").is_file()
                job["state"] = "completed" if success else "failed_score"
            job["returncode"] = code
            del active[key]
            print(key+"："+job["state"],flush=True)
        parallel = read_json(args.value_root/"parallel_status.json") or {}
        for job in jobs.values():
            if job["state"]=="waiting_baseline_result":
                baseline_root = args.comparison/job["method"]
                if (baseline_root/job["scene"]/"result.json").is_file():
                    job["state"] = "waiting_gpu"
                elif (read_json(baseline_root/"status.json") or {}).get("status") in ("failed","stopped_oom"):
                    job["state"] = "blocked_missing_online_value_result"
            if job["state"]!="waiting_value_result":
                continue
            if value_ready(args.value_root,job["scene"]):
                job["state"] = "waiting_gpu"
            else:
                state = parallel.get("jobs",{}).get(job["scene"],{}).get("state","")
                if (state.startswith(("stopped","failed")) or parallel.get("status")=="finished_with_incomplete_scenes"):
                    job["state"] = "blocked_missing_online_value_result"

        def launch(key,stage,gpu=None):
            job = jobs[key]
            log = (root/key/f"{stage}.log").open("a",buffering=1)
            env = {**os.environ,"CUDA_VISIBLE_DEVICES":"" if gpu is None else str(gpu),
                   "PYTHONPATH":str(ROOT),"PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"4",
                   "OPENBLAS_NUM_THREADS":"4","MKL_NUM_THREADS":"4",
                   "PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}
            command = [sys.executable,str(ROOT/"scripts/evaluate_value_sparse_reconstruction.py"),
                       "--method",job["method"],"--scenes",job["scene"],"--stage",stage,
                       "--comparison",str(args.comparison),"--value-root",str(args.value_root)]
            proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[key] = {"process":proc,"stage":stage,"gpu":gpu,"log":log}
            job.update(state="capturing" if stage=="capture" else "scoring_cpu",pid=proc.pid,gpu=gpu)
            print(f"启动{key}/{stage}，GPU={gpu}",flush=True)

        scoring = sum(item["stage"]=="score" for item in active.values())
        for key,job in jobs.items():
            if job["state"]=="waiting_cpu_score" and scoring<2:
                launch(key,"score")
                scoring += 1
        # Existing PAM dispatcher has priority for its remaining stairs job.
        # Once its queue is empty it will launch no more processes, avoiding
        # a check-then-launch race for the same just-freed GPU.
        dispatcher_pending = (parallel.get("pending") and process_identity(parallel.get("controller_pid",-1)))
        if not dispatcher_pending and any(job["state"]=="waiting_gpu" for job in jobs.values()):
            occupied = {item["gpu"] for item in active.values() if item["gpu"] is not None}
            # A newly launched PAM worker can still be loading its model on
            # CPU and temporarily look idle to nvidia-smi. Honour dispatcher's
            # GPU reservations as well as measured device utilisation.
            occupied.update(job["gpu"] for job in parallel.get("jobs",{}).values()
                            if job.get("state") in ("running","initializing") and "gpu" in job)
            legacy = parallel.get("legacy",{})
            if legacy.get("state","").startswith("running") and "gpu" in legacy:
                occupied.add(legacy["gpu"])
            for gpu in idle_gpus(set(args.gpus)-occupied):
                key = next((key for key,job in jobs.items() if job["state"]=="waiting_gpu"),None)
                if key is None:
                    break
                launch(key,"capture",gpu)
        signature = [key for key,job in jobs.items() if job["state"]=="completed"]
        if signature!=previous_signature:
            subprocess.run([sys.executable,str(ROOT/"scripts/report_value_sparse_comparison.py"),
                            "--root",str(args.comparison),"--value-root",str(args.value_root)],cwd=ROOT,check=False)
            previous_signature = signature
        terminal = all(job["state"] in ("completed","failed_capture","failed_score","blocked_missing_online_value_result") for job in jobs.values())
        status.update(updated_unix=time.time(),status=("completed" if all(job["state"]=="completed" for job in jobs.values())
                                                      else "finished_with_incomplete_results") if terminal else "running")
        atomic_json(root/"queue_status.json",status)
        if terminal:
            break
        time.sleep(3)


if __name__=="__main__":
    main()
