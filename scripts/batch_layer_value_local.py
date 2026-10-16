#!/usr/bin/env python3
"""Run the layer-shared L3--L9 online experiment on five 100-frame inputs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["7scenes", "eth3d"], default="7scenes")
    parser.add_argument("--scheme", choices=["recursive_local", "saf3r"], default="recursive_local")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--gpus", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.scheme == "saf3r" and args.output_dir is None:
        parser.error("Use a separate --output-dir for SAF3R results")
    if args.output_dir is None:
        args.output_dir = Path("outputs/layer_value_local/eth3d_max100_uniform_q8" if args.dataset == "eth3d"
                               else "outputs/layer_value_local/n100_s10_q8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.scenes:
        scenes = args.scenes
    elif args.dataset == "eth3d":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
        from benchmarks.da3.bench.constants import ETH3D_SCENES
        scenes = list(ETH3D_SCENES)
    else:
        scenes = ["chess", "fire", "heads", "office", "redkitchen"]
    pending, active, assignments = list(scenes), {}, {}
    status = {"state": "running", "dataset": args.dataset, "scheme": args.scheme,
              "topk": args.topk if args.scheme == "saf3r" else None, "max_frames": args.frames,
              "sampling": "uniform" if args.dataset == "eth3d" else "stride",
              "frames": args.frames, "stride": args.stride if args.dataset == "7scenes" else None,
              "retention": .1, "neighbors": 8, "iterations": 2, "pid": os.getpid(),
              "scenes": {s: {"state": "pending"} for s in scenes}}
    if args.scheme == "saf3r":
        for key in ("retention", "neighbors", "iterations"):
            status.pop(key)

    def save():
        temporary = args.output_dir / "status.tmp.json"
        temporary.write_text(json.dumps(status, indent=2)+"\n")
        temporary.replace(args.output_dir / "status.json")

    save()
    while pending or active:
        while pending and len(active) < len(args.gpus):
            scene = pending.pop(0)
            directory = args.output_dir / scene
            if (directory / "result.json").is_file():
                previous = json.loads((directory / "result.json").read_text())
                if args.scheme not in previous or previous["input"].get("dataset", "7scenes") != args.dataset:
                    raise ValueError("Existing result uses a different scheme or dataset")
                status["scenes"][scene]["state"] = "complete"
                continue
            directory.mkdir(exist_ok=True)
            gpu = next(g for g in args.gpus if g not in assignments.values())
            assignments[scene] = gpu
            log = (directory / "run.log").open("a")
            command = [sys.executable, "scripts/run_layer_value_local.py", "--dataset", args.dataset,
                       "--scheme", args.scheme, "--topk", str(args.topk),
                       "--scene", scene, "--frames", str(args.frames),
                       "--stride", str(args.stride), "--output-dir", str(directory), "--seed-backend", "host"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(Path.cwd()),
                   "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[scene] = (process, log)
            status["scenes"][scene].update(state="running", gpu=gpu, pid=process.pid)
            print(f"START {scene} GPU={gpu}", flush=True)
        for scene, (process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                child_status = args.output_dir / scene / "status.json"
                if child_status.is_file():
                    status["scenes"][scene]["stage"] = json.loads(child_status.read_text())["state"]
                continue
            log.close()
            status["scenes"][scene]["state"] = "complete" if code == 0 else "failed"
            if code:
                status["state"] = "failed"
                save()
                raise RuntimeError(f"{scene} exited {code}; inspect its run.log")
            del active[scene]
            assignments.pop(scene)
            print(f"COMPLETE {scene}", flush=True)
        save()
        if active:
            time.sleep(10)
    rows = []
    for scene in scenes:
        result = json.loads((args.output_dir / scene / "result.json").read_text())
        row = {"scene": scene, "frames": len(result["input"]["image_paths"]),
               "depth_change_vs_dense": result["raw_depth_relative_change_vs_dense"]}
        for mode in ("dense", args.scheme):
            row[f"{mode}_seconds"] = result[mode]["seconds_including_clustering_and_dumps"]
            row.update({f"{mode}_{name}": value for name, value in result[mode]["pose"].items()})
        rows.append(row)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status["state"] = "complete"
    save()
    print(f"ALL COMPLETE: {args.output_dir/'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
