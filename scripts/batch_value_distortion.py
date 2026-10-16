#!/usr/bin/env python3
"""Finish 10-frame distortion first, then queue 100-frame multi-sequence fits."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def aggregate_long(root):
    rows = []
    scenes = ["chess", "fire", "heads", "office", "redkitchen"]
    for scene in scenes:
        path = root / f"{scene}_seq01_n100_s10/distortion_comparison_10pct/layer_summary.csv"
        with path.open() as handle:
            rows.extend({"scene": scene, **row} for row in csv.DictReader(handle))
    with (root / "all_sequences_distortion_10pct.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 100帧、5序列：Global / Frozen / Local distortion，10% retention", "",
             "下表为8组相邻层Gap的算术平均，逐层原始distortion与Gap见all_sequences_distortion_10pct.csv；各序列目录有完整要求表。所有距离在当前层V上计算。", "",
             "| 序列 | 模式 | Frozen Gap | q4第1轮Gap | q4第2轮Gap | q8第1轮Gap | q8第2轮Gap |", "|---|---|---:|---:|---:|---:|---:|"]
    for scene in scenes:
        for mode, label in (("matched_head_mean", "逐头匹配"), ("mean_value", "先平均V")):
            selected = [r for r in rows if r["scene"] == scene and r["mode"] == mode]
            keys = ["frozen_gap_pct", "local_q4_iter1_gap_pct", "local_q4_iter2_gap_pct", "local_q8_iter1_gap_pct", "local_q8_iter2_gap_pct"]
            values = [sum(float(r[key]) for r in selected)/len(selected) for key in keys]
            lines.append(f"| {scene} | {label} | " + " | ".join(f"{v:+.2f}%" for v in values) + " |")
    lines += ["", "局部更新为固定原始邻域、固定上一层medoid token归属锚点的两轮同步迭代。Global为充分优化的实现参考，不保证全局最优。所有比较均为独立上一层全局划分出发的单步迁移。"]
    (root / "report_distortion_10pct.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-input", type=Path, default=Path("outputs/value_medoids/chess_n10_s10_layers_00_08"))
    parser.add_argument("--long-root", type=Path, default=Path("outputs/value_medoids_100frames"))
    parser.add_argument("--status-dir", type=Path, default=Path("outputs/value_distortion"))
    parser.add_argument("--small-only", action="store_true")
    args = parser.parse_args()
    args.status_dir.mkdir(parents=True, exist_ok=True)
    logs = args.status_dir / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "/tmp/saf3r_value_recall_kmedoids:"+str(Path.cwd()),
           "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    status = {"state": "small_global_fits", "retention": .10, "local_iterations": 2, "qs": [4, 8],
              "small": {"state": "global_fits", "input": str(args.small_input)},
              "long": {"state": "waiting_for_1pct_recall_batch", "input": str(args.long_root)}, "pid": os.getpid()}

    def save():
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["small"]["completed_global_fits"] = len(list(args.small_input.glob("keep_10%/layer_*/head_*.json"))) + len(list(args.small_input.glob("head_mean_10pct_ari/layer_*/clustering.json")))
        temp = args.status_dir / "status.tmp.json"
        temp.write_text(json.dumps(status, indent=2)+"\n")
        temp.replace(args.status_dir / "status.json")

    def launch(command, name, extra_env=None):
        log = (logs / f"{name}.log").open("a")
        proc = subprocess.Popen(command, env={**env, **(extra_env or {})}, stdout=log, stderr=subprocess.STDOUT)
        return proc, log

    def wait_group(group):
        remaining = list(group)
        while remaining:
            for item in list(remaining):
                proc, log = item
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                if code:
                    status["state"] = "failed"
                    status["failed_pid"] = proc.pid
                    save()
                    raise RuntimeError(f"Child {proc.pid} exited {code}; see {logs}")
                remaining.remove(item)
            save()
            if remaining:
                time.sleep(5)

    save()
    workers = []
    for index in range(9):
        command = [sys.executable, "scripts/run_value_recall_long.py", "cluster", "--output-dir", str(args.small_input),
                   "--backend", "rust-cpu", "--cpu-threads", "2", "--ratio", ".10"]
        command += ["--mean-only"] if index == 8 else ["--heads", str(2*index), str(2*index+1)]
        workers.append(launch(command, f"small_global_{index}"))
    status["small"]["worker_pids"] = [p.pid for p, _ in workers]
    save()
    wait_group(workers)
    status["small"]["state"] = "comparing"
    status["state"] = "small_comparing"
    save()
    wait_group([launch([sys.executable, "scripts/compare_value_distortion.py", "--input-dir", str(args.small_input)], "small_compare")])
    status["small"]["state"] = "complete"
    status["small"]["report"] = str(args.small_input / "distortion_comparison_10pct/report.md")
    status["state"] = "waiting_for_long_recall"
    save()
    print(f"SMALL COMPLETE: {status['small']['report']}", flush=True)
    if args.small_only:
        status["state"] = "complete"
        save()
        return
    while True:
        previous = json.loads((args.long_root / "status.json").read_text())
        if previous["state"] == "complete":
            break
        if previous["state"] == "failed":
            status["state"] = "waiting_for_failed_recall_batch"
            save()
            raise RuntimeError("The existing 100-frame recall batch failed; do not compete for its workers/GPUs")
        save()
        time.sleep(30)
    pending = ["chess", "fire", "heads", "office", "redkitchen"]
    active, assigned = {}, {}
    status["state"] = "long_distortion"
    status["long"]["state"] = "running"
    status["long"]["sequences"] = {s: "pending" for s in pending}
    save()
    while pending or active:
        while pending and len(active) < 3:
            scene = pending.pop(0)
            gpu = next(g for g in ("4", "5", "6") if g not in assigned.values())
            assigned[scene] = gpu
            directory = args.long_root / f"{scene}_seq01_n100_s10"
            item = launch([sys.executable, "scripts/run_value_recall_long.py", "cluster", "--output-dir", str(directory),
                           "--backend", "torch-gpu", "--ratio", ".10", "--include-mean", "--candidate-batch", "128"],
                          f"long_{scene}_global", {"CUDA_VISIBLE_DEVICES": gpu})
            active[scene] = (*item, "global")
            status["long"]["sequences"][scene] = "global_fits"
        for scene, (proc, log, stage) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            if code:
                status["state"] = "failed"
                status["long"]["sequences"][scene] = f"{stage}_failed"
                save()
                raise RuntimeError(f"{scene}/{stage} failed; see {logs}")
            if stage == "global":
                directory = args.long_root / f"{scene}_seq01_n100_s10"
                item = launch([sys.executable, "scripts/compare_value_distortion.py", "--input-dir", str(directory)], f"long_{scene}_compare")
                active[scene] = (*item, "compare")
                status["long"]["sequences"][scene] = "comparing"
            else:
                status["long"]["sequences"][scene] = "complete"
                del active[scene]
                assigned.pop(scene)
        save()
        if active:
            time.sleep(10)
    aggregate_long(args.long_root)
    status["state"] = "complete"
    status["long"]["state"] = "complete"
    save()
    print("ALL DISTORTION EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
