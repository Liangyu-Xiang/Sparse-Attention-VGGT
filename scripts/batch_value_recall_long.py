#!/usr/bin/env python3
"""Persistent batch, resumable per fit, with machine-readable progress."""
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


SCENES = ["chess", "fire", "heads", "office", "redkitchen"]


def aggregate(root):
    rows = []
    for scene in SCENES:
        with (root / f"{scene}_seq01_n100_s10/neighborhood_recall_1pct/layer_summary.csv").open() as handle:
            rows.extend({"scene": scene, **row} for row in csv.DictReader(handle))
    with (root / "all_sequences_layer_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# VGGT 100帧跨序列 Neighborhood Recall", "",
             "5个序列均使用第0、10、…、990帧；前9个global层；N=103600，K=1036。逐头按ARI选择最佳下一层头；逐层先平均16头V再聚类。下表为8组相邻层的均值。", "",
             "| 序列 | 模式 | R@4 | R@8 | R@16 | R@32 | R@64 |", "|---|---|---:|---:|---:|---:|---:|"]
    for scene in SCENES:
        for mode, label in (("matched_head_mean", "逐头匹配"), ("mean_value", "先平均V")):
            means = []
            for q in (4, 8, 16, 32, 64):
                selected = [float(r["recall"]) for r in rows if r["scene"] == scene and r["mode"] == mode and int(r["q"]) == q]
                means.append(sum(selected) / len(selected))
            lines.append(f"| {scene} | {label} | " + " | ".join(f"{100*v:.2f}%" for v in means) + " |")
    lines += ["", "各序列的 neighborhood_recall_1pct/ 下保存完整逐层、逐头、全部q结果及收敛/FP64核验。",
              "", "聚类算法：" + json.loads((root / 'chess_seq01_n100_s10/metadata.json').read_text())["algorithm"] + "。与此前10帧BUILD+采样best-swap的初始化和交换顺序不同，均不保证全局最优。不同帧数的K也不同，跨帧数比较应同时查看候选token比例。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/value_medoids_100frames"))
    parser.add_argument("--extract-gpu", default="0")
    parser.add_argument("--cpu-workers", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--dependency-path", default="/tmp/saf3r_value_recall_kmedoids")
    parser.add_argument("--backend", choices=["rust-cpu", "torch-gpu", "hybrid"], default="rust-cpu")
    parser.add_argument("--cluster-gpus", nargs="+", default=["4", "5", "6"])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"PYTHONPATH": args.dependency_path + os.pathsep + str(Path.cwd()),
                "OPENBLAS_NUM_THREADS": str(args.cpu_threads), "OMP_NUM_THREADS": str(args.cpu_threads),
                "MKL_NUM_THREADS": str(args.cpu_threads), "PYTHONUNBUFFERED": "1"})
    status = {"state": "extracting", "settings": {"frames": 100, "stride": 10, "layers": 9, "ratio": .01,
              "cpu_workers": args.cpu_workers, "cpu_threads": args.cpu_threads, "backend": args.backend,
              "cluster_gpus": args.cluster_gpus},
              "sequences": {scene: {"state": "pending"} for scene in SCENES}, "pid": os.getpid()}

    def save():
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        for scene in SCENES:
            directory = args.root / f"{scene}_seq01_n100_s10"
            status["sequences"][scene]["completed_fits"] = len(list(directory.glob("keep_1%/layer_*/head_*.json"))) + len(list(directory.glob("head_mean_1pct_ari/layer_*/clustering.json")))
            status["sequences"][scene]["total_fits"] = 153
        temp = args.root / "status.tmp.json"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(args.root / "status.json")

    save()
    for scene in SCENES:
        directory = args.root / f"{scene}_seq01_n100_s10"
        directory.mkdir(exist_ok=True)
        status["sequences"][scene]["state"] = "extracting"
        save()
        print(f"EXTRACT {scene}", flush=True)
        with (directory / "extract.log").open("a") as log:
            result = subprocess.run([sys.executable, "scripts/run_value_recall_long.py", "extract", "--scene", scene,
                                     "--output-dir", str(directory)], env={**env, "CUDA_VISIBLE_DEVICES": args.extract_gpu}, stdout=log, stderr=subprocess.STDOUT)
        status["sequences"][scene]["state"] = "ready" if result.returncode == 0 else "extraction_failed"
        save()
        if result.returncode:
            status["state"] = "failed"
            save()
            raise RuntimeError(f"Extraction failed: {scene}; see {directory/'extract.log'}")
    status["state"] = "clustering"
    pending, active, assignments = list(SCENES), {}, {}
    while pending or active:
        while pending and len(active) < args.cpu_workers:
            scene = pending.pop(0)
            directory = args.root / f"{scene}_seq01_n100_s10"
            available = [gpu for gpu in args.cluster_gpus if gpu not in assignments.values()]
            if args.backend != "rust-cpu" and not available:
                pending.insert(0, scene)
                break
            gpu = available[0] if available else "0"
            assignments[scene] = gpu
            log = (directory / "cluster.log").open("a")
            command = [sys.executable, "scripts/run_value_recall_long.py", "cluster", "--scene", scene,
                       "--output-dir", str(directory), "--backend", args.backend, "--cpu-threads", str(args.cpu_threads), "--include-mean",
                       "--candidate-batch", "128"]
            proc = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": gpu}, stdout=log, stderr=subprocess.STDOUT)
            active[scene] = (proc, log, "clustering")
            status["sequences"][scene].update({"state": "clustering", "worker_pid": proc.pid, "gpu": gpu})
            print(f"CLUSTER {scene} pid={proc.pid}", flush=True)
        for scene, (proc, log, stage) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            directory = args.root / f"{scene}_seq01_n100_s10"
            if code:
                status["sequences"][scene]["state"] = f"{stage}_failed"
                status["state"] = "failed"
                save()
                # Leave other valid workers running; do not terminate results.
                raise RuntimeError(f"{stage} failed: {scene}; inspect logs")
            if stage == "clustering":
                report_log = (directory / "report.log").open("a")
                command = [sys.executable, "scripts/report_value_neighborhood_recall.py", "--input-dir", str(directory),
                           "--matched-only", "--qs", "0", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1035"]
                report_proc = subprocess.Popen(command, env=env, stdout=report_log, stderr=subprocess.STDOUT)
                active[scene] = (report_proc, report_log, "reporting")
                status["sequences"][scene].update({"state": "reporting", "worker_pid": report_proc.pid})
                print(f"REPORT {scene}", flush=True)
            else:
                status["sequences"][scene]["state"] = "complete"
                del active[scene]
                assignments.pop(scene, None)
                print(f"COMPLETE {scene}", flush=True)
        save()
        if active:
            time.sleep(10)
    aggregate(args.root)
    status["state"] = "complete"
    save()
    print(f"ALL COMPLETE: {args.root/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
