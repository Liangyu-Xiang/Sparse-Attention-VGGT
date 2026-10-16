"""Exact vs previous vs centroid-nearest vs sampled GPU Value representatives.

Shared dense-L2 initialization is excluded. Neighbor construction, source-label
candidate constraints, anchors and two synchronous assignments are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = {"exact": ("exact", 4), "previous": ("previous", 4),
            "centroid": ("centroid", 4), "sampled4": ("sampled", 4), "sampled8": ("sampled", 8)}


def worker(args):
    import torch
    from scripts.benchmark_layer_value_gpu import prepare_seed
    from scripts.run_layer_value_local import load_input
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.patch.value_evolving_attention import install_layer_value_attention
    from saf3r.patch.value_clustering_gpu import recenter_partition_gpu
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sys.path.insert(0, str(ROOT/"evaluation"))
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous

    directory = args.output_dir/args.scene
    directory.mkdir(parents=True, exist_ok=True)
    status = {"state": "initializing", "scene": args.scene, "pid": os.getpid()}
    def save():
        p = directory/"status.tmp.json"
        p.write_text(json.dumps(status, indent=2)+"\n")
        p.replace(directory/"status.json")
    save()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        source = args.seed_root/args.scene
        old = json.loads((source/"input.json").read_text())
        paths, gt, _, _, _ = load_input("eth3d", ROOT/"workspace/benchmark_dataset/eth3d", args.scene, "", 100, 1)
        if [str(p.resolve()) for p in paths] != old["image_paths"]:
            raise ValueError("Image input drift")
        original_settings = json.loads((source/"status.json").read_text())["settings"]
        if Path(original_settings["checkpoint"]).resolve() != args.checkpoint.resolve():
            raise ValueError("Checkpoint drift")
        seed_dir = source/"selection/forward_0001/layer_02"
        expected = torch.from_numpy(np.load(seed_dir/"mean_values.npy")).cuda()
        cache, seed_stats = prepare_seed(expected, seed_dir,
            args.initialization_root/args.scene/"initialization/layer_02", args.retention)
        images = load_and_process_images_vggt([str(p) for p in paths]).cuda()
        if list(images.shape) != old["image_shape"]:
            raise ValueError("Processed shape drift")
        model = build_model(args.checkpoint, "cuda")
        originals = [b.attn for b in model.aggregator.global_blocks]
        methods = {}
        for name in args.variants:
            strategy, candidates = VARIANTS[name]
            status.update(state="timing", variant=name)
            save()
            state = install_layer_value_attention(model, retention=args.retention, neighbors=8, iterations=2,
                local_backend="gpu", seed_cache=cache, profile_cuda=True,
                representative_update=strategy, sampled_candidates=candidates)
            module_events = []
            def before(module, inputs):
                module._timing_start = torch.cuda.Event(enable_timing=True)
                module._timing_start.record()
            def after(module, inputs, output):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                module_events.append((module._timing_start, end))
            handles = [h for i in range(2, 9) for h in (
                model.aggregator.global_blocks[i].attn.register_forward_pre_hook(before),
                model.aggregator.global_blocks[i].attn.register_forward_hook(after))]
            def seconds(events):
                return sum(a.elapsed_time(b) for a, b in events)/1000
            samples = []
            for i in range(args.warmup+args.repeats):
                module_events.clear()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(images)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                if i == 0:
                    torch.testing.assert_close(state.seed_values, expected, rtol=1e-5, atol=1e-7)
                if i >= args.warmup:
                    samples.append({"seconds": elapsed, "l3_9_seconds": seconds(module_events),
                        "local_update_seconds": seconds(state.selection_events),
                        "selected_kv_sdpa_seconds": seconds(state.attention_events),
                        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30})
                if i == args.warmup+args.repeats-1:
                    pose = pred["pose_enc"].float().cpu()
                    if not torch.isfinite(pose).all():
                        raise ValueError("Nonfinite pose")
                del pred
            for handle in handles:
                handle.remove()
            ext, intr = pose_encoding_to_extri_intri(pose, images.shape[-2:])
            metrics = compute_pose(torch.from_numpy(as_homogeneous(ext[0].numpy())), torch.from_numpy(gt))
            np.savez_compressed(directory/f"{name}_poses.npz", extrinsics=ext.numpy(), intrinsics=intr.numpy())
            result = {"representative_update": strategy, "sampled_candidates": candidates if strategy == "sampled" else None,
                      "samples": samples, "pose": {k: float(v) for k, v in metrics.items()}}
            result.update({"median_"+k: float(np.median([row[k] for row in samples])) for k in samples[0]})
            # Separate untimed pass: hold the selected partition fixed and find
            # its exact L2 medoid oracle. Never add this work to runtime samples.
            status.update(state="untimed_partition_oracle", variant=name)
            save()
            state.capture_diagnostics = True
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                diagnostic_prediction = model(images)
            torch.testing.assert_close(diagnostic_prediction["pose_enc"].float().cpu(), pose, rtol=1e-4, atol=1e-4)
            del diagnostic_prediction
            layers = []
            with torch.inference_mode():
                for row in state.records[1:]:
                    x, z, m = row["values_tensor"].double(), row["labels_tensor"], row["medoids_tensor"]
                    torch.testing.assert_close(z[m], torch.arange(len(m), device=z.device))
                    oracle = m if strategy == "exact" else recenter_partition_gpu(x, z, m)[0]
                    exact_d = (x-x[oracle[z]]).norm(dim=1).mean()
                    approx_d = row["objective_tensor"]/len(x)
                    gap = float((approx_d-exact_d)/exact_d.clamp_min(1e-12)*100)
                    if gap < -1e-7:
                        raise ValueError("Approximation beat its same-partition exact oracle beyond tolerance")
                    layers.append({"layer": row["global_layer_1based"], "distortion": float(approx_d),
                        "same_partition_exact_distortion": float(exact_d), "gap_percent": gap})
            result["layers"] = layers
            result["mean_distortion"] = float(np.mean([r["distortion"] for r in layers]))
            result["mean_same_partition_exact_distortion"] = float(np.mean([r["same_partition_exact_distortion"] for r in layers]))
            result["same_partition_gap_percent"] = 100*(result["mean_distortion"]-result["mean_same_partition_exact_distortion"])/max(result["mean_same_partition_exact_distortion"], 1e-12)
            methods[name] = result
            (directory/f"{name}_result.json").write_text(json.dumps(result, indent=2)+"\n")
            print(args.scene, name, result["median_l3_9_seconds"], result["pose"], result["same_partition_gap_percent"], flush=True)
            state.reset_hook_handle.remove()
            for b, original in zip(model.aggregator.global_blocks, originals):
                b.attn = original
            delattr(model, "layer_value_state")
        result = {"scene": args.scene, "frames": len(paths), "image_paths": old["image_paths"],
            "image_shape": list(images.shape), "patch_retention": args.retention,
            "connection_retention": (seed_stats["k"]+5*len(paths))/(len(expected)+5*len(paths)),
            "warmup": args.warmup, "repeats": args.repeats, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "methods": methods,
            "diagnostic_note": "Euclidean (not squared) distortion. Each method's final current partition/current V gets its own exact-medoid oracle outside measured forwards. These gaps isolate representative choice, not differences between methods' recursive partitions/features. Reassignment algorithm is identical; labels may diverge when representatives change.",
            "timing_note": "Fresh process with cached same-scene converged L2 seed, excluded. Two warmups/five repeats by default. No timed oracle diagnostics or file dumps. Module intervals include all local work and QKV/projection."}
        (directory/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        status["state"] = "complete"
    except Exception as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        save()


def batch(args):
    sys.path.insert(0, str(ROOT/"evaluation"))
    from benchmarks.da3.bench.constants import ETH3D_SCENES
    scenes = args.scenes or list(ETH3D_SCENES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending, active = list(scenes), {}
    status = {"state": "running", "pid": os.getpid(), "variants": args.variants,
              "scenes": {s: {"state": "pending"} for s in scenes}}
    def save():
        p = args.output_dir/"status.tmp.json"
        p.write_text(json.dumps(status, indent=2)+"\n")
        p.replace(args.output_dir/"status.json")
    save()
    while pending or active:
        for gpu in args.gpus:
            if not pending or gpu in active:
                continue
            scene = pending.pop(0)
            directory = args.output_dir/scene
            directory.mkdir(exist_ok=True)
            if (directory/"result.json").is_file():
                r = json.loads((directory/"result.json").read_text())
                if r["patch_retention"] != args.retention or set(r["methods"]) != set(args.variants):
                    raise ValueError("Existing experiment uses different settings")
                status["scenes"][scene]["state"] = "complete"
                continue
            log = (directory/"run.log").open("a")
            command = [sys.executable, str(Path(__file__).resolve()), "--scene", scene,
                "--output-dir", str(args.output_dir), "--seed-root", str(args.seed_root),
                "--initialization-root", str(args.initialization_root), "--checkpoint", str(args.checkpoint),
                "--retention", str(args.retention), "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                "--variants", *args.variants]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(ROOT),
                   "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
            p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (scene, p, log)
            status["scenes"][scene].update(state="running", gpu=gpu, pid=p.pid)
            print("START", scene, gpu, flush=True)
        for gpu, (scene, p, log) in list(active.items()):
            code = p.poll()
            child = args.output_dir/scene/"status.json"
            if child.is_file():
                status["scenes"][scene]["stage"] = json.loads(child.read_text())
            if code is None:
                continue
            log.close()
            del active[gpu]
            status["scenes"][scene]["state"] = "complete" if code == 0 else "failed"
            print("END", scene, code, flush=True)
        save()
        if active:
            time.sleep(3)
    rows = []
    for scene in scenes:
        p = args.output_dir/scene/"result.json"
        if not p.is_file():
            continue
        r = json.loads(p.read_text())
        for variant, data in r["methods"].items():
            rows.append({"scene": scene, "variant": variant, "frames": r["frames"],
                **{k: v for k, v in data.items() if k.startswith("median_")},
                **data["pose"], "distortion": data["mean_distortion"],
                "same_partition_exact_distortion": data["mean_same_partition_exact_distortion"],
                "same_partition_gap_percent": data["same_partition_gap_percent"]})
    if rows:
        with (args.output_dir/"summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    status["state"] = "complete" if len(rows) == len(scenes)*len(args.variants) else "failed"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--variants", choices=list(VARIANTS), nargs="+", default=list(VARIANTS))
    parser.add_argument("--retention", type=float, default=.035)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5", "6"])
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/layer_value_local/eth3d_representative_updates_35pct")
    parser.add_argument("--seed-root", type=Path, default=ROOT/"outputs/layer_value_local/eth3d_max100_uniform_q8")
    parser.add_argument("--initialization-root", type=Path, default=ROOT/"outputs/layer_value_local/eth3d_gpu_value_35pct_vs10pct_saf3r_l3_9")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/vggt/model_tracker_fixed_e20.pt")
    args = parser.parse_args()
    if not 0 < args.retention <= 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("Invalid retention/warmup/repeats")
    worker(args) if args.scene else batch(args)


if __name__ == "__main__":
    main()
