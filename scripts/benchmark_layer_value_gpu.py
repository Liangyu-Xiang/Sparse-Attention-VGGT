"""Matched ETH3D benchmark: exact GPU local medoids, cached L2 seed excluded."""
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


def prepare_seed(values, source_directory, fit_directory, retention):
    """Reuse a matching fit, or fit the requested K from the same dense L2 V.

    This entire function runs before any timed model forward. A lower K is
    never obtained by silently truncating the old set of representatives.
    """
    import torch
    from saf3r.patch.value_evolving_attention import LayerValueState
    n, k = len(values), max(1, round(retention*len(values)))
    source_stats = json.loads((source_directory/"clustering.json").read_text())
    if source_stats["n"] != n:
        raise ValueError("Seed feature count mismatch")
    if source_stats["k"] == k:
        location, stats = source_directory, source_stats
    else:
        location = fit_directory
        location.mkdir(parents=True, exist_ok=True)
        if (location/"clustering.json").is_file():
            stats = json.loads((location/"clustering.json").read_text())
            if stats["n"] != n or stats["k"] != k or stats["feature_source"] != str(source_directory.resolve()):
                raise ValueError("Existing initialization has different K or source features")
        else:
            if values.is_cuda:
                torch.cuda.synchronize(values.device)
            started = time.perf_counter()
            labels, medoids, details = LayerValueState(retention=retention, seed_backend="host")._seed(values)
            if values.is_cuda:
                torch.cuda.synchronize(values.device)
            elapsed = time.perf_counter()-started
            stats = {**details, "n": n, "k": k, "retention": k/n,
                     "feature_source": str(source_directory.resolve()),
                     "initialization_seconds_excluded": elapsed}
            np.savez_compressed(location/"clusters.npz", assignments=labels, medoids=medoids)
            (location/"clustering.json").write_text(json.dumps(stats, indent=2)+"\n")
            print("fresh_L2_seed", k, "excluded_seconds", elapsed, flush=True)
    if not stats["converged"]:
        raise ValueError("Initialization did not converge")
    with np.load(location/"clusters.npz") as saved:
        labels, medoids = saved["assignments"], saved["medoids"]
        if labels.shape != (n,) or medoids.shape != (k,) or len(np.unique(medoids)) != k:
            raise ValueError("Invalid initialization cardinality")
        if np.any(labels < 0) or np.any(labels >= k) or np.any(medoids < 0) or np.any(medoids >= n):
            raise ValueError("Invalid initialization indices")
        np.testing.assert_array_equal(labels[medoids], np.arange(k))
        cache = {"assignments": torch.as_tensor(labels.copy(), device=values.device, dtype=torch.long),
                 "medoids": torch.as_tensor(medoids.copy(), device=values.device, dtype=torch.long)}
    return cache, stats


def worker(args):
    import torch
    from scripts.overlay_vggt_query_attention import build_model
    from scripts.run_layer_value_local import load_input
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.patch.value_evolving_attention import install_layer_value_attention
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sys.path.insert(0, str(ROOT/"evaluation"))
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous

    directory = args.output_dir/args.scene
    directory.mkdir(parents=True, exist_ok=True)
    status = {"state": "initializing", "pid": os.getpid(), "scene": args.scene}

    def save():
        temporary = directory/"status.tmp.json"
        temporary.write_text(json.dumps(status, indent=2)+"\n")
        temporary.replace(directory/"status.json")

    save()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        paths, gt, _, _, _ = load_input("eth3d", ROOT/"workspace/benchmark_dataset/eth3d",
                                        args.scene, "", 100, 1)
        source = args.seed_root/args.scene
        old_input = json.loads((source/"input.json").read_text())
        if [str(p.resolve()) for p in paths] != old_input["image_paths"]:
            raise ValueError("Image path/order mismatch against original L2 seed")
        old_settings = json.loads((source/"status.json").read_text())["settings"]
        if Path(old_settings["checkpoint"]).resolve() != args.checkpoint.resolve():
            raise ValueError("Checkpoint mismatch against cached seed")
        seed_directory = source/"selection/forward_0001/layer_02"
        seed_stats = json.loads((seed_directory/"clustering.json").read_text())
        if not seed_stats["converged"]:
            raise ValueError("Cached seed did not converge")
        expected_values = torch.from_numpy(np.load(seed_directory/"mean_values.npy")).cuda()
        status.update(state="fitting_requested_L2_seed", retention=args.retention)
        save()
        cache, seed_stats = prepare_seed(expected_values, seed_directory,
                                         directory/"initialization/layer_02", args.retention)
        images = load_and_process_images_vggt([str(p) for p in paths]).cuda()
        if list(images.shape) != old_input["image_shape"]:
            raise ValueError("Processed image shape mismatch against cached seed")
        if len(cache["assignments"]) != seed_stats["n"] or len(cache["medoids"]) != round(args.retention*seed_stats["n"]):
            raise ValueError("Invalid seed cardinality")
        model = build_model(args.checkpoint, "cuda")
        module_events = []

        def instrument(module, inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            module._benchmark_event = event

        def instrument_end(module, inputs, output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            module_events.append((module._benchmark_event, event))

        def hooks():
            return [handle for i in range(2, 9) for handle in (
                model.aggregator.global_blocks[i].attn.register_forward_pre_hook(instrument),
                model.aggregator.global_blocks[i].attn.register_forward_hook(instrument_end))]

        def event_seconds(events):
            return sum(a.elapsed_time(b) for a, b in events)/1000

        def run(name, state=None):
            status.update(state=name)
            save()
            handles = hooks()
            samples = []
            try:
                for i in range(args.warmup+args.repeats):
                    module_events.clear()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model(images)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    if i == 0 and state is not None:
                        # Outside all measured samples: no stale scene/feature seed reuse.
                        torch.testing.assert_close(state.seed_values, expected_values, rtol=1e-5, atol=1e-7)
                    if i >= args.warmup:
                        samples.append({"seconds": elapsed,
                            "global_l3_9_attention_modules_seconds": event_seconds(module_events),
                            "local_update_seconds": event_seconds(state.selection_events) if state else 0.,
                            "selected_kv_sdpa_seconds": event_seconds(state.attention_events) if state else None,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30})
                    if i != args.warmup+args.repeats-1:
                        del prediction
                pose, depth = prediction["pose_enc"].float().cpu(), prediction["depth"].float().cpu()
                del prediction
                if not torch.isfinite(pose).all() or not torch.isfinite(depth).all():
                    raise ValueError("Non-finite prediction")
                ext, intr = pose_encoding_to_extri_intri(pose, images.shape[-2:])
                metrics = compute_pose(torch.from_numpy(as_homogeneous(ext[0].numpy())), torch.from_numpy(gt))
                np.savez_compressed(directory/f"{name}_poses.npz", extrinsics=ext.numpy(), intrinsics=intr.numpy())
                result = {"samples": samples, "pose": {k: float(v) for k, v in metrics.items()}}
                result.update({f"median_{key}": float(np.median([s[key] for s in samples]))
                               for key in samples[0] if samples[0][key] is not None})
                print(name, json.dumps(result), flush=True)
                return result, depth
            finally:
                for handle in handles:
                    handle.remove()

        dense, dense_depth = run("dense")
        original_attentions = [block.attn for block in model.aggregator.global_blocks]
        state = install_layer_value_attention(model, retention=args.retention, neighbors=8, iterations=2,
                                              local_backend="gpu", seed_cache=cache, profile_cuda=True)
        sparse, sparse_depth = run("gpu_local_excluding_seed", state)
        layers = []
        n, k, d, h = seed_stats["n"], seed_stats["k"], expected_values.shape[1], 16
        special = 5*len(paths)
        for row in state.records[1:]:
            pair_counts = [int(x) for x in row["member_pair_counts"]]
            # Direct difference/square/sum: approximately 3*d FLOPs per distance,
            # excluding sqrt, sorting, gathers, reductions and memory traffic.
            local_flops = 2*d*k*k+3*d*(2*n*(8+2)+sum(pair_counts)+n)
            dense_flops = 4*h*d*(n+special)**2
            layers.append({"layer": row["global_layer_1based"], "n": n, "k": k, "d": d, "heads": h,
                           "member_pair_counts": pair_counts,
                           "distortion": float(row["objective_tensor"])/n,
                           "approx_local_distance_flops": local_flops,
                           "dense_qk_av_flops": dense_flops,
                           "sparse_qk_av_flops": 4*h*d*(n+special)*(k+special),
                           "local_distance_flops_over_dense": local_flops/dense_flops})
        result = {"scene": args.scene, "frames": len(paths), "image_paths": old_input["image_paths"],
                  "image_shape": list(images.shape), "warmup": args.warmup, "repeats": args.repeats,
                  "requested_patch_retention": args.retention, "actual_patch_retention": k/n,
                  "equivalent_connection_retention": (k+special)/(n+special),
                  "initialization": seed_stats,
                  "gpu": torch.cuda.get_device_name(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "dense": dense, "gpu_local_excluding_seed": sparse,
                  "speedup": dense["median_seconds"]/sparse["median_seconds"],
                  "raw_depth_relative_change_vs_dense": float(((sparse_depth-dense_depth).abs()/dense_depth.abs().clamp_min(1e-6)).mean()),
                  "layers": layers,
                  "seed_policy": "Same-scene converged L2 PAM assignments/medoid indices preloaded before timing; verified against actual current L2 V on first warmup. L2 dense attention and all local updates remain timed. Not full cold-start inference.",
                  "precision": "BF16 model; FP32 head mean; FP64 GPU Euclidean local clustering; TF32 disabled",
                  "gpu_local_implementation": "FP64 GEMM Euclidean neighbor distances; topk with exact boundary-tie repair; chunked ragged exhaustive FP64 member medoids",
                  "timing": "Two warmups, five measured forwards by default; paired median wall time with synchronization, no in-forward file dumps. GPU events instrument L3--L9 module, local selection incl mean V, selected-KV gather+SDPA. Normal input/model loading and final metrics/saving excluded."}
        def restore_dense(active_state):
            active_state.reset_hook_handle.remove()
            for block, original in zip(model.aggregator.global_blocks, original_attentions):
                block.attn = original
            delattr(model, "layer_value_state")

        if args.compare_original_value:
            restore_dense(state)
            reference_cache, reference_stats = prepare_seed(expected_values, seed_directory,
                                                            directory/"reference_initialization/layer_02", .1)
            state = install_layer_value_attention(model, retention=.1, neighbors=8, iterations=2,
                                                  local_backend="gpu", seed_cache=reference_cache, profile_cuda=True)
            reference, reference_depth = run("gpu_local_reference_10pct", state)
            result["gpu_local_reference_10pct"] = reference
            result["reference_equivalent_connection_retention"] = (reference_stats["k"]+special)/(n+special)
            result["reference_raw_depth_relative_change_vs_dense"] = float(
                ((reference_depth-dense_depth).abs()/dense_depth.abs().clamp_min(1e-6)).mean())
        if args.compare_saf3r:
            from scripts.run_layer_value_local import install_saf3r_layer39
            from saf3r.patch.patch_vggt import update_args_per_forward
            # Restore dense modules and remove only this installer's reset hook.
            # SAF3R must not accidentally run after a Value-sparse layer 2.
            restore_dense(state)
            result["saf3r_profile"] = install_saf3r_layer39(model, topk=128)
            update_args_per_forward(model, *images.shape[-2:])
            saf3r, saf3r_depth = run("saf3r")
            result["saf3r"] = saf3r
            result["value_layer_seconds_over_saf3r_layer_seconds"] = (
                sparse["median_global_l3_9_attention_modules_seconds"] /
                saf3r["median_global_l3_9_attention_modules_seconds"])
            result["saf3r_raw_depth_relative_change_vs_dense"] = float(
                ((saf3r_depth-dense_depth).abs()/dense_depth.abs().clamp_min(1e-6)).mean())
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
    status = {"state": "running", "pid": os.getpid(), "patch_retention": args.retention,
              "compare_saf3r": args.compare_saf3r, "compare_original_value": args.compare_original_value,
              "warmup": args.warmup, "repeats": args.repeats,
              "scenes": {s: {"state": "pending"} for s in scenes}}

    def save():
        temporary = args.output_dir/"status.tmp.json"
        temporary.write_text(json.dumps(status, indent=2)+"\n")
        temporary.replace(args.output_dir/"status.json")

    save()
    while pending or active:
        for gpu in args.gpus:
            if not pending or gpu in active:
                continue
            scene = pending.pop(0)
            directory = args.output_dir/scene
            directory.mkdir(exist_ok=True)
            if (directory/"result.json").is_file():
                previous = json.loads((directory/"result.json").read_text())
                if previous.get("requested_patch_retention", .1) != args.retention:
                    raise ValueError("Existing result has a different retention; use a separate output directory")
                if args.compare_saf3r and "saf3r" not in previous:
                    raise ValueError("Existing result has no SAF3R comparison; select a separate output directory")
                if args.compare_original_value and "gpu_local_reference_10pct" not in previous:
                    raise ValueError("Existing result has no 10% Value comparison")
                status["scenes"][scene]["state"] = "complete"
                continue
            log = (directory/"run.log").open("a")
            command = [sys.executable, str(Path(__file__).resolve()), "--scene", scene,
                       "--seed-root", str(args.seed_root), "--output-dir", str(args.output_dir),
                       "--checkpoint", str(args.checkpoint), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
            command += ["--retention", str(args.retention)]
            if args.compare_saf3r:
                command.append("--compare-saf3r")
            if args.compare_original_value:
                command.append("--compare-original-value")
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": str(ROOT),
                           "OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (scene, process, log)
            status["scenes"][scene].update(state="running", gpu=gpu, pid=process.pid)
            print("START", scene, gpu, flush=True)
        for gpu, (scene, process, log) in list(active.items()):
            code = process.poll()
            if code is None:
                child_status = args.output_dir/scene/"status.json"
                if child_status.is_file():
                    status["scenes"][scene]["stage"] = json.loads(child_status.read_text())["state"]
                continue
            log.close()
            del active[gpu]
            status["scenes"][scene]["state"] = "complete" if code == 0 else "failed"
            print("END", scene, code, flush=True)
            if code:
                # Keep other authorized scenes running; don't hide individual errors.
                status["state"] = "running_with_failures"
        save()
        if active:
            time.sleep(3)
    rows = []
    for scene in scenes:
        path = args.output_dir/scene/"result.json"
        if not path.is_file():
            continue
        r = json.loads(path.read_text())
        row = {"scene": scene, "frames": r["frames"], "speedup": r["speedup"]}
        for method in ("dense", "gpu_local_excluding_seed"):
            row[method+"_seconds"] = r[method]["median_seconds"]
            row[method+"_l3_9_seconds"] = r[method]["median_global_l3_9_attention_modules_seconds"]
            row.update({method+"_"+k: v for k, v in r[method]["pose"].items()})
        row["local_update_seconds"] = r["gpu_local_excluding_seed"]["median_local_update_seconds"]
        if args.compare_original_value:
            reference = r["gpu_local_reference_10pct"]
            row["reference_10pct_seconds"] = reference["median_seconds"]
            row["reference_10pct_l3_9_seconds"] = reference["median_global_l3_9_attention_modules_seconds"]
            row["reference_10pct_local_update_seconds"] = reference["median_local_update_seconds"]
        if args.compare_saf3r:
            row["saf3r_seconds"] = r["saf3r"]["median_seconds"]
            row["saf3r_l3_9_seconds"] = r["saf3r"]["median_global_l3_9_attention_modules_seconds"]
            row["value_layer_seconds_over_saf3r_layer_seconds"] = r["value_layer_seconds_over_saf3r_layer_seconds"]
        rows.append(row)
    if rows:
        with (args.output_dir/"summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    status["state"] = "complete" if len(rows) == len(scenes) else "failed"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--gpus", nargs="+", default=["1", "2", "3", "4", "5", "6"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compare-saf3r", action="store_true")
    parser.add_argument("--compare-original-value", action="store_true")
    parser.add_argument("--retention", type=float, default=.1)
    parser.add_argument("--seed-root", type=Path, default=ROOT/"outputs/layer_value_local/eth3d_max100_uniform_q8")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/vggt/model_tracker_fixed_e20.pt")
    args = parser.parse_args()
    if args.output_dir is None:
        if args.retention != .1 or args.compare_original_value:
            parser.error("Use a separate --output-dir for changed retention/reference experiments")
        args.output_dir = ROOT/"outputs/layer_value_local"/(
            "eth3d_gpu_value_vs_saf3r_l3_9" if args.compare_saf3r else "eth3d_gpu_local_excluding_seed_gemm")
    if args.warmup < 1 or args.repeats < 1:
        parser.error("At least one warmup and one measured forward required")
    if not 0 < args.retention <= 1:
        parser.error("--retention must lie in (0, 1]")
    worker(args) if args.scene else batch(args)


if __name__ == "__main__":
    main()
