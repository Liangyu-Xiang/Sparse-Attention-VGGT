#!/usr/bin/env python3
"""End-to-end VGGT dense vs recursive layer-shared Value sparse attention."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from saf3r.patch.value_evolving_attention import install_layer_value_attention


def install_saf3r_layer39(model, topk=128):
    """Keep the calibrated SAF3R patterns, patch only natural global L3--L9."""
    from saf3r.patch.patch_vggt import HeadwiseSparseAttentionWrapper
    profile_path = Path(__file__).resolve().parents[1] / "configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json"
    profile = json.loads(profile_path.read_text())
    selected = {str(i): profile[str(i)] for i in range(2, 9)}
    # This matched runner intentionally avoids changing the aggregator forward.
    # The current calibrated L3--L9 profile has no correspondence/DINO heads.
    if any(g["mode"] == "dino_topk" for groups in selected.values() for g in groups):
        raise ValueError("Profile changed: DINO Top-K requires its index-sharing aggregator path")
    import copy
    for i in range(2, 9):
        original = model.aggregator.global_blocks[i].attn
        model.aggregator.global_blocks[i].attn = HeadwiseSparseAttentionWrapper(
            original, layer_idx=i, dino_topk=topk, lazy_dino_topk=True,
            config=copy.deepcopy(selected[str(i)])).eval()
    return {"profile_path": str(profile_path), "profile": selected,
            "configured_topk": topk, "topk_effective": False,
            "topk_note": "topk controls DINO/correspondence heads only; L3--L9 has none. Query-probe budget is controlled by profile stride, not this topk."}


def load_input(dataset, data_root, scene, sequence, frames, stride):
    """Return matched image paths and world-to-camera GT in dataset order."""
    if frames < 2 or stride < 1:
        raise ValueError("Need at least two frames and a positive stride")
    if dataset == "eth3d":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
        from benchmarks.da3.bench.datasets.eth3d import ETH3D
        reader = ETH3D()
        reader.data_root = str(data_root)
        data = reader.get_data(scene)
        count = len(data.image_files)
        if count < 2:
            raise ValueError(f"ETH3D/{scene}: fewer than two valid calibrated images")
        # ETH3D is an unordered calibrated image collection, not a video.
        # Use the benchmark's uniform sampling rather than a video stride.
        indices = np.linspace(0, count-1, min(frames, count), dtype=int).tolist()
        paths = [Path(data.image_files[i]) for i in indices]
        return paths, np.asarray(data.extrinsics)[indices], indices, count, "uniform"
    indices = list(range(0, frames*stride, stride))
    paths = [data_root / scene / sequence / f"frame-{i:06d}.color.png" for i in indices]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("Requested stride-controlled input contains missing frames")
    gt = np.stack([np.linalg.inv(np.loadtxt(str(p).replace(".color.png", ".pose.txt"))) for p in paths])
    return paths, gt, indices, None, "stride"


def main():
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["7scenes", "eth3d"], default="7scenes")
    parser.add_argument("--scheme", choices=["recursive_local", "saf3r"], default="recursive_local")
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--sequence", default="seq-01")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--retention", type=float, default=.10)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed-backend", choices=["device", "host"], default="host")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/layer_value_local/chess_n10_s10_q8"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.scheme == "saf3r" and args.topk < 1:
        parser.error("--topk must be positive")
    if args.data_root is None:
        args.data_root = Path("workspace/benchmark_dataset/eth3d" if args.dataset == "eth3d"
                              else "workspace/benchmark_dataset/7scenes/7Scenes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = {"state": "initializing", "pid": os.getpid(), "settings": vars(args).copy()}
    status["settings"] = {key: str(value) if isinstance(value, Path) else value for key, value in status["settings"].items()}

    def save_status():
        temp = args.output_dir / "status.tmp.json"
        temp.write_text(json.dumps(status, indent=2)+"\n")
        temp.replace(args.output_dir / "status.json")

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    save_status()
    try:
        paths, gt, indices, available, sampling = load_input(
            args.dataset, args.data_root, args.scene, args.sequence, args.frames, args.stride)
        images = load_and_process_images_vggt([str(p) for p in paths]).to(args.device)
        shape = tuple(images.shape[-2:])
        metadata = {"dataset": args.dataset, "scene": args.scene,
                    "sequence": args.sequence if args.dataset == "7scenes" else None,
                    "frame_indices": indices, "sampling": sampling,
                    "available_calibrated_images": available,
                    "image_paths": [str(p.resolve()) for p in paths], "image_shape": list(images.shape),
                    "scheme": args.scheme,
                    "global_seed_layer": 2 if args.scheme == "recursive_local" else None,
                    "sparse_layers": list(range(3, 10)),
                    "dense_layers": [1, 2]+list(range(10, 25)), "retention": args.retention,
                    "neighbors": args.neighbors, "iterations": args.iterations,
                    "special_queries": "use the same selected patch keys plus all special keys as every other query",
                    "head_clustering": "one mean raw V partition per layer; no head matching"}
        if args.scheme == "saf3r":
            for key in ("retention", "neighbors", "iterations", "special_queries", "head_clustering"):
                metadata.pop(key)
            metadata["attention"] = "original SAF3R per-head patterns at global L3--L9; all other attention unchanged"
        (args.output_dir / "input.json").write_text(json.dumps(metadata, indent=2)+"\n")
        model = build_model(args.checkpoint, args.device)

        def infer(name):
            status["state"] = name
            save_status()
            if images.is_cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(images.device)
                torch.cuda.synchronize(images.device)
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
                prediction = model(images)
            if images.is_cuda:
                torch.cuda.synchronize(images.device)
            seconds = time.perf_counter()-start
            pose = prediction["pose_enc"].float().cpu()
            depth = prediction["depth"].float().cpu()
            if not torch.isfinite(pose).all() or not torch.isfinite(depth).all():
                raise RuntimeError(f"Non-finite {name} predictions")
            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose, shape)
            np.savez_compressed(args.output_dir / f"{name}_predictions.npz", pose_enc=pose.numpy(),
                                extrinsics=extrinsics.numpy(), intrinsics=intrinsics.numpy(), depth=depth.numpy())
            metrics = compute_pose(torch.from_numpy(as_homogeneous(extrinsics[0].numpy())), torch.from_numpy(gt))
            result = {"seconds_including_clustering_and_dumps": seconds,
                      "pose": {key: float(value) for key, value in metrics.items()},
                      "peak_allocated_gib": torch.cuda.max_memory_allocated(images.device)/2**30 if images.is_cuda else None}
            del prediction
            print(f"{name}: {json.dumps(result)}", flush=True)
            return result, depth

        dense_result, dense_depth = infer("dense")
        (args.output_dir / "dense_result.json").write_text(json.dumps(dense_result, indent=2)+"\n")
        if args.scheme == "recursive_local":
            state = install_layer_value_attention(model, retention=args.retention, neighbors=args.neighbors,
                                                  iterations=args.iterations, seed_backend=args.seed_backend,
                                                  output_dir=args.output_dir / "selection", save_values=True)
        else:
            from saf3r.patch.patch_vggt import update_args_per_forward
            metadata["saf3r"] = install_saf3r_layer39(model, args.topk)
            update_args_per_forward(model, *shape)
            (args.output_dir / "input.json").write_text(json.dumps(metadata, indent=2)+"\n")
        sparse_result, sparse_depth = infer(args.scheme)
        if args.scheme == "recursive_local":
            assert [r["global_layer_1based"] for r in state.records] == list(range(2, 10))
            assert [r["source_layer_1based"] for r in state.records] == [None]+list(range(2, 9))
        relative = (sparse_depth-dense_depth).abs()/dense_depth.abs().clamp_min(1e-6)
        result = {"input": metadata, "dense": dense_result, args.scheme: sparse_result,
                  "raw_depth_relative_change_vs_dense": float(relative.mean()),
                  "layer_selection": state.records if args.scheme == "recursive_local" else [],
                  "timing_note": "Single-run synchronized inference, no warmup, including online selection and any JIT/diagnostics; not an attention-kernel benchmark"}
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        status["state"] = "complete"
        status["result"] = str(args.output_dir / "result.json")
    except Exception as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        save_status()


if __name__ == "__main__":
    main()
