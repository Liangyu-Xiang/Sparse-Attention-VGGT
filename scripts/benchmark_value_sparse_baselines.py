"""Matched 7Scenes baselines for the online mean-V PAM sparse experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_manifest(data_root, scenes, frames):
    if frames < 2:
        raise ValueError("Pose evaluation requires at least two frames")
    records = {}
    for scene in scenes:
        directory = Path(data_root) / scene / ("seq-02" if scene == "stairs" else "seq-01")
        candidates = [directory / f"frame-{i:06d}.color.png"
                      for i in range(500 if scene == "stairs" else 1000)
                      if (directory / f"frame-{i:06d}.color.png").is_file()
                      and (directory / f"frame-{i:06d}.pose.txt").is_file()]
        if len(candidates) < frames:
            raise ValueError(f"{scene}: insufficient eligible frames, refusing to shorten input")
        indices = np.linspace(0, len(candidates)-1, frames, dtype=np.int64)
        paths = [str(candidates[i].absolute()) for i in indices]
        records[scene] = {"paths": paths, "frames": frames,
                          "path_digest_sha256": hashlib.sha256(json.dumps(paths).encode()).hexdigest()}
    return {"dataset": "7scenes", "sample_mode": "uniform", "frames": frames, "scenes": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("dense", "saf3r"))
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--scenes", nargs="+", default=["chess", "fire", "heads"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "workspace/benchmark_dataset/7scenes/7Scenes")
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/compare_value_sparse_50/inputs.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/compare_value_sparse_50")
    parser.add_argument("--value-root",type=Path,help="Value run root for checking an existing input manifest")
    args = parser.parse_args()
    if args.prepare_only:
        manifest = make_manifest(args.data_root, args.scenes, args.frames)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        # Reject drift against any already-started online Value experiment.
        for scene, entry in manifest["scenes"].items():
            value_root = args.value_root or ROOT/f"outputs/value_medoid_sparse_{args.frames}"
            path = value_root / scene / "input.json"
            if path.exists():
                assert json.loads(path.read_text())["paths"] == entry["paths"], "Value experiment input mismatch"
        print(f"已生成共享{args.frames}帧输入清单：{list(manifest['scenes'])}")
        return
    if args.method is None:
        parser.error("--method is required unless --prepare-only")

    import torch
    from addict import Dict
    from omegaconf import OmegaConf
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sys.path.insert(0, str(ROOT / "evaluation"))
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous

    manifest = json.loads(args.manifest.read_text())
    if manifest["frames"] != args.frames:
        raise ValueError("Frame count does not match input manifest")
    output = args.output / args.method
    output.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "configs/evaluation" / ("eval_vggt.yaml" if args.method == "dense" else "eval_saf3r_vggt.yaml")
    cfg = Dict(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)["model"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    status = {"status": "initializing", "method": args.method, "completed_scenes": [],
              "manifest": str(args.manifest), "model_config": cfg,
              "device": torch.cuda.get_device_name(0),
              "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch": torch.__version__, "precision": "BF16 autocast; FP32 weights; TF32 disabled",
              "timing": "one online forward per scene, synchronized; no warmup; excludes image loading, model loading and pose scoring; includes on-forward selection and JIT compilation",}

    def save_status():
        (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")

    save_status()
    try:
        model = build_model(cfg, "cuda").eval()
        for scene, entry in manifest["scenes"].items():
            directory = output / scene
            directory.mkdir(exist_ok=True)
            paths = entry["paths"]
            images = load_and_process_images_vggt(paths).to("cuda")
            if args.method == "saf3r":
                from saf3r.patch import update_args_per_forward
                update_args_per_forward(model, "vggt", *images.shape[-2:])
            (directory / "input.json").write_text(json.dumps({"paths": paths, "shape": list(images.shape)}, indent=2) + "\n")
            status.update(status="running", current_scene=scene)
            save_status()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(images)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated()/2**30
            peak_reserved = torch.cuda.max_memory_reserved()/2**30
            ext, intr = pose_encoding_to_extri_intri(prediction["pose_enc"], images.shape[-2:])
            ext_np, intr_np = ext[0].float().cpu().numpy(), intr[0].float().cpu().numpy()
            gt = np.stack([np.linalg.inv(np.loadtxt(p.replace(".color.png", ".pose.txt"))) for p in paths])
            metrics = compute_pose(torch.from_numpy(as_homogeneous(ext_np)), torch.from_numpy(gt))
            result = {"method": args.method, "scene": scene, "frames": args.frames,
                      "processed_shape": list(images.shape), "path_digest_sha256": entry["path_digest_sha256"],
                      "latency_seconds": elapsed, "peak_allocated_gib": peak,
                      "peak_reserved_gib": peak_reserved,
                      "pose": {name: float(value) for name, value in metrics.items()}}
            np.savez_compressed(directory / "poses.npz", extrinsics=ext_np, intrinsics=intr_np)
            (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            status["completed_scenes"].append(result)
            save_status()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            del prediction, images, ext, intr
            torch.cuda.empty_cache()
        status["status"] = "completed"
    except torch.cuda.OutOfMemoryError as error:
        status.update(status="stopped_oom", reason=str(error))
        print("显存不足，停止，不改变输入帧数或分辨率", flush=True)
    except Exception as error:
        status.update(status="failed", reason=repr(error))
        raise
    finally:
        save_status()


if __name__ == "__main__":
    main()
