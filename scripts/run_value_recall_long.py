#!/usr/bin/env python3
"""Extract and fit original VGGT V on long, stride-controlled sequences."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from scripts.value_medoids_scalable import blocked_distances, eager_pam, rust_pam, host_distances
from scripts.visualize_vggt_value_medoids import patch_values_from_tokens, valid_cached_cluster


def extract(args):
    from scripts.overlay_vggt_query_attention import build_model, run_to_global_layer
    from saf3r.utils.data_utils import load_and_process_images_vggt
    indices = list(range(0, args.frames * args.stride, args.stride))
    paths = [args.data_root / args.scene / args.sequence / f"frame-{i:06d}.color.png" for i in indices]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} frames: {missing[:5]}")
    if (args.output_dir / "metadata.json").is_file():
        metadata = json.loads((args.output_dir / "metadata.json").read_text())
        if metadata["frame_indices"] == indices and metadata["scene"] == args.scene and all(
                (args.output_dir / f"values/layer_{l}.npy").is_file() for l in range(args.layers)):
            print("Using completed original dense extraction", flush=True)
            return
    images = load_and_process_images_vggt([str(p) for p in paths]).to(args.device)
    grid = [images.shape[-2]//14, images.shape[-1]//14]
    count = int(np.prod(grid))
    directory = args.output_dir / "values"
    directory.mkdir(parents=True, exist_ok=True)
    model = build_model(args.checkpoint, args.device)

    def hook(layer):
        def capture(module, inputs, output):
            x = inputs[0]
            qkv = module.qkv(x).reshape(1, x.shape[1], 3, module.num_heads, module.head_dim)
            values = patch_values_from_tokens(qkv[0, :, 2], args.frames, count)
            np.save(directory / f"layer_{layer}.npy", values.float().cpu().numpy())
            print(f"Original dense global layer {layer+1}: {tuple(values.shape)}", flush=True)
        return capture

    handles = [model.aggregator.global_blocks[l].attn.register_forward_hook(hook(l)) for l in range(args.layers)]
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            run_to_global_layer(model, images.unsqueeze(0), args.layers-1)
    finally:
        for handle in handles:
            handle.remove()
    metadata = {"scene": args.scene, "sequence": args.sequence, "frame_indices": indices,
                "image_paths": [str(p.resolve()) for p in paths], "processed_image_shape": list(images.shape),
                "patch_grid": grid, "special_tokens_excluded": True, "global_layer_indices": list(range(args.layers)),
                "num_frames": args.frames, "values_per_head": args.frames*count, "head_dim": 64,
                "distance": "euclidean_raw_values", "checkpoint": str(args.checkpoint.resolve()),
                "values_storage": "per_layer_npy", "precision": "Original dense BF16 autocast; clustering FP32, objective FP64",
                "algorithm": "Two seeded distance-weighted starts; exhaustive eager PAM until no improving single swap"}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def cluster(args):
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    algorithm = ("Two seeded-random kmedoids 0.5.5 FasterPAM starts, best objective; FastPAM1 exhaustive no-improving-swap certificate"
                             if args.backend == "rust-cpu" else
                             "Two seeded distance-weighted eager PAM starts, best objective; exhaustive no-improving-swap final pass")
    metadata.setdefault("clustering_algorithms", {})[f"{args.ratio:.0%}"] = algorithm
    if "source_metadata" in metadata and args.ratio != .01:
        metadata["algorithm"] = "PAM exhaustive best single swap"
    temporary = args.output_dir / f"metadata.{os.getpid()}.tmp.json"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(args.output_dir / "metadata.json")
    n = metadata["values_per_head"]
    k = round(args.ratio*n)
    for layer in range(args.layers):
        if (args.output_dir / "values").is_dir():
            raw = np.load(args.output_dir / f"values/layer_{layer}.npy", mmap_mode="r")
        else:
            with np.load(args.output_dir / "values.npz") as cache:
                raw = cache[f"layer_{layer}"]
        for head in ([] if args.mean_only else list(args.heads)) + ([-1] if args.include_mean or args.mean_only else []):
            name = f"head_{head:02d}" if head >= 0 else "clusters"
            directory = (args.output_dir / f"keep_{args.ratio:.0%}/layer_{layer:02d}" if head >= 0 else
                         args.output_dir / f"head_mean_{round(100*args.ratio)}pct_ari/layer_{layer:02d}")
            stats_name = name if head >= 0 else "clustering"
            if valid_cached_cluster(directory / f"{name}.npz", directory / f"{stats_name}.json", n, k, args.ratio):
                print(f"Resume layer={layer} head={head}", flush=True)
                continue
            start = time.perf_counter()
            values = np.array(raw[head]) if head >= 0 else raw.mean(0, dtype=np.float32)
            print(f"START layer={layer} head={head} N={n} K={k}", flush=True)
            distance = (host_distances(torch.from_numpy(values), args.device) if args.backend == "hybrid" else
                        blocked_distances(torch.from_numpy(values).to("cpu" if args.backend == "rust-cpu" else args.device)))
            print(f"Distance matrix ready in {time.perf_counter()-start:.1f}s", flush=True)
            results = [(rust_pam(distance, k, seed=20261015+100*layer+head+10000*trial,
                                  max_passes=args.max_passes, cpu_threads=args.cpu_threads)
                        if args.backend == "rust-cpu" else
                        eager_pam(distance, k, seed=20261015+100*layer+head+10000*trial,
                                 max_passes=args.max_passes, candidate_batch=args.candidate_batch, verbose=True,
                                 compute_device=args.device if args.backend == "hybrid" else None)
                        )
                       for trial in range(args.starts)]
            result = min(results, key=lambda r: r["objective"])
            if not result["converged"]:
                raise RuntimeError(f"Selected fit has not converged: layer={layer} head={head}")
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(directory / f"{name}.npz", **{key: result[key] for key in ("medoids", "assignments", "distances")},
                                cluster_sizes=np.bincount(result["assignments"], minlength=k))
            stats = {key: value for key, value in result.items() if key not in ("medoids", "assignments", "distances")}
            stats.update({"layer": layer, "head": head, "n": n, "k": k, "requested_retention": args.ratio,
                          "elapsed_seconds": time.perf_counter()-start,
                          "starts": [{key: value for key, value in r.items() if key not in ("medoids", "assignments", "distances")} for r in results]})
            (directory / f"{stats_name}.json").write_text(json.dumps(stats, indent=2) + "\n")
            print(f"DONE layer={layer} head={head} objective={result['objective']:.6f} seconds={stats['elapsed_seconds']:.1f}", flush=True)
            del distance, results, result
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["extract", "cluster"])
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--sequence", default="seq-01")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--layers", type=int, default=9)
    parser.add_argument("--data-root", type=Path, default=Path("workspace/benchmark_dataset/7scenes/7Scenes"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--heads", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--include-mean", action="store_true")
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--candidate-batch", type=int, default=32)
    parser.add_argument("--max-passes", type=int, default=100)
    parser.add_argument("--backend", choices=["torch-gpu", "rust-cpu", "hybrid"], default="torch-gpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=.01)
    parser.add_argument("--mean-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    {"extract": extract, "cluster": cluster}[args.stage](args)


if __name__ == "__main__":
    main()
