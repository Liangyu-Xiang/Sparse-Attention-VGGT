#!/usr/bin/env python3
"""Measure output fidelity of uniform 2x2 sampling for one VGGT global layer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.find_vggt_dense_sparse_patterns import (
    NUM_HEADS,
    NUM_SPECIAL_TOKENS,
    build_model,
    frame_path,
    make_token_indices,
    run_to_global_layer,
)
from scripts.find_vggt_iterative_pruning_sparse_attention import dense_output, support_nmse
from saf3r.utils.data_utils import load_and_process_images_vggt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--sequence", default="seq-01")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--frame-gap", type=int, default=5)
    parser.add_argument("--retention", type=float, default=0.25675675675675674)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--data-root", type=Path, default=Path("workspace/benchmark_dataset/7scenes/7Scenes")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.layer < 0:
        parser.error("--layer must be non-negative")
    if args.num_frames < 2 or args.frame_gap < 1:
        parser.error("--num-frames must be at least two and --frame-gap must be positive")
    if not 0 < args.retention < 1:
        parser.error("--retention must lie in (0, 1)")
    return args


class UniformGridProfiler:
    def __init__(
        self,
        layer: int,
        frames: int,
        patch_height: int,
        patch_width: int,
        requested_retention: float,
        chunk_size: int,
    ) -> None:
        self.layer = layer
        self.frames = frames
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.requested_retention = requested_retention
        self.chunk_size = chunk_size
        self.rows: list[dict[str, Any]] = []
        self.per_head: list[dict[str, Any]] = []

    def __call__(
        self,
        module: torch.nn.Module,
        input_args: tuple[Any, ...],
        input_kwargs: dict[str, Any],
        _output: torch.Tensor,
    ) -> None:
        x = input_args[0]
        position = input_kwargs.get("pos")
        batch, tokens, _ = x.shape
        if batch != 1:
            raise ValueError("Uniform-grid profiling requires batch size one")
        qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = module.q_norm(query), module.k_norm(key)
        if module.rope is not None:
            query, key = module.rope(query, position), module.rope(key, position)
        query, key, value = query[0], key[0], value[0]
        expected_tokens = self.frames * (NUM_SPECIAL_TOKENS + self.patch_height * self.patch_width)
        if tokens != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} tokens, received {tokens}")
        special, patch, global_grids = make_token_indices(
            self.frames, self.patch_height, self.patch_width, x.device
        )
        local_grids = [torch.searchsorted(patch, grid) for grid in global_grids]
        patch_keys = patch.numel()
        budget = local_grids[0].numel()
        actual_retention = budget / patch_keys
        if abs(actual_retention - self.requested_retention) > 0.02:
            raise ValueError("The 2x2 grid budget does not match requested retention")

        for head in range(NUM_HEADS):
            query_patch = query[head, patch].float()
            special_logits = (query_patch @ key[head, special].float().transpose(0, 1)) * module.scale
            patch_logits = (query_patch @ key[head, patch].float().transpose(0, 1)) * module.scale
            special_values = value[head, special].float()
            patch_values = value[head, patch].float()
            dense_target = dense_output(
                special_logits, patch_logits, special_values, patch_values, self.chunk_size
            )
            dense_energy = torch.sum(dense_target ** 2, dtype=torch.float64).clamp_min(1.0e-12)
            corner_scores: dict[int, float] = {}
            for corner, grid in enumerate(local_grids):
                support = torch.zeros(patch_keys, device=x.device, dtype=torch.bool)
                support[grid] = True
                nmse = support_nmse(
                    special_logits,
                    patch_logits,
                    special_values,
                    patch_values,
                    support.view(1, -1).expand(query_patch.shape[0], -1),
                    dense_target,
                    dense_energy,
                    self.chunk_size,
                )
                corner_scores[corner] = nmse
                self.rows.append(
                    {
                        "layer": self.layer,
                        "head": head,
                        "corner": corner,
                        "patch_keys_per_query": budget,
                        "patch_key_retention": actual_retention,
                        "special_keys_always_dense": int(special.numel()),
                        "nmse_to_dense": nmse,
                    }
                )
            default_corner = head % 4
            best_corner = min(corner_scores, key=corner_scores.get)
            self.per_head.append(
                {
                    "head": head,
                    "head_rank_modulo_four_corner": default_corner,
                    "head_rank_modulo_four_nmse_to_dense": corner_scores[default_corner],
                    "best_uniform_corner": best_corner,
                    "best_uniform_nmse_to_dense": corner_scores[best_corner],
                    "nmse_by_corner": {str(corner): score for corner, score in corner_scores.items()},
                }
            )
            print(
                f"layer {self.layer:02d} head {head:02d}: "
                f"rank-mod4={corner_scores[default_corner]:.6g} "
                f"best-uniform=c{best_corner}:{corner_scores[best_corner]:.6g}",
                flush=True,
            )
            del special_logits, patch_logits, dense_target
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    frame_indices = [args.first_frame + offset * args.frame_gap for offset in range(args.num_frames)]
    paths = [frame_path(args.data_root, args.scene, args.sequence, index) for index in frame_indices]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input frame(s):\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = load_and_process_images_vggt([str(path) for path in paths]).to(args.device)
    patch_height, patch_width = images.shape[-2] // 14, images.shape[-1] // 14
    model = build_model(args.checkpoint, args.device)
    if args.layer >= len(model.aggregator.global_blocks):
        raise ValueError(f"Layer {args.layer} is out of range for {len(model.aggregator.global_blocks)} global blocks")
    profiler = UniformGridProfiler(
        args.layer,
        args.num_frames,
        patch_height,
        patch_width,
        args.retention,
        args.query_chunk_size,
    )
    hook = model.aggregator.global_blocks[args.layer].attn.register_forward_hook(profiler, with_kwargs=True)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            run_to_global_layer(model, images.unsqueeze(0), args.layer)
    finally:
        hook.remove()
    if not profiler.per_head:
        raise RuntimeError("The selected global layer did not execute")
    metrics_path = args.output_dir / f"layer_{args.layer:02d}_uniform_2x2_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiler.rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(profiler.rows)
    summary = {
        "scene": args.scene,
        "sequence": args.sequence,
        "frame_indices": frame_indices,
        "layer": args.layer,
        "patch_grid": [patch_height, patch_width],
        "method": "uniform_2x2_grid_sampling",
        "special_keys_always_dense": int(args.num_frames * NUM_SPECIAL_TOKENS),
        "per_head": profiler.per_head,
        "mean_head_rank_modulo_four_nmse_to_dense": float(
            np.mean([item["head_rank_modulo_four_nmse_to_dense"] for item in profiler.per_head])
        ),
        "mean_best_uniform_nmse_to_dense": float(
            np.mean([item["best_uniform_nmse_to_dense"] for item in profiler.per_head])
        ),
    }
    summary_path = args.output_dir / f"layer_{args.layer:02d}_uniform_2x2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved uniform 2x2 profile for layer {args.layer} to {args.output_dir}")


if __name__ == "__main__":
    main()
