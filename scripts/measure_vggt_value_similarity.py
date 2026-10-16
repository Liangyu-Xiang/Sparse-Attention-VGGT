#!/usr/bin/env python3
"""Measure exact patch-value cosine similarity for one VGGT global-attention layer."""

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
from saf3r.utils.data_utils import load_and_process_images_vggt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--sequence", default="seq-01")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--frame-gap", type=int, default=5)
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
        parser.error("--num-frames must be at least two and --frame-gap positive")
    return args


def pairwise_cosine_statistics(values: torch.Tensor) -> dict[str, float]:
    """Calculate exact off-diagonal cosine moments without materializing an N-by-N matrix."""
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Expected at least two value vectors shaped [keys, channels]")
    norms = torch.linalg.vector_norm(values, dim=1).clamp_min(1.0e-12)
    unit_values = values / norms.unsqueeze(1)
    count = unit_values.shape[0]
    denominator = count * (count - 1)

    # sum_{i != j} u_i dot u_j = ||sum_i u_i||^2 - N.
    mean_cosine = (torch.sum(unit_values, dim=0).square().sum() - count) / denominator
    # sum_{i,j} (u_i dot u_j)^2 is ||U^T U||_F^2; subtract N diagonal terms.
    channel_gram = unit_values.transpose(0, 1) @ unit_values
    mean_squared_cosine = (channel_gram.square().sum() - count) / denominator
    cosine_variance = (mean_squared_cosine - mean_cosine.square()).clamp_min(0.0)
    return {
        "mean_pairwise_cosine": float(mean_cosine.cpu()),
        "pairwise_cosine_std": float(torch.sqrt(cosine_variance).cpu()),
        "mean_pairwise_cosine_distance": float((1.0 - mean_cosine).cpu()),
        "mean_value_l2_norm": float(norms.mean().cpu()),
        "std_value_l2_norm": float(norms.std(unbiased=False).cpu()),
    }


class ValueSimilarityProfiler:
    def __init__(self, layer: int, frames: int, patch_height: int, patch_width: int) -> None:
        self.layer = layer
        self.frames = frames
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.rows: list[dict[str, Any]] = []

    def __call__(
        self,
        module: torch.nn.Module,
        input_args: tuple[Any, ...],
        input_kwargs: dict[str, Any],
        _output: torch.Tensor,
    ) -> None:
        x = input_args[0]
        batch, tokens, _ = x.shape
        if batch != 1:
            raise ValueError("Value-similarity profiling requires batch size one")
        qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        _query, _key, value = qkv.unbind(0)
        value = value[0]
        expected_tokens = self.frames * (NUM_SPECIAL_TOKENS + self.patch_height * self.patch_width)
        if tokens != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} tokens, received {tokens}")
        _special, patch, _ = make_token_indices(
            self.frames, self.patch_height, self.patch_width, x.device
        )
        for head in range(NUM_HEADS):
            statistics = pairwise_cosine_statistics(value[head, patch].float())
            row = {
                "layer": self.layer,
                "head": head,
                "num_patch_values": patch.numel(),
                **statistics,
            }
            self.rows.append(row)
            print(
                f"layer {self.layer:02d} head {head:02d}: "
                f"mean-cos={row['mean_pairwise_cosine']:.6f} "
                f"std={row['pairwise_cosine_std']:.6f}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    frame_indices = [args.first_frame + offset * args.frame_gap for offset in range(args.num_frames)]
    image_paths = [frame_path(args.data_root, args.scene, args.sequence, index) for index in frame_indices]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input frame(s):\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = load_and_process_images_vggt([str(path) for path in image_paths]).to(args.device)
    patch_height, patch_width = images.shape[-2] // 14, images.shape[-1] // 14
    model = build_model(args.checkpoint, args.device)
    if args.layer >= len(model.aggregator.global_blocks):
        raise ValueError(f"Layer {args.layer} is out of range for {len(model.aggregator.global_blocks)} global blocks")
    profiler = ValueSimilarityProfiler(args.layer, args.num_frames, patch_height, patch_width)
    hook = model.aggregator.global_blocks[args.layer].attn.register_forward_hook(profiler, with_kwargs=True)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            run_to_global_layer(model, images.unsqueeze(0), args.layer)
    finally:
        hook.remove()
    if not profiler.rows:
        raise RuntimeError("The selected global layer did not execute")
    metrics_path = args.output_dir / f"layer_{args.layer:02d}_value_similarity.csv"
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
        "method": "exact_patch_value_pairwise_cosine",
        "definition": (
            "All ordered off-diagonal pairs of L2-normalized patch-key value vectors; "
            "special-token values are excluded."
        ),
        "per_head": profiler.rows,
    }
    summary_path = args.output_dir / f"layer_{args.layer:02d}_value_similarity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved value-similarity profile for layer {args.layer} to {args.output_dir}")


if __name__ == "__main__":
    main()
