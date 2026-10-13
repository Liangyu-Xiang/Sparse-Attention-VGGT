#!/usr/bin/env python3
"""Find output-optimal sparse supports from dense VGGT attention in one layer.

The selected global layer itself executes densely.  For each patch-query head,
we compare sparse supports at a fixed patch-key retention rate against its exact
dense pre-projection attention output.  Camera/register keys remain available
in every candidate and special-token queries are deliberately not profiled.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from saf3r.models.official_vggt.models.aggregator import slice_expand_and_flatten
from saf3r.models.official_vggt.models.vggt import VGGT
from saf3r.utils.data_utils import load_and_process_images_vggt


NUM_SPECIAL_TOKENS = 5
NUM_HEADS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, required=True, help="Dense global layer to profile (0..2).")
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--sequence", default="seq-01")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--frame-gap", type=int, default=5)
    parser.add_argument("--retention", type=float, default=0.25)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("workspace/benchmark_dataset/7scenes/7Scenes"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.layer <= 2:
        parser.error("This first-stage profiler accepts only --layer 0, 1, or 2.")
    if args.num_frames < 2 or args.frame_gap < 1:
        parser.error("--num-frames must be at least two and --frame-gap positive.")
    if not 0 < args.retention < 1:
        parser.error("--retention must lie in (0, 1).")
    if args.query_chunk_size < 1:
        parser.error("--query-chunk-size must be positive.")
    return args


def frame_path(root: Path, scene: str, sequence: str, frame: int) -> Path:
    return root / scene / sequence / f"frame-{frame:06d}.color.png"


def build_model(checkpoint: Path, device: str) -> VGGT:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
    model = VGGT(save_intermediates=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval()


def run_to_global_layer(model: VGGT, images: torch.Tensor, last_layer: int) -> None:
    """Mirror the VGGT aggregator through the requested alternating global block."""
    aggregator = model.aggregator
    batch, frames, channels_in, height, width = images.shape
    normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
    patch_tokens = aggregator.patch_embed(normalized.view(batch * frames, channels_in, height, width))
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]
    tokens = torch.cat(
        [
            slice_expand_and_flatten(aggregator.camera_token, batch, frames),
            slice_expand_and_flatten(aggregator.register_token, batch, frames),
            patch_tokens,
        ],
        dim=1,
    )
    _, tokens_per_frame, channels = tokens.shape
    position = None
    if aggregator.rope is not None:
        position = aggregator.position_getter(
            batch * frames, height // aggregator.patch_size, width // aggregator.patch_size,
            device=images.device,
        )
        position = position + 1
        special_position = torch.zeros(
            batch * frames, aggregator.patch_start_idx, 2,
            device=images.device, dtype=position.dtype,
        )
        position = torch.cat([special_position, position], dim=1)

    frame_index = global_index = 0
    for _ in range(last_layer + 1):
        tokens, frame_index, _ = aggregator._process_frame_attention(
            tokens, batch, frames, tokens_per_frame, channels, frame_index, pos=position, save_feats=False
        )
        tokens, global_index, _ = aggregator._process_global_attention(
            tokens, batch, frames, tokens_per_frame, channels, global_index, pos=position, save_feats=False
        )


def make_token_indices(
    frames: int, patch_height: int, patch_width: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    patches_per_frame = patch_height * patch_width
    tokens_per_frame = NUM_SPECIAL_TOKENS + patches_per_frame
    special = torch.cat([
        torch.arange(frame * tokens_per_frame, frame * tokens_per_frame + NUM_SPECIAL_TOKENS, device=device)
        for frame in range(frames)
    ])
    patch = torch.cat([
        torch.arange(
            frame * tokens_per_frame + NUM_SPECIAL_TOKENS,
            (frame + 1) * tokens_per_frame,
            device=device,
        )
        for frame in range(frames)
    ])
    grids: list[torch.Tensor] = []
    grid_rows, grid_cols = (patch_height + 1) // 2, (patch_width + 1) // 2
    for corner in range(4):
        row_offset, col_offset = corner // 2, corner % 2
        selected = []
        for frame in range(frames):
            base = frame * tokens_per_frame + NUM_SPECIAL_TOKENS
            for grid_row in range(grid_rows):
                for grid_col in range(grid_cols):
                    row = min(2 * grid_row + row_offset, patch_height - 1)
                    col = min(2 * grid_col + col_offset, patch_width - 1)
                    selected.append(base + row * patch_width + col)
        grids.append(torch.tensor(selected, device=device, dtype=torch.long))
    return special, patch, grids


def sparse_nmse(
    logits: torch.Tensor,
    value: torch.Tensor,
    dense_output: torch.Tensor,
    special_indices: torch.Tensor,
    patch_support: torch.Tensor,
    chunk_size: int,
) -> float:
    """Evaluate exact sparse softmax/output NMSE for a [query, retained-patch] support."""
    queries = logits.shape[0]
    if patch_support.shape[0] != queries:
        raise ValueError("Support must specify one patch-key set per patch query")
    special = special_indices.view(1, -1).expand(queries, -1)
    key_indices = torch.cat([special, patch_support], dim=1)
    squared_error = torch.zeros((), device=logits.device, dtype=torch.float64)
    for start in range(0, queries, chunk_size):
        end = min(queries, start + chunk_size)
        selected = key_indices[start:end]
        scores = logits[start:end].gather(1, selected).float()
        probabilities = scores.softmax(dim=-1)
        selected_values = value[selected].float()
        sparse_output = torch.sum(probabilities.unsqueeze(-1) * selected_values, dim=1)
        squared_error += torch.sum((sparse_output - dense_output[start:end]) ** 2, dtype=torch.float64)
    energy = torch.sum(dense_output ** 2, dtype=torch.float64).clamp_min(1.0e-12)
    return float((squared_error / energy).item())


def value_aware_support(
    patch_attention: torch.Tensor,
    patch_value: torch.Tensor,
    dense_output: torch.Tensor,
    budget: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select keys by first-order output influence a_ij ||v_j - o_i||.

    The query-specific result is an output-aware alternative to QK Top-K.  Its
    mean saliency also yields a reusable query-independent key pattern.
    """
    queries, keys = patch_attention.shape
    supports = torch.empty((queries, budget), dtype=torch.long, device=patch_attention.device)
    aggregate = torch.zeros(keys, dtype=torch.float64, device=patch_attention.device)
    for start in range(0, queries, chunk_size):
        end = min(queries, start + chunk_size)
        residual_norm = torch.linalg.vector_norm(
            patch_value.unsqueeze(0).float() - dense_output[start:end].unsqueeze(1), dim=-1
        )
        score = patch_attention[start:end].float() * residual_norm
        supports[start:end] = score.topk(budget, dim=-1, sorted=False).indices
        aggregate += score.sum(dim=0, dtype=torch.float64)
    return supports, aggregate


class DensePatternProfiler:
    def __init__(
        self,
        layer: int,
        frames: int,
        patch_height: int,
        patch_width: int,
        retention: float,
        chunk_size: int,
        output_dir: Path,
    ) -> None:
        self.layer = layer
        self.frames = frames
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.retention = retention
        self.chunk_size = chunk_size
        self.output_dir = output_dir
        self.rows: list[dict[str, Any]] = []
        self.summary: dict[str, Any] | None = None

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
            raise ValueError("Dense sparse-pattern profiling requires batch size one")
        qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = module.q_norm(query), module.k_norm(key)
        if module.rope is not None:
            query, key = module.rope(query, position), module.rope(key, position)
        query, key, value = query[0], key[0], value[0]
        expected = self.frames * (NUM_SPECIAL_TOKENS + self.patch_height * self.patch_width)
        if tokens != expected:
            raise ValueError(f"Expected {expected} tokens, got {tokens}")
        special, patch, grids = make_token_indices(
            self.frames, self.patch_height, self.patch_width, x.device
        )
        budget = grids[0].numel()
        if budget / patch.numel() < self.retention - 0.02 or budget / patch.numel() > self.retention + 0.02:
            raise ValueError("The 2x2 grid budget is inconsistent with requested retention")
        per_head_best: list[dict[str, Any]] = []
        for head in range(NUM_HEADS):
            q_patch = query[head, patch]
            k_all = key[head]
            v_all = value[head]
            logits = (q_patch.float() @ k_all.float().transpose(0, 1)) * module.scale
            dense_attention = logits.softmax(dim=-1)
            dense_output = dense_attention @ v_all.float()
            patch_attention = dense_attention[:, patch]
            attention_topk = patch_attention.topk(budget, dim=-1, sorted=False).indices
            value_topk, value_score = value_aware_support(
                patch_attention, v_all[patch], dense_output, budget, self.chunk_size
            )
            shared_attention = patch_attention.mean(dim=0).topk(budget, sorted=False).indices
            shared_value = value_score.topk(budget, sorted=False).indices
            candidates: dict[str, torch.Tensor] = {
                "qk_topk_oracle": patch[attention_topk],
                "value_aware_topk_oracle": patch[value_topk],
                "shared_attention_mass": patch[shared_attention].view(1, -1).expand(q_patch.shape[0], -1),
                "shared_value_aware": patch[shared_value].view(1, -1).expand(q_patch.shape[0], -1),
            }
            for corner, grid in enumerate(grids):
                candidates[f"uniform_2x2_corner_{corner}"] = grid.view(1, -1).expand(q_patch.shape[0], -1)
            head_metrics = []
            for candidate, support in candidates.items():
                nmse = sparse_nmse(
                    logits, v_all, dense_output, special, support, self.chunk_size
                )
                record = {
                    "layer": self.layer,
                    "head": head,
                    "candidate": candidate,
                    "patch_keys_per_query": int(budget),
                    "patch_key_retention": float(budget / patch.numel()),
                    "special_keys_always_dense": int(special.numel()),
                    "nmse": nmse,
                }
                self.rows.append(record)
                head_metrics.append(record)
            best = min(head_metrics, key=lambda item: item["nmse"])
            per_head_best.append(best)
            print(
                f"layer {self.layer:02d} head {head:02d}: "
                f"best={best['candidate']} nmse={best['nmse']:.6g}",
                flush=True,
            )
            del logits, dense_attention, dense_output, patch_attention, attention_topk, value_topk, value_score
            torch.cuda.empty_cache()

        candidate_means: dict[str, float] = {}
        for candidate in sorted({row["candidate"] for row in self.rows}):
            candidate_means[candidate] = float(np.mean([row["nmse"] for row in self.rows if row["candidate"] == candidate]))
        self.summary = {
            "layer": self.layer,
            "patch_grid": [self.patch_height, self.patch_width],
            "heads": NUM_HEADS,
            "per_head_optimal_pattern": per_head_best,
            "candidate_mean_nmse": candidate_means,
            "layer_optimal_candidate_by_mean_nmse": min(candidate_means, key=candidate_means.get),
        }


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
    profiler = DensePatternProfiler(
        args.layer, args.num_frames, patch_height, patch_width,
        args.retention, args.query_chunk_size, args.output_dir,
    )
    hook = model.aggregator.global_blocks[args.layer].attn.register_forward_hook(profiler, with_kwargs=True)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            run_to_global_layer(model, images.unsqueeze(0), args.layer)
    finally:
        hook.remove()
    if profiler.summary is None:
        raise RuntimeError("The selected global layer did not execute")
    with (args.output_dir / f"layer_{args.layer:02d}_pattern_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiler.rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(profiler.rows)
    metadata = {
        "scene": args.scene,
        "sequence": args.sequence,
        "frame_indices": frame_indices,
        "retention_requested": args.retention,
        "pattern_protocol": (
            "all special keys remain available; patch-query outputs are compared against exact dense "
            "attention at the same layer/head using output NMSE"
        ),
        "candidate_definitions": {
            "qk_topk_oracle": "per-query top patch keys by dense attention probability",
            "value_aware_topk_oracle": "per-query top a_ij ||v_j - dense_output_i|| patch keys",
            "shared_attention_mass": "one query-independent head mask from mean dense patch attention",
            "shared_value_aware": "one query-independent head mask from mean value-aware saliency",
            "uniform_2x2_corner_k": "one fixed corner from every 2x2 patch grid",
        },
        **profiler.summary,
    }
    (args.output_dir / f"layer_{args.layer:02d}_optimal_patterns.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Saved dense sparse-pattern profile for layer {args.layer} to {args.output_dir}")


if __name__ == "__main__":
    main()
