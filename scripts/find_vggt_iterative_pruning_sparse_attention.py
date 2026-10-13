#!/usr/bin/env python3
"""Find per-query sparse supports by iterative output-preserving pruning.

For every patch-query attention row, this profiler starts with all patch keys.
At each retention stage it recomputes the currently normalized attention and
removes the patch keys with the smallest exact single-key deletion cost:

    a_j / (1 - a_j) * ||v_j - o||_2.

Special keys remain available throughout.  The implementation deletes a batch
per stage (100 -> 75 -> 50 -> 37.5 -> target%) rather than one key at a time;
therefore it is a batched greedy optimizer, not an exhaustive global search.
"""

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
    parser.add_argument("--layer", type=int, required=True, help="Global layer to profile (0..2).")
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--sequence", default="seq-01")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--frame-gap", type=int, default=5)
    parser.add_argument(
        "--retention",
        type=float,
        default=0.25675675675675674,
        help="Final patch-key retention.  Default matches one key per 2x2 cell on a 28x37 grid.",
    )
    parser.add_argument(
        "--prune-stages",
        type=float,
        nargs="*",
        default=[0.75, 0.50, 0.375],
        help="Intermediate patch-key retention fractions before the final target.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("workspace/benchmark_dataset/7scenes/7Scenes"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
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
    if any(not args.retention < stage < 1 for stage in args.prune_stages):
        parser.error("Every --prune-stages value must lie strictly between retention and 1.")
    if args.prune_stages != sorted(args.prune_stages, reverse=True):
        parser.error("--prune-stages must be strictly descending.")
    return args


def patch_counts(total_patch_keys: int, retention: float, stages: list[float]) -> list[int]:
    """Return strictly decreasing retained-key counts, ending at target retention."""
    counts = [int(round(total_patch_keys * stage)) for stage in stages]
    counts.append(int(round(total_patch_keys * retention)))
    counts = [min(total_patch_keys - 1, max(1, count)) for count in counts]
    if any(left <= right for left, right in zip(counts, counts[1:])):
        raise ValueError(f"Pruning schedule must strictly decrease, received {counts}")
    return counts


def attention_output(
    special_logits: torch.Tensor,
    patch_logits: torch.Tensor,
    special_values: torch.Tensor,
    patch_values: torch.Tensor,
    patch_kept: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return output and patch probabilities under the current key support."""
    if patch_kept is None:
        active_patch_logits = patch_logits
    else:
        active_patch_logits = patch_logits.masked_fill(~patch_kept, float("-inf"))
    probabilities = torch.cat([special_logits, active_patch_logits], dim=1).softmax(dim=-1)
    special_count = special_values.shape[0]
    output = probabilities[:, :special_count] @ special_values
    output = output + probabilities[:, special_count:] @ patch_values
    return output, probabilities[:, special_count:]


def dense_output(
    special_logits: torch.Tensor,
    patch_logits: torch.Tensor,
    special_values: torch.Tensor,
    patch_values: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    outputs = torch.empty(
        (patch_logits.shape[0], patch_values.shape[1]),
        device=patch_logits.device,
        dtype=torch.float32,
    )
    for start in range(0, patch_logits.shape[0], chunk_size):
        end = min(start + chunk_size, patch_logits.shape[0])
        outputs[start:end], _ = attention_output(
            special_logits[start:end], patch_logits[start:end], special_values, patch_values, None
        )
    return outputs


def prune_stage(
    special_logits: torch.Tensor,
    patch_logits: torch.Tensor,
    special_values: torch.Tensor,
    patch_values: torch.Tensor,
    patch_kept: torch.Tensor,
    dense_target: torch.Tensor,
    dense_energy: torch.Tensor,
    remove_per_query: int,
    chunk_size: int,
) -> tuple[float, float]:
    """Score the current support, then remove its lowest-cost patch keys."""
    error = torch.zeros((), device=patch_logits.device, dtype=torch.float64)
    removed_cost = torch.zeros((), device=patch_logits.device, dtype=torch.float64)
    for start in range(0, patch_logits.shape[0], chunk_size):
        end = min(start + chunk_size, patch_logits.shape[0])
        current_kept = patch_kept[start:end]
        output, patch_probabilities = attention_output(
            special_logits[start:end],
            patch_logits[start:end],
            special_values,
            patch_values,
            current_kept,
        )
        error += torch.sum((output - dense_target[start:end]) ** 2, dtype=torch.float64)
        value_distance = torch.linalg.vector_norm(
            patch_values.unsqueeze(0) - output.unsqueeze(1), dim=-1
        )
        removal_cost = (
            patch_probabilities / (1.0 - patch_probabilities).clamp_min(1.0e-12)
        ) * value_distance
        removal_cost = removal_cost.masked_fill(~current_kept, float("inf"))
        chosen_cost, chosen_indices = removal_cost.topk(
            remove_per_query, dim=-1, largest=False, sorted=False
        )
        current_kept.scatter_(
            1,
            chosen_indices,
            torch.zeros_like(chosen_indices, dtype=torch.bool),
        )
        removed_cost += torch.sum(chosen_cost, dtype=torch.float64)
    denominator = patch_logits.shape[0] * remove_per_query
    return float((error / dense_energy).item()), float((removed_cost / denominator).item())


def support_nmse(
    special_logits: torch.Tensor,
    patch_logits: torch.Tensor,
    special_values: torch.Tensor,
    patch_values: torch.Tensor,
    patch_kept: torch.Tensor,
    dense_target: torch.Tensor,
    dense_energy: torch.Tensor,
    chunk_size: int,
) -> float:
    error = torch.zeros((), device=patch_logits.device, dtype=torch.float64)
    for start in range(0, patch_logits.shape[0], chunk_size):
        end = min(start + chunk_size, patch_logits.shape[0])
        output, _ = attention_output(
            special_logits[start:end],
            patch_logits[start:end],
            special_values,
            patch_values,
            patch_kept[start:end],
        )
        error += torch.sum((output - dense_target[start:end]) ** 2, dtype=torch.float64)
    return float((error / dense_energy).item())


class IterativePruningProfiler:
    def __init__(
        self,
        layer: int,
        frames: int,
        patch_height: int,
        patch_width: int,
        retention: float,
        prune_stages: list[float],
        chunk_size: int,
    ) -> None:
        self.layer = layer
        self.frames = frames
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.retention = retention
        self.prune_stages = prune_stages
        self.chunk_size = chunk_size
        self.rows: list[dict[str, Any]] = []
        self.per_head_final: list[dict[str, Any]] = []
        self.key_usage_counts: list[np.ndarray] = []

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
            raise ValueError("Iterative sparse profiling requires batch size one")
        qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query, key = module.q_norm(query), module.k_norm(key)
        if module.rope is not None:
            query, key = module.rope(query, position), module.rope(key, position)
        query, key, value = query[0], key[0], value[0]
        expected_tokens = self.frames * (NUM_SPECIAL_TOKENS + self.patch_height * self.patch_width)
        if tokens != expected_tokens:
            raise ValueError(f"Expected {expected_tokens} tokens, received {tokens}")
        special, patch, _ = make_token_indices(
            self.frames, self.patch_height, self.patch_width, x.device
        )
        final_count = patch_counts(patch.numel(), self.retention, self.prune_stages)[-1]
        target_counts = patch_counts(patch.numel(), self.retention, self.prune_stages)

        for head in range(NUM_HEADS):
            query_patch = query[head, patch].float()
            key_special = key[head, special].float()
            key_patch = key[head, patch].float()
            special_values = value[head, special].float()
            patch_values = value[head, patch].float()
            special_logits = (query_patch @ key_special.transpose(0, 1)) * module.scale
            patch_logits = (query_patch @ key_patch.transpose(0, 1)) * module.scale
            dense_target = dense_output(
                special_logits,
                patch_logits,
                special_values,
                patch_values,
                self.chunk_size,
            )
            dense_energy = torch.sum(dense_target ** 2, dtype=torch.float64).clamp_min(1.0e-12)
            patch_kept = torch.ones(
                patch_logits.shape, device=x.device, dtype=torch.bool
            )
            current_count = patch.numel()

            for target_count in target_counts:
                nmse_before_prune, mean_removed_cost = prune_stage(
                    special_logits,
                    patch_logits,
                    special_values,
                    patch_values,
                    patch_kept,
                    dense_target,
                    dense_energy,
                    current_count - target_count,
                    self.chunk_size,
                )
                self.rows.append(
                    {
                        "layer": self.layer,
                        "head": head,
                        "patch_keys_per_query": current_count,
                        "patch_key_retention": current_count / patch.numel(),
                        "nmse_to_dense": nmse_before_prune,
                        "removed_keys_per_query": current_count - target_count,
                        "mean_deleted_key_cost": mean_removed_cost,
                    }
                )
                current_count = target_count

            final_nmse = support_nmse(
                special_logits,
                patch_logits,
                special_values,
                patch_values,
                patch_kept,
                dense_target,
                dense_energy,
                self.chunk_size,
            )
            self.rows.append(
                {
                    "layer": self.layer,
                    "head": head,
                    "patch_keys_per_query": final_count,
                    "patch_key_retention": final_count / patch.numel(),
                    "nmse_to_dense": final_nmse,
                    "removed_keys_per_query": 0,
                    "mean_deleted_key_cost": "",
                }
            )
            self.per_head_final.append(
                {
                    "layer": self.layer,
                    "head": head,
                    "patch_keys_per_query": final_count,
                    "patch_key_retention": final_count / patch.numel(),
                    "special_keys_always_dense": int(special.numel()),
                    "final_nmse_to_dense": final_nmse,
                }
            )
            self.key_usage_counts.append(
                patch_kept.sum(dim=0, dtype=torch.int32).cpu().numpy()
            )
            print(
                f"layer {self.layer:02d} head {head:02d}: "
                f"iterative-pruning NMSE={final_nmse:.6g}",
                flush=True,
            )
            del patch_kept, dense_target, special_logits, patch_logits
            torch.cuda.empty_cache()


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
    profiler = IterativePruningProfiler(
        args.layer,
        args.num_frames,
        patch_height,
        patch_width,
        args.retention,
        args.prune_stages,
        args.query_chunk_size,
    )
    hook = model.aggregator.global_blocks[args.layer].attn.register_forward_hook(profiler, with_kwargs=True)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            run_to_global_layer(model, images.unsqueeze(0), args.layer)
    finally:
        hook.remove()
    if not profiler.per_head_final:
        raise RuntimeError("The selected global layer did not execute")
    metrics_path = args.output_dir / f"layer_{args.layer:02d}_iterative_pruning_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiler.rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(profiler.rows)
    usage_path = args.output_dir / f"layer_{args.layer:02d}_iterative_pruning_key_usage.npz"
    np.savez_compressed(
        usage_path,
        key_usage_counts=np.stack(profiler.key_usage_counts),
        patch_grid=np.array([patch_height, patch_width]),
        frame_indices=np.array(frame_indices),
    )
    summary = {
        "scene": args.scene,
        "sequence": args.sequence,
        "frame_indices": frame_indices,
        "layer": args.layer,
        "patch_grid": [patch_height, patch_width],
        "method": "batched_iterative_output_preserving_pruning",
        "method_note": (
            "At each stage the current softmax/output are recomputed, then the lowest exact "
            "single-key deletion costs are removed in one batch. This is a greedy local search, "
            "not an exhaustive globally optimal subset search."
        ),
        "deletion_cost": "a_j / (1 - a_j) * ||v_j - o||_2 under the current retained support",
        "special_keys_always_dense": int(args.num_frames * NUM_SPECIAL_TOKENS),
        "prune_stage_retentions": args.prune_stages + [args.retention],
        "per_head_final": profiler.per_head_final,
        "mean_final_nmse_to_dense": float(np.mean([row["final_nmse_to_dense"] for row in profiler.per_head_final])),
        "key_usage_counts_file": usage_path.name,
    }
    summary_path = args.output_dir / f"layer_{args.layer:02d}_iterative_pruning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved iterative pruning profile for layer {args.layer} to {args.output_dir}")


if __name__ == "__main__":
    main()
