#!/usr/bin/env python3
"""Measure whether an earlier VGGT head's sparse support serves later heads.

The source contributes only a per-query Top-rho patch-key mask.  Every target
recomputes logits, softmax weights, and values from its own Q/K/V.  Results are
reported against target-self Top-rho and uniform supports at the same budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from saf3r.models.official_vggt.models.vggt import VGGT
from saf3r.utils.data_utils import load_and_process_images_vggt


NUM_GLOBAL_LAYERS = 24
NUM_HEADS = 16
NUM_SPECIAL_TOKENS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sequence", default="seq-01")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--frame-gap", type=int, default=20)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--retention", type=float, default=0.25)
    parser.add_argument("--layer-deltas", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--query-chunk-size", type=int, default=64)
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
    if not 0.0 < args.retention < 1.0:
        parser.error("--retention must lie in (0, 1)")
    if args.num_frames < 1 or args.frame_gap < 1:
        parser.error("--num-frames and --frame-gap must be positive")
    if not args.layer_deltas or any(delta < 1 for delta in args.layer_deltas):
        parser.error("--layer-deltas must contain positive integers")
    if args.query_chunk_size < 1:
        parser.error("--query-chunk-size must be positive")
    args.layer_deltas = sorted(set(args.layer_deltas))
    return args


def build_model(checkpoint: Path, device: str) -> VGGT:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
    model = VGGT(save_intermediates=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval()


def frame_path(root: Path, scene: str, sequence: str, frame: int) -> Path:
    return root / scene / sequence / f"frame-{frame:06d}.color.png"


def patch_indices(total_tokens: int, num_frames: int, device: torch.device) -> torch.Tensor:
    if total_tokens % num_frames:
        raise ValueError(f"{total_tokens} tokens cannot be split across {num_frames} frames")
    per_frame = total_tokens // num_frames
    if per_frame <= NUM_SPECIAL_TOKENS:
        raise ValueError("No patch tokens remain after excluding special tokens")
    return torch.cat([
        torch.arange(
            frame * per_frame + NUM_SPECIAL_TOKENS,
            (frame + 1) * per_frame,
            device=device,
        )
        for frame in range(num_frames)
    ])


def qkv_from_module(
    module: torch.nn.Module,
    input_args: tuple[Any, ...],
    input_kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = input_args[0]
    position = input_kwargs.get("pos")
    batch, tokens, _ = x.shape
    if batch != 1:
        raise ValueError(f"Expected batch size one, got {batch}")
    qkv = (
        module.qkv(x)
        .reshape(batch, tokens, 3, module.num_heads, module.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, value = qkv.unbind(0)
    query, key = module.q_norm(query), module.k_norm(key)
    if module.rope is not None:
        query = module.rope(query, position)
        key = module.rope(key, position)
    return query[0], key[0], value[0]


def top_supports(query: torch.Tensor, key: torch.Tensor, keep: int) -> torch.Tensor:
    """Return [heads, queries, keep] int32 patch-key indices."""
    logits = torch.matmul(query, key.transpose(-2, -1)) * (query.shape[-1] ** -0.5)
    return logits.topk(keep, dim=-1, sorted=False).indices.to(torch.int32)


def sparse_outputs(
    logits: torch.Tensor,
    value: torch.Tensor,
    supports: torch.Tensor,
    query_chunk_size: int,
) -> torch.Tensor:
    """Recompute target softmax/value aggregation for one or more supports."""
    support_count, queries, _ = supports.shape
    outputs = torch.empty(
        (support_count, queries, value.shape[-1]), device=value.device, dtype=torch.float32
    )
    for start in range(0, queries, query_chunk_size):
        end = min(queries, start + query_chunk_size)
        index = supports[:, start:end].long()
        selected_logits = logits[start:end].unsqueeze(0).expand(support_count, -1, -1)
        selected_logits = selected_logits.gather(2, index)
        probability = selected_logits.float().softmax(dim=-1)
        selected_value = value[index].float()
        outputs[:, start:end] = torch.sum(probability.unsqueeze(-1) * selected_value, dim=2)
    return outputs


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    denominator = torch.linalg.vector_norm(reference.float(), dim=-1).clamp_min(1.0e-8)
    error = torch.linalg.vector_norm(candidate.float() - reference.float().unsqueeze(0), dim=-1)
    return (error / denominator.unsqueeze(0)).mean(dim=-1)


class TransferCollector:
    def __init__(
        self,
        num_frames: int,
        retention: float,
        layer_deltas: list[int],
        query_chunk_size: int,
    ) -> None:
        self.num_frames = num_frames
        self.retention = retention
        self.layer_deltas = layer_deltas
        self.query_chunk_size = query_chunk_size
        self.layer = 0
        self.history: dict[int, torch.Tensor] = {}
        self.transfer: dict[int, list[np.ndarray]] = {delta: [] for delta in layer_deltas}
        self.self_error: list[np.ndarray] = []
        self.uniform_error: list[np.ndarray] = []
        self.patch_count: int | None = None
        self.keep: int | None = None

    def __call__(
        self,
        module: torch.nn.Module,
        input_args: tuple[Any, ...],
        input_kwargs: dict[str, Any],
        _output: torch.Tensor,
    ) -> None:
        query, key, value = qkv_from_module(module, input_args, input_kwargs)
        indices = patch_indices(query.shape[1], self.num_frames, query.device)
        query, key, value = query[:, indices], key[:, indices], value[:, indices]
        patch_count = query.shape[1]
        keep = max(1, int(round(self.retention * patch_count)))
        if self.patch_count is None:
            self.patch_count, self.keep = patch_count, keep
        elif (self.patch_count, self.keep) != (patch_count, keep):
            raise RuntimeError("Patch token shape changed between global layers")

        current_support = top_supports(query, key, keep)
        uniform_1d = torch.div(
            torch.arange(keep, device=query.device, dtype=torch.long) * patch_count,
            keep,
            rounding_mode="floor",
        ).to(torch.int32)
        uniform = uniform_1d.view(1, 1, keep).expand(1, patch_count, keep)

        layer_self = np.empty(NUM_HEADS, dtype=np.float32)
        layer_uniform = np.empty(NUM_HEADS, dtype=np.float32)
        available = {
            delta: self.history[self.layer - delta]
            for delta in self.layer_deltas
            if self.layer - delta in self.history
        }
        layer_transfer = {
            delta: np.empty((NUM_HEADS, NUM_HEADS), dtype=np.float32)
            for delta in available
        }

        for target_head in range(NUM_HEADS):
            target_query = query[target_head]
            target_key = key[target_head]
            target_value = value[target_head]
            logits = torch.matmul(target_query, target_key.transpose(0, 1)) * (
                target_query.shape[-1] ** -0.5
            )
            dense = torch.matmul(logits.float().softmax(dim=-1), target_value.float())
            controls = torch.cat((current_support[target_head:target_head + 1], uniform), dim=0)
            control_outputs = sparse_outputs(
                logits, target_value, controls, self.query_chunk_size
            )
            control_error = relative_l2(control_outputs, dense)
            layer_self[target_head] = control_error[0].item()
            layer_uniform[target_head] = control_error[1].item()

            for delta, source_support in available.items():
                transferred = sparse_outputs(
                    logits, target_value, source_support, self.query_chunk_size
                )
                layer_transfer[delta][:, target_head] = (
                    relative_l2(transferred, dense).cpu().numpy()
                )

        self.self_error.append(layer_self)
        self.uniform_error.append(layer_uniform)
        for delta, matrix in layer_transfer.items():
            self.transfer[delta].append(matrix)
        self.history[self.layer] = current_support
        oldest_needed = self.layer - max(self.layer_deltas)
        self.history.pop(oldest_needed, None)
        print(
            f"layer {self.layer:02d}: self={layer_self.mean():.4f}, "
            f"uniform={layer_uniform.mean():.4f}, deltas={list(available)}",
            flush=True,
        )
        self.layer += 1


def save_results(args: argparse.Namespace, collector: TransferCollector, frame_indices: list[int]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_error = np.stack(collector.self_error)
    uniform_error = np.stack(collector.uniform_error)
    arrays: dict[str, np.ndarray] = {
        "self_error": self_error,
        "uniform_error": uniform_error,
    }
    rows: list[dict[str, Any]] = []
    delta_summary: dict[str, Any] = {}
    for delta in args.layer_deltas:
        transfer = np.stack(collector.transfer[delta])
        target_layers = np.arange(delta, NUM_GLOBAL_LAYERS)
        target_self = self_error[target_layers]
        target_uniform = uniform_error[target_layers]
        regret = transfer - target_self[:, None, :]
        denominator = target_uniform - target_self
        valid_normalization = denominator > 1.0e-8
        normalized = np.full_like(regret, np.nan)
        np.divide(
            regret,
            denominator[:, None, :],
            out=normalized,
            where=valid_normalization[:, None, :],
        )
        self_ratio = transfer / (target_self[:, None, :] + 1.0e-8)
        beats_uniform = transfer <= target_uniform[:, None, :]
        arrays[f"transfer_error_d{delta}"] = transfer
        arrays[f"transfer_regret_d{delta}"] = regret
        arrays[f"normalized_regret_d{delta}"] = normalized
        arrays[f"self_error_ratio_d{delta}"] = self_ratio
        arrays[f"beats_uniform_d{delta}"] = beats_uniform

        best_transfer = transfer.min(axis=1)
        best_source = transfer.argmin(axis=1)
        best_ratio = best_transfer / (target_self + 1.0e-8)
        best_beats_uniform = best_transfer <= target_uniform
        valid_values = normalized[np.broadcast_to(valid_normalization[:, None, :], normalized.shape)]
        best_normalized = np.where(np.isfinite(normalized), normalized, np.inf).min(axis=1)
        best_normalized[~np.isfinite(best_normalized)] = np.nan
        valid_best = best_normalized[np.isfinite(best_normalized)]
        delta_summary[str(delta)] = {
            "fraction_targets_self_better_than_uniform": float(valid_normalization.mean()),
            "fraction_best_transfer_within_10pct_of_self": float((best_ratio <= 1.10).mean()),
            "fraction_best_transfer_beats_uniform": float(best_beats_uniform.mean()),
            "fraction_best_transfer_both": float(
                ((best_ratio <= 1.10) & best_beats_uniform).mean()
            ),
            "median_best_transfer_to_self_error_ratio": float(np.median(best_ratio)),
            "mean_best_normalized_regret_valid_only": (
                float(valid_best.mean()) if valid_best.size else None
            ),
            "median_best_normalized_regret_valid_only": (
                float(np.median(valid_best)) if valid_best.size else None
            ),
            "mean_all_normalized_regret_valid_only": (
                float(valid_values.mean()) if valid_values.size else None
            ),
        }
        for pair_index, target_layer in enumerate(target_layers):
            source_layer = int(target_layer - delta)
            for source_head in range(NUM_HEADS):
                for target_head in range(NUM_HEADS):
                    rows.append({
                        "scene": args.scene,
                        "delta": delta,
                        "source_layer": source_layer,
                        "target_layer": int(target_layer),
                        "source_head": source_head,
                        "target_head": target_head,
                        "transfer_error": float(transfer[pair_index, source_head, target_head]),
                        "self_error": float(target_self[pair_index, target_head]),
                        "uniform_error": float(target_uniform[pair_index, target_head]),
                        "transfer_regret": float(regret[pair_index, source_head, target_head]),
                        "normalized_regret": (
                            float(normalized[pair_index, source_head, target_head])
                            if np.isfinite(normalized[pair_index, source_head, target_head])
                            else ""
                        ),
                        "transfer_to_self_error_ratio": float(
                            self_ratio[pair_index, source_head, target_head]
                        ),
                        "beats_uniform": bool(
                            beats_uniform[pair_index, source_head, target_head]
                        ),
                        "best_source_for_target": bool(
                            best_source[pair_index, target_head] == source_head
                        ),
                    })

    np.savez_compressed(args.output_dir / "transfer_metrics.npz", **arrays)
    with (args.output_dir / "transfer_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "scene": args.scene,
        "sequence": args.sequence,
        "frame_indices": frame_indices,
        "retention": args.retention,
        "patch_token_count": collector.patch_count,
        "kept_keys_per_query": collector.keep,
        "layer_deltas": args.layer_deltas,
        "metric": "mean per-query relative L2 error on patch-only per-head pre-projection output",
        "protocol": (
            "source transfers only its Top-rho patch-key indices; target recomputes its own "
            "QK logits, restricted softmax, and V aggregation"
        ),
        "summary": delta_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    frame_indices = [
        args.first_frame + offset * args.frame_gap for offset in range(args.num_frames)
    ]
    paths = [frame_path(args.data_root, args.scene, args.sequence, frame) for frame in frame_indices]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input frames:\n" + "\n".join(missing))

    images = load_and_process_images_vggt([str(path) for path in paths]).to(args.device)
    model = build_model(args.checkpoint, args.device)
    blocks = model.aggregator.global_blocks
    if len(blocks) != NUM_GLOBAL_LAYERS or blocks[0].attn.num_heads != NUM_HEADS:
        raise ValueError("Unexpected VGGT global-attention layout")
    collector = TransferCollector(
        args.num_frames, args.retention, args.layer_deltas, args.query_chunk_size
    )
    hooks = [block.attn.register_forward_hook(collector, with_kwargs=True) for block in blocks]
    try:
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            model.aggregator(images.unsqueeze(0))
    finally:
        for hook in hooks:
            hook.remove()
    if collector.layer != NUM_GLOBAL_LAYERS:
        raise RuntimeError(f"Collected {collector.layer} global layers, expected {NUM_GLOBAL_LAYERS}")
    save_results(args, collector, frame_indices)


if __name__ == "__main__":
    main()
