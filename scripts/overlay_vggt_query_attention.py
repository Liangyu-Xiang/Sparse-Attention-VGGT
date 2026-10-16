#!/usr/bin/env python3
"""Overlay exact VGGT patch-key attention rows on the input RGB images.

Each selected query is described as ``frame:patch_row:patch_col``.  The script
recomputes the unbinned dense attention row of every requested head, removes
the five non-spatial tokens per frame only for rendering, and overlays the
remaining patch-key probability on every input image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
import torch

from saf3r.models.official_vggt.models.aggregator import slice_expand_and_flatten
from saf3r.models.official_vggt.models.vggt import VGGT
from saf3r.utils.data_utils import load_and_process_images_vggt


NUM_GLOBAL_LAYERS = 24
NUM_SPECIAL_TOKENS = 5


def parse_layer_head(value: str) -> tuple[int, int]:
    try:
        layer_text, head_text = value.split(":")
        layer, head = int(layer_text), int(head_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use LAYER:HEAD, for example 14:10") from error
    if not 0 <= layer < NUM_GLOBAL_LAYERS or not 0 <= head < 16:
        raise argparse.ArgumentTypeError("VGGT layer must be 0..23 and head 0..15")
    return layer, head


def parse_query(value: str) -> tuple[int, int, int]:
    try:
        frame_text, row_text, col_text = value.split(":")
        return int(frame_text), int(row_text), int(col_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use FRAME:PATCH_ROW:PATCH_COL, for example 0:14:18") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--sequence", default="seq-01")
    parser.add_argument("--first-frame", type=int, default=0)
    parser.add_argument("--frame-gap", type=int, default=20)
    parser.add_argument("--data-root", type=Path, default=Path("workspace/benchmark_dataset/7scenes/7Scenes"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/query_attention_overlays/chess_f000000_f000020"))
    parser.add_argument(
        "--layer-head",
        nargs="+",
        type=parse_layer_head,
        default=[(12, 8), (13, 10), (14, 10), (15, 8)],
        help="One or more LAYER:HEAD pairs (default: 12:8 13:10 14:10 15:8).",
    )
    parser.add_argument(
        "--query",
        nargs="+",
        type=parse_query,
        default=[(0, 7, 9), (0, 20, 27), (1, 7, 27), (1, 20, 9)],
        help="One or more FRAME:PATCH_ROW:PATCH_COL queries.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.frame_gap <= 0:
        parser.error("--frame-gap must be positive")
    return args


def frame_path(root: Path, scene: str, sequence: str, frame: int) -> Path:
    return root / scene / sequence / f"frame-{frame:06d}.color.png"


def build_model(checkpoint: Path, device: str) -> VGGT:
    model = VGGT(save_intermediates=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval()


def run_to_global_layer(model: VGGT, images: torch.Tensor, last_global_layer: int) -> None:
    """Run the VGGT alternating trunk through the requested global layer."""
    aggregator = model.aggregator
    batch, frames, channels_in, height, width = images.shape
    normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
    patch_tokens = aggregator.patch_embed(normalized.view(batch * frames, channels_in, height, width))
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]
    camera = slice_expand_and_flatten(aggregator.camera_token, batch, frames)
    registers = slice_expand_and_flatten(aggregator.register_token, batch, frames)
    tokens = torch.cat([camera, registers, patch_tokens], dim=1)
    _, tokens_per_frame, channels = tokens.shape
    position = aggregator.position_getter(
        batch * frames, height // aggregator.patch_size, width // aggregator.patch_size, device=images.device
    )
    position = position + 1
    specials = torch.zeros(
        batch * frames, aggregator.patch_start_idx, 2, device=images.device, dtype=position.dtype
    )
    position = torch.cat([specials, position], dim=1)

    frame_index = global_index = 0
    for _ in range(last_global_layer + 1):
        tokens, frame_index, _ = aggregator._process_frame_attention(
            tokens, batch, frames, tokens_per_frame, channels, frame_index, pos=position, save_feats=False
        )
        tokens, global_index, _ = aggregator._process_global_attention(
            tokens, batch, frames, tokens_per_frame, channels, global_index, pos=position, save_feats=False
        )


class RowCollector:
    def __init__(self, wanted: set[tuple[int, int]], queries: list[tuple[int, int, int]]) -> None:
        self.wanted = wanted
        self.queries = queries
        self.rows: dict[tuple[int, int], torch.Tensor] = {}
        self.tokens_per_frame: int | None = None
        self.patch_grid: tuple[int, int] | None = None

    def hook(self, layer: int):
        def capture(
            module: torch.nn.Module,
            input_args: tuple[Any, ...],
            input_kwargs: dict[str, Any],
            _output: torch.Tensor,
        ) -> None:
            target_heads = [head for candidate_layer, head in self.wanted if candidate_layer == layer]
            if not target_heads:
                return
            x = input_args[0]
            position = input_kwargs.get("pos")
            batch, tokens, _ = x.shape
            qkv = module.qkv(x).reshape(batch, tokens, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            query, key, _ = qkv.unbind(0)
            query, key = module.q_norm(query), module.k_norm(key)
            if module.rope is not None:
                query, key = module.rope(query, position), module.rope(key, position)
            if tokens % 2:
                raise ValueError("This visualizer expects exactly two frames")
            per_frame = tokens // 2
            patch_count = per_frame - NUM_SPECIAL_TOKENS
            width = int(round(patch_count ** 0.5))
            # The actual grid dimensions are attached by main after image preprocessing.
            if self.tokens_per_frame is None:
                self.tokens_per_frame = per_frame
            for head in target_heads:
                query_indices = torch.tensor(
                    [frame * per_frame + NUM_SPECIAL_TOKENS + row * self.patch_grid[1] + col for frame, row, col in self.queries],
                    device=x.device,
                )
                selected_query = query[0, head, query_indices]
                logits = (selected_query * module.scale) @ key[0, head].transpose(0, 1)
                self.rows[(layer, head)] = logits.float().softmax(dim=-1).cpu()
        return capture


def render(
    rows: torch.Tensor,
    layer: int,
    head: int,
    queries: list[tuple[int, int, int]],
    originals: list[Image.Image],
    patch_grid: tuple[int, int],
    tokens_per_frame: int,
    frame_indices: list[int],
    pdf: PdfPages,
    destination: Path,
) -> None:
    height, width = patch_grid
    query_count = len(queries)
    figure, axes = plt.subplots(query_count, 2, figsize=(12, 3.9 * query_count), constrained_layout=True)
    axes = np.atleast_2d(axes)
    patch_values = []
    for row in rows:
        for frame in range(2):
            begin = frame * tokens_per_frame + NUM_SPECIAL_TOKENS
            patch_values.append(row[begin:begin + height * width].numpy())
    positive = np.concatenate(patch_values)
    norm = LogNorm(vmin=max(float(np.quantile(positive, 0.02)), 1e-10), vmax=float(np.quantile(positive, 0.995)))
    image_artist = None
    for query_index, ((query_frame, query_row, query_col), row) in enumerate(zip(queries, rows)):
        special_mass = sum(float(row[frame * tokens_per_frame:frame * tokens_per_frame + NUM_SPECIAL_TOKENS].sum()) for frame in range(2))
        for key_frame, axis in enumerate(axes[query_index]):
            image = originals[key_frame]
            image_width, image_height = image.size
            begin = key_frame * tokens_per_frame + NUM_SPECIAL_TOKENS
            attention = row[begin:begin + height * width].reshape(height, width).numpy()
            axis.imshow(image, extent=(0, image_width, image_height, 0))
            image_artist = axis.imshow(
                attention,
                cmap="magma",
                norm=norm,
                alpha=0.62,
                extent=(0, image_width, image_height, 0),
                interpolation="bilinear",
            )
            if key_frame == query_frame:
                x0, x1 = query_col * image_width / width, (query_col + 1) * image_width / width
                y0, y1 = query_row * image_height / height, (query_row + 1) * image_height / height
                axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="cyan", linewidth=2.2))
            patch_mass = float(attention.sum())
            axis.set_title(f"key image F{frame_indices[key_frame]} · patch mass={patch_mass:.3f}")
            axis.set_axis_off()
        axes[query_index, 0].set_ylabel(
            f"query F{frame_indices[query_frame]}\npatch ({query_row}, {query_col})\nspecial mass={special_mass:.3f}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=88,
        )
    assert image_artist is not None
    figure.colorbar(image_artist, ax=axes.ravel().tolist(), shrink=0.84, label="attention probability (log scale)")
    figure.suptitle(
        f"VGGT global attention overlay · layer {layer:02d}, head {head:02d}\n"
        "Cyan box marks the query patch; each panel maps its attention over all patch keys in that image."
    )
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    pdf.savefig(figure, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    frame_indices = [args.first_frame, args.first_frame + args.frame_gap]
    paths = [frame_path(args.data_root, args.scene, args.sequence, frame) for frame in frame_indices]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input frames:\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    originals = [Image.open(path).convert("RGB") for path in paths]
    images = load_and_process_images_vggt([str(path) for path in paths]).to(args.device)
    patch_grid = (images.shape[-2] // 14, images.shape[-1] // 14)
    for frame, row, col in args.query:
        if not 0 <= frame < 2 or not 0 <= row < patch_grid[0] or not 0 <= col < patch_grid[1]:
            raise ValueError(f"Invalid query {(frame, row, col)} for patch grid {patch_grid}")
    model = build_model(args.checkpoint, args.device)
    wanted = set(args.layer_head)
    collector = RowCollector(wanted, args.query)
    collector.patch_grid = patch_grid
    last_layer = max(layer for layer, _ in wanted)
    hooks = [
        block.attn.register_forward_hook(collector.hook(layer), with_kwargs=True)
        for layer, block in enumerate(model.aggregator.global_blocks[:last_layer + 1])
        if any(candidate_layer == layer for candidate_layer, _ in wanted)
    ]
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            run_to_global_layer(model, images.unsqueeze(0), last_layer)
    finally:
        for hook in hooks:
            hook.remove()
    if set(collector.rows) != wanted:
        raise RuntimeError(f"Missing captured rows: {wanted - set(collector.rows)}")
    assert collector.tokens_per_frame is not None
    with PdfPages(args.output_dir / "query_attention_overlays.pdf") as pdf:
        for layer, head in sorted(wanted):
            render(
                collector.rows[(layer, head)], layer, head, args.query, originals, patch_grid,
                collector.tokens_per_frame, frame_indices, pdf,
                args.output_dir / f"overlay_layer_{layer:02d}_head_{head:02d}.png",
            )
    metadata = {
        "scene": args.scene,
        "sequence": args.sequence,
        "frame_indices": frame_indices,
        "patch_grid": patch_grid,
        "layer_head": sorted([list(item) for item in wanted]),
        "queries": [list(query) for query in args.query],
        "attention": "exact dense attention; special-token mass is reported but excluded from spatial overlays",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(wanted)} query-overlay figures to {args.output_dir}")


if __name__ == "__main__":
    main()
