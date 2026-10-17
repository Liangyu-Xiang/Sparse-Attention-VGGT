#!/usr/bin/env python3
"""Render selected VGGT global-attention layers for a 7Scenes frame sequence.

The global attention matrix has one row and one column per image token.  A
full-resolution matrix is too large to inspect across all 24 x 16 heads, so
this script applies 2-D adaptive mean pooling only for rendering and stores the
resulting 128 x 128 matrices. It does not select, average, or drop heads.
Only the requested prefix of alternating frame/global blocks is executed, so a
visualization of early global layers does not incur the cost of all 24 layers.
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
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from saf3r.models.official_vggt.models.vggt import VGGT
from saf3r.models.official_vggt.models.aggregator import slice_expand_and_flatten
from saf3r.utils.data_utils import load_and_process_images_vggt


NUM_GLOBAL_LAYERS = 24
NUM_HEADS = 16
NUM_SPECIAL_TOKENS = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize every head in selected VGGT global-attention layers."
    )
    parser.add_argument("--scene", default="chess", help="7Scenes scene name (default: chess).")
    parser.add_argument("--sequence", default="seq-01", help="7Scenes sequence directory.")
    parser.add_argument("--first-frame", type=int, default=0, help="Index of the first input frame.")
    parser.add_argument(
        "--num-frames", type=int, default=2, help="Number of input frames (default: 2)."
    )
    parser.add_argument(
        "--frame-gap", type=int, default=20, help="Stride between consecutive input frames."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("workspace/benchmark_dataset/7scenes/7Scenes"),
        help="Path containing 7Scenes scene directories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"),
        help="VGGT checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/global_attention/<sequence-name>.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=128,
        help="Token-axis bins in each rendered matrix (default: 128).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(NUM_GLOBAL_LAYERS)),
        help="Global-layer indices to render (default: all 24).",
    )
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda).")
    args = parser.parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive.")
    if args.frame_gap <= 0:
        raise ValueError("--frame-gap must be positive.")
    if args.bins < 16:
        raise ValueError("--bins must be at least 16.")
    if not args.layers or any(not 0 <= layer < NUM_GLOBAL_LAYERS for layer in args.layers):
        raise ValueError(f"--layers must lie in [0, {NUM_GLOBAL_LAYERS - 1}].")
    args.layers = sorted(set(args.layers))
    return args


def _frame_path(data_root: Path, scene: str, sequence: str, frame: int) -> Path:
    return data_root / scene / sequence / f"frame-{frame:06d}.color.png"


def _save_input_frames(image_paths: list[Path], destination: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    columns = min(5, len(images))
    rows = (len(images) + columns - 1) // columns
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images) + 32
    width = columns * cell_width
    height = rows * cell_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, image_path) in enumerate(zip(images, image_paths)):
        row, column = divmod(index, columns)
        x_offset = column * cell_width
        y_offset = row * cell_height
        canvas.paste(image, (x_offset, y_offset + 32))
        draw.text((x_offset + 8, y_offset + 8), image_path.stem.replace(".color", ""), fill="black")
    canvas.save(destination)


def _pool_attention(attention: torch.Tensor, bins: int) -> np.ndarray:
    """Return one binned global matrix per head without combining heads."""
    # Keep bfloat16 on GPU: a 10-frame global attention tensor is several GiB.
    pooled = F.adaptive_avg_pool2d(attention, output_size=(bins, bins))
    return pooled.to(dtype=torch.float16).cpu().numpy()


def _capture_hook(
    module: torch.nn.Module,
    input_args: tuple[Any, ...],
    input_kwargs: dict[str, Any],
    _output: torch.Tensor,
    *,
    layer_index: int,
    bins: int,
    output: list[np.ndarray | None],
) -> None:
    """Recompute exactly the softmax attention used by one fused attention layer."""
    x = input_args[0]
    position = input_kwargs.get("pos")
    batch, tokens, channels = x.shape
    if batch != 1:
        raise ValueError(f"Expected batch size 1, got {batch}.")

    qkv = (
        module.qkv(x)
        .reshape(batch, tokens, 3, module.num_heads, module.head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, _ = qkv.unbind(0)
    query, key = module.q_norm(query), module.k_norm(key)
    if module.rope is not None:
        query = module.rope(query, position)
        key = module.rope(key, position)

    attention = (query * module.scale) @ key.transpose(-2, -1)
    attention = attention.softmax(dim=-1).squeeze(0)
    output[layer_index] = _pool_attention(attention, bins)


def _build_model(checkpoint: Path, device: str) -> VGGT:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT checkpoint not found: {checkpoint}")
    model = VGGT(save_intermediates=False)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def _run_to_global_layer(model: VGGT, images: torch.Tensor, last_global_layer: int) -> None:
    """Run the alternating VGGT trunk only through ``last_global_layer``.

    This mirrors ``Aggregator.forward`` through its token construction and
    alternating frame/global blocks, but omits later blocks that cannot affect
    an early-layer attention visualization.
    """
    aggregator = model.aggregator
    batch, frames, channels_in, height, width = images.shape
    if channels_in != 3:
        raise ValueError(f"Expected RGB inputs, got {channels_in} channels.")
    if aggregator.aa_block_size != 1 or aggregator.aa_order != ["frame", "global"]:
        raise ValueError("Partial visualization currently expects alternating 1-by-1 frame/global blocks.")

    normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
    patch_tokens = aggregator.patch_embed(normalized.view(batch * frames, channels_in, height, width))
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"]

    camera_tokens = slice_expand_and_flatten(aggregator.camera_token, batch, frames)
    register_tokens = slice_expand_and_flatten(aggregator.register_token, batch, frames)
    tokens = torch.cat([camera_tokens, register_tokens, patch_tokens], dim=1)
    _, tokens_per_frame, channels = tokens.shape

    position = None
    if aggregator.rope is not None:
        position = aggregator.position_getter(
            batch * frames,
            height // aggregator.patch_size,
            width // aggregator.patch_size,
            device=images.device,
        )
        if aggregator.patch_start_idx > 0:
            position = position + 1
            special_position = torch.zeros(
                batch * frames,
                aggregator.patch_start_idx,
                2,
                device=images.device,
                dtype=position.dtype,
            )
            position = torch.cat([special_position, position], dim=1)

    frame_index = 0
    global_index = 0
    for _ in range(last_global_layer + 1):
        tokens, frame_index, _ = aggregator._process_frame_attention(
            tokens,
            batch,
            frames,
            tokens_per_frame,
            channels,
            frame_index,
            pos=position,
            save_feats=False,
        )
        tokens, global_index, _ = aggregator._process_global_attention(
            tokens,
            batch,
            frames,
            tokens_per_frame,
            channels,
            global_index,
            pos=position,
            save_feats=False,
        )


def _render_layer(
    matrices: np.ndarray,
    layer_index: int,
    frame_indices: list[int],
    destination: Path,
    pdf: PdfPages,
    log_vmin: float,
    log_vmax: float,
) -> None:
    figure, axes = plt.subplots(4, 4, figsize=(14, 14), constrained_layout=True)
    frame_count = len(frame_indices)
    frame_boundaries = [matrices.shape[-1] * index / frame_count - 0.5 for index in range(1, frame_count)]
    frame_centers = [matrices.shape[-1] * (index + 0.5) / frame_count - 0.5 for index in range(frame_count)]
    frame_labels = [f"F{frame}" for frame in frame_indices]
    image = None
    for head_index, axis in enumerate(axes.flat):
        rendered = np.log10(np.maximum(matrices[head_index].astype(np.float32), 1e-12))
        image = axis.imshow(
            rendered,
            cmap="magma",
            vmin=log_vmin,
            vmax=log_vmax,
            interpolation="nearest",
            aspect="equal",
        )
        for boundary in frame_boundaries:
            axis.axhline(boundary, color="cyan", linewidth=0.45)
            axis.axvline(boundary, color="cyan", linewidth=0.45)
        axis.set_title(f"Head {head_index}")
        axis.set_xticks(frame_centers)
        axis.set_yticks(frame_centers)
        axis.set_xticklabels(frame_labels, rotation=55, ha="right")
        axis.set_yticklabels(frame_labels)
        axis.tick_params(axis="both", labelsize=5)
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="log10(mean attention probability)")
    figure.suptitle(
        f"VGGT global attention · layer {layer_index:02d} · all 16 heads\n"
        "Rows are queries; columns are keys. Cyan lines separate input frames."
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    pdf.savefig(figure, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    frame_indices = [args.first_frame + index * args.frame_gap for index in range(args.num_frames)]
    image_paths = [
        _frame_path(args.data_root, args.scene, args.sequence, frame) for frame in frame_indices
    ]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing requested 7Scenes frame(s):\n" + "\n".join(missing))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    sequence_name = (
        f"7scenes_{args.scene}_f{frame_indices[0]:06d}_n{args.num_frames}_s{args.frame_gap}"
    )
    output_dir = args.output_dir or Path("outputs/global_attention") / sequence_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_input_frames(image_paths, output_dir / "input_frames.png")
    images = load_and_process_images_vggt([str(path) for path in image_paths]).to(args.device)
    model = _build_model(args.checkpoint, args.device)
    global_blocks = model.aggregator.global_blocks
    if len(global_blocks) != NUM_GLOBAL_LAYERS:
        raise ValueError(f"Expected {NUM_GLOBAL_LAYERS} global blocks, got {len(global_blocks)}.")
    if global_blocks[0].attn.num_heads != NUM_HEADS:
        raise ValueError(f"Expected {NUM_HEADS} heads, got {global_blocks[0].attn.num_heads}.")

    last_layer = max(args.layers)
    captured: list[np.ndarray | None] = [None] * (last_layer + 1)
    hooks = [
        block.attn.register_forward_hook(
            lambda module, input_args, input_kwargs, output, layer_index=index: _capture_hook(
                module,
                input_args,
                input_kwargs,
                output,
                layer_index=layer_index,
                bins=args.bins,
                output=captured,
            ),
            with_kwargs=True,
        )
        for index, block in enumerate(global_blocks[: last_layer + 1])
        if index in args.layers
    ]
    try:
        autocast_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type="cuda" if args.device.startswith("cuda") else "cpu",
            dtype=autocast_dtype,
            enabled=args.device.startswith("cuda"),
        ):
            _run_to_global_layer(model, images.unsqueeze(0), last_layer)
    finally:
        for hook in hooks:
            hook.remove()

    if any(captured[layer] is None for layer in args.layers):
        missing_layers = [layer for layer in args.layers if captured[layer] is None]
        raise RuntimeError(f"No attention map captured for global layer(s): {missing_layers}")
    all_matrices = np.stack([captured[layer] for layer in args.layers if captured[layer] is not None], axis=0)
    np.savez_compressed(output_dir / "global_attention_binned.npz", attention=all_matrices)

    log_attention = np.log10(np.maximum(all_matrices.astype(np.float32), 1e-12))
    log_vmin, log_vmax = np.quantile(log_attention, [0.01, 0.999])
    with PdfPages(output_dir / "all_global_attention_heads.pdf") as pdf:
        for layer_index, matrices in zip(args.layers, all_matrices):
            _render_layer(
                matrices,
                layer_index,
                frame_indices,
                output_dir / f"global_attention_layer_{layer_index:02d}.png",
                pdf,
                float(log_vmin),
                float(log_vmax),
            )

    patch_height, patch_width = images.shape[-2] // 14, images.shape[-1] // 14
    metadata = {
        "model": "VGGT dense global attention",
        "scene": args.scene,
        "sequence": args.sequence,
        "frame_indices": list(frame_indices),
        "frame_gap": args.frame_gap,
        "num_frames": args.num_frames,
        "image_paths": [str(path) for path in image_paths],
        "processed_image_shape": list(images.shape),
        "patch_grid": [patch_height, patch_width],
        "special_tokens_per_frame": NUM_SPECIAL_TOKENS,
        "global_token_count": int(args.num_frames * (NUM_SPECIAL_TOKENS + patch_height * patch_width)),
        "global_layers_rendered": args.layers,
        "global_layer_forward_stop": last_layer,
        "heads_per_layer": NUM_HEADS,
        "matrix_binning": {
            "method": "2-D adaptive mean pooling for rendering only",
            "shape_per_head": [args.bins, args.bins],
            "value": "mean attention probability in each token-bin cell",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(args.layers) * NUM_HEADS} head matrices to {output_dir}")


if __name__ == "__main__":
    main()
