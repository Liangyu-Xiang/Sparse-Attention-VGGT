#!/usr/bin/env python3
"""PAM k-medoids on dense VGGT values, with original-image patch overlays.

Distances are Euclidean in the unnormalised, per-head value space. Clustering
is joint across input images, excluding special tokens. Every selected medoid
is a real patch value. Two starts (greedy BUILD and seeded distance-weighted
sampling) each undergo exhaustive best single-swap PAM optimisation.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def pairwise_distances(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    squared = values.square().sum(1)
    distance = (squared[:, None] + squared[None, :] - 2 * (values @ values.T)).clamp_min_(0).sqrt_()
    distance.fill_diagonal_(0)
    return distance


def initialise(distance: torch.Tensor, k: int, method: str, seed: int, progress=None) -> torch.Tensor:
    n = len(distance)
    selected = torch.zeros(n, dtype=torch.bool, device=distance.device)
    medoids = [int(distance.sum(0).argmin())]
    selected[medoids[0]] = True
    nearest = distance[:, medoids[0]].clone()
    generator = torch.Generator(device=distance.device).manual_seed(seed)
    if progress:
        progress({"stage":"initializing","method":method,"completed":1,"total":k})
    for _ in range(1, k):
        if method == "build":
            gain = (nearest[:, None] - distance).clamp_min_(0).sum(0)
            gain[selected] = -torch.inf
            candidate = int(gain.argmax())
        else:
            weights = nearest.square().masked_fill(selected, 0)
            if float(weights.sum()) == 0:
                candidate = int((~selected).nonzero()[0])
            else:
                candidate = int(torch.multinomial(weights, 1, generator=generator))
        medoids.append(candidate)
        selected[candidate] = True
        nearest = torch.minimum(nearest, distance[:, candidate])
        if progress and ((_+1)%100==0 or _+1==k):
            progress({"stage":"initializing","method":method,"completed":_+1,"total":k})
    return torch.tensor(medoids, device=distance.device)


def best_swap(distance: torch.Tensor, medoids: torch.Tensor) -> tuple[float, int, int]:
    """Evaluate every (removed medoid, added non-medoid) pair exactly.

    For candidate c, the base gain is sum_i min(D_ic-d1_i,0). Only points
    assigned to the removed medoid need correction using second distance d2.
    scatter_add accumulates this correction for every removal in parallel.
    """
    n, k = len(distance), len(medoids)
    if k < 2:
        current = distance[:, medoids[0]].sum()
        costs = distance.sum(0) - current
        costs[medoids] = torch.inf
        candidate = int(costs.argmin())
        return float(costs[candidate]), 0, candidate
    nearest_two, positions = distance[:, medoids].topk(2, dim=1, largest=False)
    d1, d2 = nearest_two[:, 0], nearest_two[:, 1]
    owner = positions[:, 0]
    # Rows index insertion candidates and columns index data points.
    candidate_distances = distance.T
    base_gain = (candidate_distances - d1[None, :]).clamp_max(0)
    correction = torch.minimum(candidate_distances, d2[None, :]) - d1[None, :] - base_gain
    grouped = torch.zeros((n, k), dtype=distance.dtype, device=distance.device)
    grouped.scatter_add_(1, owner[None, :].expand(n, n), correction)
    costs = grouped + base_gain.sum(1, keepdim=True)
    costs[medoids] = torch.inf
    flat = int(costs.argmin())
    candidate, removed = divmod(flat, k)
    return float(costs[candidate, removed]), removed, candidate


def pam(distance: torch.Tensor, k: int, method: str, seed: int, max_swaps: int, progress=None) -> dict:
    medoids = initialise(distance, k, method, seed, progress=progress)
    objective = float(distance[:, medoids].amin(1).double().sum())
    history = [objective]
    tolerance = max(1e-6, 1e-7 * objective)
    converged = False
    stop_reason = "max_swaps"
    last_delta = None
    if progress:
        progress({"stage":"swapping","method":method,"iteration":0,"max_swaps":max_swaps,"objective":objective})
    for _ in range(max_swaps):
        delta, removed, candidate = best_swap(distance, medoids)
        last_delta = delta
        if delta >= -tolerance:
            converged = True
            stop_reason = "no_improving_single_swap"
            break
        proposed = medoids.clone()
        proposed[removed] = candidate
        new_objective = float(distance[:, proposed].amin(1).double().sum())
        if objective - new_objective <= tolerance:
            stop_reason = "numerical_tolerance"
            break
        medoids, objective = proposed, new_objective
        history.append(objective)
        if progress and (_+1)%25==0:
            progress({"stage":"swapping","method":method,"iteration":_+1,"max_swaps":max_swaps,"objective":objective})
    nearest, assignments = distance[:, medoids].min(1)
    if progress:
        progress({"stage":"fit_finished","method":method,"swaps":len(history)-1,"converged":converged,"objective":objective})
    return {
        "medoids": medoids.cpu().numpy(),
        "assignments": assignments.cpu().numpy(),
        "distances": nearest.cpu().numpy(),
        "objective": objective,
        "initial_objective": history[0],
        "objective_history": history,
        "swaps": len(history) - 1,
        "converged": converged,
        "stop_reason": stop_reason,
        "best_final_swap_delta": last_delta,
        "tolerance": tolerance,
        "initialisation": method,
    }


def patch_values_from_tokens(values: torch.Tensor, frames: int, patches_per_frame: int,
                             special_tokens: int = 5) -> torch.Tensor:
    """[frame-major tokens, heads, dim] -> [heads, frame-major patches, dim]."""
    if values.ndim != 3 or values.shape[0] != frames*(patches_per_frame+special_tokens):
        raise ValueError("Value token shape does not match the frame/patch layout")
    heads, dim = values.shape[1:]
    return (values.reshape(frames,patches_per_frame+special_tokens,heads,dim)[:,special_tokens:]
            .permute(2,0,1,3).reshape(heads,frames*patches_per_frame,dim))


def extract(args: argparse.Namespace) -> None:
    from scripts.overlay_vggt_query_attention import build_model, run_to_global_layer
    from saf3r.utils.data_utils import load_and_process_images_vggt

    source = json.loads(args.source_metadata.read_text())
    paths = source["image_paths"]
    frames = len(paths)
    if frames < 1 or frames != len(source["frame_indices"]):
        raise ValueError("Image paths and frame indices must have equal nonzero length")
    images = load_and_process_images_vggt(paths).to(args.device)
    grid = (images.shape[-2] // 14, images.shape[-1] // 14)
    count = grid[0] * grid[1]
    captured = {}

    def hook(layer):
        def capture(module, inputs, _output):
            x = inputs[0]
            qkv = module.qkv(x).reshape(1, x.shape[1], 3, module.num_heads, module.head_dim)
            # Capture V before attention and before output projection; Q/K
            # normalisation and RoPE do not transform these value vectors.
            values = patch_values_from_tokens(qkv[0, :, 2], frames, count)
            captured[layer] = values.float().cpu().numpy()
            print(f"提取第 {layer + 1} 个全局层：{captured[layer].shape}", flush=True)
        return capture

    model = build_model(args.checkpoint, args.device)
    handles = [model.aggregator.global_blocks[layer].attn.register_forward_hook(hook(layer)) for layer in args.layers]
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            run_to_global_layer(model, images.unsqueeze(0), max(args.layers))
    finally:
        for handle in handles:
            handle.remove()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "values.npz", **{f"layer_{layer}": value for layer, value in captured.items()})
    metadata = {
        "source_metadata": str(args.source_metadata.resolve()),
        "scene": source["scene"], "sequence": source["sequence"],
        "frame_indices": source["frame_indices"], "image_paths": paths,
        "processed_image_shape": list(images.shape), "patch_grid": list(grid),
        "special_tokens_excluded": True,
        "global_layer_indices": list(args.layers),
        "num_frames": frames, "values_per_head": frames * count, "head_dim": int(captured[args.layers[0]].shape[-1]),
        "distance": "euclidean_raw_values", "algorithm": "PAM exhaustive best single swap",
        "initialisations": ["greedy_BUILD", "seeded_distance_weighted"],
        "ratios_are_retained_fraction": True,
        "precision": "dense model BF16 autocast; distance and optimisation FP32; objective reduction FP64",
        "checkpoint": str(args.checkpoint.resolve()),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def valid_cached_cluster(array_path: Path, stats_path: Path, n: int, k: int, ratio: float) -> bool:
    """Only resume complete, converged and internally consistent cached fits."""
    try:
        stats = json.loads(stats_path.read_text())
        if not (stats["converged"] and stats["n"]==n and stats["k"]==k
                and np.isclose(stats["requested_retention"],ratio)):
            return False
        with np.load(array_path) as saved:
            medoids,assignments,distances,sizes = (saved[key]for key in ("medoids","assignments","distances","cluster_sizes"))
            if medoids.shape!=(k,) or assignments.shape!=(n,) or distances.shape!=(n,) or sizes.shape!=(k,):
                return False
            if not ((medoids>=0).all() and (medoids<n).all() and len(np.unique(medoids))==k
                    and (assignments>=0).all() and (assignments<k).all() and np.isfinite(distances).all()):
                return False
            return bool(np.array_equal(assignments[medoids],np.arange(k)) and (sizes>0).all()
                        and np.array_equal(np.bincount(assignments,minlength=k),sizes)
                        and np.isclose(stats["objective"],distances.astype(np.float64).sum()))
    except (OSError,ValueError,KeyError,TypeError,IndexError,EOFError,zipfile.BadZipFile):
        return False


def cluster(args: argparse.Namespace) -> None:
    data = np.load(args.output_dir / "values.npz")
    for layer in args.layers:
        values = data[f"layer_{layer}"]
        for head in (args.heads if args.heads is not None else range(len(values))):
            head_values = values[head]
            distance = None
            for ratio in args.ratios:
                k = max(1, round(ratio * len(head_values)))
                destination = args.output_dir / f"keep_{ratio:.0%}" / f"layer_{layer:02d}"
                destination.mkdir(parents=True, exist_ok=True)
                if getattr(args,"skip_existing",False) and valid_cached_cluster(
                    destination/f"head_{head:02d}.npz",destination/f"head_{head:02d}.json",len(head_values),k,ratio
                ):
                    print(f"复用已验证缓存：层 {layer+1} 头 {head:02d} 保留 {ratio:.0%}",flush=True)
                    continue
                if distance is None:
                    distance = pairwise_distances(torch.from_numpy(head_values).to(args.device))
                start = time.perf_counter()
                with torch.inference_mode():
                    results = [pam(distance, k, method, 20261014 + 100 * layer + head, args.max_swaps)
                               for method in ("build", "sample")]
                result = min(results, key=lambda item: item["objective"])
                medoids = result.pop("medoids")
                assignments = result.pop("assignments")
                nearest = result.pop("distances")
                cluster_sizes = np.bincount(assignments, minlength=k)
                np.savez_compressed(destination / f"head_{head:02d}.npz", medoids=medoids,
                                    assignments=assignments, distances=nearest, cluster_sizes=cluster_sizes)
                stats = {
                    "layer": layer, "head": head, "requested_retention": ratio,
                    "n": len(head_values), "k": k, "actual_retention": k / len(head_values),
                    "mean_distance": float(nearest.mean()), "max_distance": float(nearest.max()),
                    "elapsed_seconds": time.perf_counter() - start,
                    **result,
                    "starts": [{key: value for key, value in item.items()
                                if key not in ("medoids", "assignments", "distances")} for item in results],
                }
                (destination / f"head_{head:02d}.json").write_text(json.dumps(stats, indent=2) + "\n")
                print(f"层 {layer + 1} 头 {head:02d} 保留 {ratio:.0%}: K={k}, "
                      f"平均距离={stats['mean_distance']:.5f}, 交换={stats['swaps']}, "
                      f"收敛={stats['converged']}, {stats['elapsed_seconds']:.1f}s", flush=True)


FONT = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"


@lru_cache(maxsize=16)
def font(size, ascii_only=False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" if ascii_only else FONT
    return ImageFont.truetype(path, size)


def label(draw, xy, content, size, fill="black"):
    # Droid Sans Fallback supplies Chinese glyphs but not Latin/digits.
    # Explicit fallback runs keep the layer/head IDs and ratios readable.
    import itertools
    x, y = xy
    for ascii_only, chars in itertools.groupby(content, key=lambda char: ord(char) < 128):
        run = "".join(chars)
        chosen = font(size, ascii_only)
        draw.text((x, y + size), run, fill=fill, font=chosen, anchor="ls")
        x += draw.textlength(run, font=chosen)


def patch_box(image: Image.Image, row: int, col: int, grid: list[int]) -> tuple[float, float, float, float]:
    # Invert VGGT's width=518 resize (height rounded to a multiple of 14)
    # and possible vertical centre crop. These two Chess frames need no crop.
    width, height = image.size
    resized_h = round(height * (518 / width) / 14) * 14
    crop_y = max(0, (resized_h - 518) // 2)
    return (col * 14 * width / 518, (row * 14 + crop_y) * height / resized_h,
            (col + 1) * 14 * width / 518, ((row + 1) * 14 + crop_y) * height / resized_h)


def overlay(original, ids, grid):
    canvas = original.convert("RGBA")
    marks = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(marks)
    for patch in ids:
        row, col = divmod(int(patch), grid[1])
        box = patch_box(original, row, col, grid)
        draw.rectangle(box, fill=(255, 150, 0, 80), outline=(0, 240, 255, 240), width=2)
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 240, 40, 255), outline=(15, 15, 15, 255))
    return Image.alpha_composite(canvas, marks).convert("RGB")


def render(args: argparse.Namespace) -> None:
    import html
    metadata = json.loads((args.output_dir / "metadata.json").read_text())
    grid = metadata["patch_grid"]
    per_frame = grid[0] * grid[1]
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    rows, medoid_rows, pages, gallery = [], [], [], []
    for ratio in args.ratios:
        for layer in args.layers:
            thumbnails = []
            directory = args.output_dir / f"keep_{ratio:.0%}" / f"layer_{layer:02d}"
            gallery_heads = []
            for head in range(16):
                record = json.loads((directory / f"head_{head:02d}.json").read_text())
                arrays = np.load(directory / f"head_{head:02d}.npz")
                medoids, sizes = arrays["medoids"], arrays["cluster_sizes"]
                panels = []
                counts = []
                for frame, original in enumerate(originals):
                    local = medoids[(medoids // per_frame) == frame] % per_frame
                    counts.append(len(local))
                    marked = overlay(original, local, grid)
                    panel = Image.new("RGB", (original.width, original.height + 40), "white")
                    panel.paste(marked, (0, 40))
                    label(ImageDraw.Draw(panel), (12, 7), f"原图帧 {metadata['frame_indices'][frame]}：{len(local)} 个代表点", 20)
                    panels.append(panel)
                width, height = sum(p.width for p in panels), max(p.height for p in panels)
                image = Image.new("RGB", (width, height + 100), "white")
                draw = ImageDraw.Draw(image)
                label(draw, (12, 8), f"第 {layer + 1} 个全局层（L{layer:02d}）  注意力头 H{head:02d}  保留 {ratio:.0%}（K={len(medoids)}）", 25)
                label(draw, (12, 45), "黄色点：真实 medoid 的 patch 中心；青框：对应 patch；两帧联合聚类", 20, "#333333")
                x = 0
                for panel in panels:
                    image.paste(panel, (x, 90))
                    x += panel.width
                image.save(directory / f"head_{head:02d}_overlay.png")
                pages.append(image)
                thumb = image.resize((640, round(image.height * 640 / image.width)), Image.Resampling.LANCZOS)
                thumbnails.append(thumb)
                relative = (directory / f"head_{head:02d}_overlay.png").relative_to(args.output_dir)
                gallery_heads.append(f'<a href="{relative}"><img loading="lazy" src="{relative}" alt="H{head:02d}"></a>')
                row = {key: record[key] for key in ("layer", "head", "requested_retention", "n", "k", "actual_retention", "mean_distance", "max_distance", "swaps", "converged", "elapsed_seconds")}
                row.update(frame_0_count=counts[0], frame_20_count=counts[1])
                rows.append(row)
                for cluster_id, (medoid, size) in enumerate(zip(medoids, sizes)):
                    frame, local = divmod(int(medoid), per_frame)
                    patch_row, patch_col = divmod(local, grid[1])
                    box = patch_box(originals[frame], patch_row, patch_col, grid)
                    medoid_rows.append({"layer": layer, "head": head, "retention": ratio,
                                        "cluster": cluster_id, "joint_patch_index": int(medoid),
                                        "frame_index": metadata["frame_indices"][frame],
                                        "patch_row": patch_row, "patch_col": patch_col,
                                        "original_x": (box[0] + box[2]) / 2,
                                        "original_y": (box[1] + box[3]) / 2,
                                        "cluster_size": int(size)})
            tile_w, tile_h = thumbnails[0].size
            sheet = Image.new("RGB", (4 * tile_w, 4 * tile_h + 70), "#eeeeee")
            label(ImageDraw.Draw(sheet), (20, 12), f"VGGT 第 {layer + 1} 个全局层：16 个头的 Value k-medoids，保留 {ratio:.0%}", 30)
            for head, thumb in enumerate(thumbnails):
                sheet.paste(thumb, ((head % 4) * tile_w, 70 + (head // 4) * tile_h))
            sheet_path = args.output_dir / f"overview_layer_{layer:02d}_keep_{ratio:.0%}.jpg"
            sheet.save(sheet_path, quality=94)
            gallery.append(f'<h2>第 {layer + 1} 个全局层 · 保留 {ratio:.0%}</h2>'
                           f'<a href="{sheet_path.name}">查看 16 头总览</a><div class="grid">' + "".join(gallery_heads) + "</div>")
    for name, records in (("metrics.csv", rows), ("medoid_locations.csv", medoid_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    pages[0].save(args.output_dir / "all_heads_value_medoids.pdf", save_all=True, append_images=pages[1:], resolution=120)
    layer_names = "、".join(str(layer + 1) for layer in args.layers)
    frame_names = "、".join(str(frame) for frame in metadata["frame_indices"])
    title = f"VGGT 第{layer_names}个全局层 Value k-medoids：{metadata['scene']} 第{frame_names}帧"
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
                '<style>body{max-width:1500px;margin:32px auto;padding:0 16px;font-family:system-ui;line-height:1.65}'
                '.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}img{width:100%}'
                'a{color:#1261a0}h2{margin-top:36px}</style><h1>' + title + '</h1>'
                f'<p>两帧共 {metadata["values_per_head"]} 个 patch values，在每层每头独立联合聚类。'
                + '；'.join(f'保留 {ratio:.0%}={max(1, round(ratio * metadata["values_per_head"]))} 个代表点' for ratio in args.ratios) + '。'
                '黄色点标记真实 medoid 的 patch 中心，青框显示该 patch 在原图的位置。点击图片查看原尺寸。</p>'
                '<p>距离：原始 V 的欧氏距离（不做 L2 归一化）；PAM 穷举最优单次交换；BUILD 和固定种子初始化各跑一次，取目标更小者。'
                '这是局部最优搜索，不保证全局最优；特殊 token 不参与聚类。层和头的 L/H 编号从 0 开始。</p>'
                f'<p><a href="all_heads_value_medoids.pdf">下载全部 {len(rows)} 张图 PDF</a> · '
                '<a href="metrics.csv">每头指标 CSV</a> · <a href="medoid_locations.csv">全部代表点坐标 CSV</a></p>'
                + "".join(gallery) + '</html>')
    (args.output_dir / "index.html").write_text(document)
    summary = {
        "head_ratio_runs": len(rows), "all_converged": all(row["converged"] for row in rows),
        "nonconverged": [row for row in rows if not row["converged"]],
        "per_layer_ratio": [{"layer": layer, "retention": ratio,
                             "mean_distance_across_heads": float(np.mean([r["mean_distance"] for r in rows if r["layer"] == layer and r["requested_retention"] == ratio])),
                             "frame_0_medoids": sum(r["frame_0_count"] for r in rows if r["layer"] == layer and r["requested_retention"] == ratio),
                             "frame_20_medoids": sum(r["frame_20_count"] for r in rows if r["layer"] == layer and r["requested_retention"] == ratio)}
                            for ratio in args.ratios for layer in args.layers],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "cluster", "render"))
    parser.add_argument("--source-metadata", type=Path, default=Path("outputs/global_attention/7scenes_chess_f000000_f000020/metadata.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/vggt/model_tracker_fixed_e20.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/value_medoids/chess_f000000_f000020"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--heads", nargs="+", type=int, help="Optional independent head shard for clustering")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--max-swaps", type=int, default=200)
    parser.add_argument("--skip-existing",action="store_true",help="Resume complete converged clustering fits after consistency checks")
    args = parser.parse_args()
    if any(not 0 <= layer < 24 for layer in args.layers):
        parser.error("Global layer indices must be between 0 and 23 (zero-based)")
    args.layers = sorted(set(args.layers))
    if args.heads is not None:
        if any(not 0 <= head < 16 for head in args.heads):
            parser.error("Head indices must be between 0 and 15")
        args.heads = sorted(set(args.heads))
    if any(not 0 < ratio < 1 for ratio in args.ratios):
        parser.error("Retained ratios must be between 0 and 1")
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    {"extract": extract, "cluster": cluster, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
