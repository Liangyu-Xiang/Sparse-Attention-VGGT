#!/usr/bin/env python3
"""Match heads by medoid neighbourhoods, without any head-index correspondence.

For each radius, a head-pair score is the symmetric fraction of medoids with
another layer's medoid within that radius in the SAME image. Target heads
are selected independently per source head by this score (many-to-one is
allowed); a global one-to-one assignment is reported separately. Each random
baseline replicate refits both matching rules using the same radius.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

from scripts.analyze_value_medoid_positions import PositionComparison


def point_masks(sets, n):
    mask = np.zeros((len(sets), 2 * n), dtype=np.float32)
    for head, indices in enumerate(sets):
        mask[head, indices] = 1
    return mask


def neighborhood_masks(masks, height, width, radius):
    if radius == 0:
        return masks
    reach = int(np.ceil(radius))
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    disk = (yy * yy + xx * xx <= radius * radius + 1e-8)[None, None]
    # Singleton head and frame footprint dimensions forbid dilation across
    # different heads or images, including the image-boundary token indices.
    result = maximum_filter(masks.reshape(len(masks), 2, height, width), footprint=disk, mode="constant", cval=0)
    return result.reshape(masks.shape)


def scores(a, b, covered_a, covered_b):
    count_a = a.sum(1)[:, None]
    count_b = b.sum(1)[None, :]
    forward = (a @ covered_b.T) / count_a
    backward = (covered_a @ b.T) / count_b
    return (forward + backward) * 0.5


def selections(matrix):
    n = len(matrix)
    ii, jj = linear_sum_assignment(-matrix)
    return {
        "row_best": (np.arange(n), matrix.argmax(1)),
        "one_to_one": (ii, jj),
        "all_pairs": np.indices(matrix.shape).reshape(2, -1),
    }


def null_scores(a, b, height, width, radii, repetitions, rng):
    n = height * width
    covered_a = {radius: neighborhood_masks(a, height, width, radius) for radius in radii}
    samples = {radius: {strategy: [] for strategy in ("row_best", "one_to_one", "all_pairs")} for radius in radii}
    for _ in range(repetitions):
        permutation = np.concatenate([rng.permutation(n), n + rng.permutation(n)])
        permuted_b = b[:, permutation]
        for radius in radii:
            covered_b = neighborhood_masks(permuted_b, height, width, radius)
            matrix = scores(a, permuted_b, covered_a[radius], covered_b)
            for strategy, (ii, jj) in selections(matrix).items():
                samples[radius][strategy].append(float(matrix[ii, jj].mean()))
    return {radius: {strategy: np.asarray(values) for strategy, values in strategies.items()}
            for radius, strategies in samples.items()}


def render_example(metadata, a, b, source_head, target_head, radius, score, destination, ratio):
    from PIL import Image, ImageDraw
    from scripts.visualize_vggt_value_medoids import label, patch_box

    grid = metadata["patch_grid"]
    n = grid[0] * grid[1]
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    image = Image.new("RGB", (sum(im.width for im in originals), max(im.height for im in originals) + 145), "white")
    draw = ImageDraw.Draw(image)
    label(draw, (12, 5), f"第2层 H{source_head:02d} 与第3层 H{target_head:02d}：保留{ratio:.0%}，允许{radius:g} patch偏移", 24)
    label(draw, (12, 43), f"绿色点：邻域内找到另一层代表点；灰点：未找到；双向命中率 {score:.2%}", 20)
    label(draw, (12, 75), "青框=第2层；橙框=第3层；此图为这两个层中最佳头对，非全头平均。", 19)
    offset = 0
    for frame, original in enumerate(originals):
        aa, bb = a[a // n == frame] % n, b[b // n == frame] % n
        ax = np.stack([aa // grid[1], aa % grid[1]], 1)
        bx = np.stack([bb // grid[1], bb % grid[1]], 1)
        distance = np.sqrt(((ax[:, None] - bx[None, :]) ** 2).sum(2))
        ahit, bhit = distance.min(1) <= radius, distance.min(0) <= radius
        overlay = Image.new("RGBA", original.size)
        marker = ImageDraw.Draw(overlay)
        for indices, hits, color in ((aa, ahit, (0, 225, 255, 255)), (bb, bhit, (255, 140, 0, 255))):
            for patch, hit in zip(indices, hits):
                row, col = divmod(int(patch), grid[1])
                box = patch_box(original, row, col, grid)
                x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
                marker.rectangle(box, outline=color, width=2)
                marker.ellipse((x-3, y-3, x+3, y+3), fill=(70, 255, 60, 255) if hit else (150, 150, 150, 255), outline="black")
        marked = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")
        image.paste(marked, (offset, 145))
        label(draw, (offset+12, 110), f"原图帧 {metadata['frame_indices'][frame]}：第2层命中{ahit.sum()}/{len(aa)}，第3层命中{bhit.sum()}/{len(bb)}", 18)
        offset += original.width
    image.save(destination / f"best_layer_01_02_radius_{radius:g}_on_images.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/value_medoids/chess_f000000_f000020"))
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--radii", nargs="+", type=float, default=[0, 1, 2])
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    height, width = metadata["patch_grid"]
    n = height * width
    radii = sorted(set(args.radii))
    if args.permutations < 1 or any(r < 0 for r in radii):
        parser.error("Positive permutations and nonnegative radii are required")
    sets = [[np.load(args.input_dir / f"keep_{args.ratio:.0%}" / f"layer_{layer:02d}" / f"head_{head:02d}.npz")["medoids"]
             for head in range(16)] for layer in range(3)]
    k = len(sets[0][0])
    assert all(len(indices) == k for layer in sets for indices in layer)
    destination = args.input_dir / f"cross_layer_neighborhoods_keep_{args.ratio:.0%}"
    destination.mkdir(exist_ok=True)
    comparison = PositionComparison(height, width)
    masks = [point_masks(layer_sets, n) for layer_sets in sets]
    covers = {radius: [neighborhood_masks(mask, height, width, radius) for mask in masks] for radius in radii}
    rng = np.random.default_rng(args.seed)
    pair_records, summaries, matched_rows, matrices = [], [], [], {}
    pairs = list(combinations(range(3), 2))
    for source_layer, target_layer in pairs:
        base_rows = {}
        for ha in range(16):
            for hb in range(16):
                base_rows[ha, hb] = comparison.compare(sets[source_layer][ha], sets[target_layer][hb])
        null = null_scores(masks[source_layer], masks[target_layer], height, width, radii, args.permutations, rng)
        for radius in radii:
            matrix = scores(masks[source_layer], masks[target_layer], covers[radius][source_layer], covers[radius][target_layer])
            matrices[f"L{source_layer}_L{target_layer}_radius_{radius:g}"] = matrix
            prefix = {"retention": args.ratio, "k": k, "source_layer": source_layer, "target_layer": target_layer, "radius_patches": radius}
            for ha in range(16):
                for hb in range(16):
                    pair_records.append({**prefix, "source_head": ha, "target_head": hb,
                                         "symmetric_neighborhood_hit": float(matrix[ha, hb]), **base_rows[ha, hb]})
            for strategy, (ii, jj) in selections(matrix).items():
                observed = float(matrix[ii, jj].mean())
                samples = null[radius][strategy]
                row = {**prefix, "head_matching": strategy, "symmetric_neighborhood_hit": observed,
                       "random_refitted_matching_mean": float(samples.mean()),
                       "random_q025": float(np.quantile(samples, 0.025)),
                       "random_q975": float(np.quantile(samples, 0.975)),
                       "excess_percentage_points": 100 * (observed - float(samples.mean())),
                       "permutation_p_ge": float((1 + (samples >= observed-1e-7).sum()) / (1 + len(samples))),
                       "mean_exact_overlap": float(np.mean([base_rows[int(a), int(b)]["exact_overlap"] for a, b in zip(ii, jj)])),
                       "mean_nearest_distance_patches": float(np.mean([base_rows[int(a), int(b)]["nearest_distance_patches"] for a, b in zip(ii, jj)]))}
                summaries.append(row)
                if strategy != "all_pairs":
                    for ha, hb in zip(ii, jj):
                        matched_rows.append({**prefix, "head_matching": strategy, "source_head": int(ha), "target_head": int(hb),
                                             "symmetric_neighborhood_hit": float(matrix[ha, hb]), **base_rows[int(ha), int(hb)]})
                if strategy == "row_best":
                    print(f"第{source_layer+1}→{target_layer+1}层 半径{radius:g}：逐源头最佳命中={observed:.2%}，"
                          f"随机也重匹配={samples.mean():.2%}，增益={row['excess_percentage_points']:.2f}百分点", flush=True)
    for filename, records in (("all_head_pairs.csv", pair_records), ("summary.csv", summaries), ("matched_heads.csv", matched_rows)):
        with (destination / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    np.savez_compressed(destination / "neighborhood_matrices.npz", **matrices)
    (destination / "summary.json").write_text(json.dumps({
        "retention": args.ratio, "k": k, "input_metadata": metadata, "radii": radii,
        "head_identity_assumed": False,
        "primary_matching": "row_best: each source head independently chooses target head by this radius's symmetric neighborhood hit rate; many-to-one head matching allowed",
        "secondary_matching": "one_to_one: globally optimal head assignment using this radius's scores",
        "metric": "mean of source-to-target and target-to-source nearest-medoid hit fractions; same frame only; Euclidean grid units; multiple points may share a nearest medoid",
        "null": "shared target-head position permutations within each frame; preserve per-head frame counts and target cross-head intersections; refit heads separately for each radius in each replicate",
        "permutations": args.permutations, "seed": args.seed,
        "limitation": "single sequence and two frames; tests conditional on observed point sets; no multiple-testing correction",
        "summary": summaries}, ensure_ascii=False, indent=2) + "\n")
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    figure, axes = plt.subplots(len(radii), 3, figsize=(16, 4.8 * len(radii)), constrained_layout=True, squeeze=False)
    for row, radius in enumerate(radii):
        for col, (a, b) in enumerate(pairs):
            matrix = matrices[f"L{a}_L{b}_radius_{radius:g}"]
            axis = axes[row, col]
            artist = axis.imshow(100*matrix, vmin=0, vmax=100, cmap="viridis")
            axis.scatter(matrix.argmax(1), np.arange(16), facecolors="none", edgecolors="red", marker="s", s=70)
            axis.set_xticks(range(16)); axis.set_yticks(range(16))
            axis.set_xlabel(f"第{b+1}层目标头"); axis.set_ylabel(f"第{a+1}层源头")
            axis.set_title(f"第{a+1}→{b+1}层，允许{radius:g} patch偏移")
            figure.colorbar(artist, ax=axis, shrink=0.85, label="双向邻域命中率（%）")
    figure.suptitle(f"保留{args.ratio:.0%}：按位置邻域选择跨层头匹配（红框），无头编号对应假设", fontsize=16)
    figure.savefig(destination / "neighborhood_heatmaps.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, len(radii), figsize=(6*len(radii), 5), constrained_layout=True, squeeze=False)
    for axis, radius in zip(axes.flat, radii):
        selected = [r for r in summaries if r["radius_patches"] == radius and r["head_matching"] == "row_best"]
        values = np.array([100*r["symmetric_neighborhood_hit"] for r in selected])
        baseline = np.array([100*r["random_refitted_matching_mean"] for r in selected])
        errors = np.array([[100*(r["random_refitted_matching_mean"]-r["random_q025"]) for r in selected],
                           [100*(r["random_q975"]-r["random_refitted_matching_mean"]) for r in selected]])
        x = np.arange(3)
        axis.bar(x-0.18, values, 0.34, label="跨层最佳匹配", color="#2a86a3")
        axis.bar(x+0.18, baseline, 0.34, yerr=np.maximum(0, errors), capsize=4, label="随机位置也最佳匹配", color="#c5c5c5")
        for i, v in enumerate(values):
            axis.text(i-0.18, v+1, f"{v:.1f}%", ha="center", fontsize=10)
        axis.set_xticks(x, [f"第{a+1}→{b+1}层" for a, b in pairs])
        axis.set_ylim(0, 100); axis.set_ylabel("双向邻域命中率（%）")
        axis.set_title(f"保留{args.ratio:.0%}，允许{radius:g} patch偏移")
        axis.legend(fontsize=9)
    figure.savefig(destination / "neighborhood_vs_random.png", dpi=180)
    plt.close(figure)
    for radius in radii:
        matrix = matrices[f"L1_L2_radius_{radius:g}"]
        ha, hb = np.unravel_index(matrix.argmax(), matrix.shape)
        render_example(metadata, sets[1][ha], sets[2][hb], ha, hb, radius, float(matrix[ha, hb]), destination, args.ratio)


if __name__ == "__main__":
    main()
