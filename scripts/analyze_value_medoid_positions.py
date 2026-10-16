#!/usr/bin/env python3
"""Compare medoid positions across layers, keeping the two image domains separate.

Only saved patch indices are used. There is no feature-space comparison or
cross-image geometric matching. Random baselines condition on every head's
per-frame medoid count. Head matching is repeated under each permutation to
account for the selection advantage of choosing a matching after seeing data.
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
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln


def render_overlap_example(metadata, source_ids, target_ids, source_head, target_head, destination):
    from PIL import Image, ImageDraw
    from scripts.visualize_vggt_value_medoids import label, patch_box

    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    grid = metadata["patch_grid"]
    n = grid[0] * grid[1]
    common = np.intersect1d(source_ids, target_ids)
    image = Image.new("RGB", (sum(im.width for im in originals), max(im.height for im in originals) + 145), "white")
    draw = ImageDraw.Draw(image)
    label(draw, (14, 6), f"位置重合示例：第2层 H{source_head:02d} 与第3层 H{target_head:02d}，保留5%", 25)
    label(draw, (14, 43), f"青色：仅第2层；橙色：仅第3层；绿色：共同位置，共 {len(common)}/104={len(common)/104:.2%}", 21)
    label(draw, (14, 75), "此图展示第2、3层中重合最多的头配对；不代表所有头的平均结果。", 18)
    offset = 0
    colors = ((np.setdiff1d(source_ids, common), (0, 225, 255, 255)),
              (np.setdiff1d(target_ids, common), (255, 140, 0, 255)),
              (common, (80, 255, 60, 255)))
    for frame, original in enumerate(originals):
        marked = original.convert("RGBA")
        marks = Image.new("RGBA", marked.size)
        marker = ImageDraw.Draw(marks)
        for indices, color in colors:
            for patch in indices[indices // n == frame] % n:
                row, col = divmod(int(patch), grid[1])
                box = patch_box(original, row, col, grid)
                x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
                radius = 5 if color == (80, 255, 60, 255) else 3
                marker.rectangle(box, outline=color, width=2)
                marker.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="black")
        marked = Image.alpha_composite(marked, marks).convert("RGB")
        image.paste(marked, (offset, 145))
        frame_common = int((common // n == frame).sum())
        label(draw, (offset+14, 111), f"原图帧 {metadata['frame_indices'][frame]}：{frame_common} 个共同位置", 20)
        offset += original.width
    image.save(destination / "strongest_L1_L2_overlap_on_images.png")


def no_hit_probability(n, excluded, k):
    """Probability of missing excluded cells in a uniform k-subset of n."""
    excluded = np.asarray(excluded)
    remaining = n - excluded
    log_probability = (gammaln(remaining + 1) - gammaln(remaining - k + 1)
                       - gammaln(n + 1) + gammaln(n - k + 1))
    return np.where(remaining >= k, np.exp(log_probability), 0.0)


class PositionComparison:
    def __init__(self, height, width):
        self.n = height * width
        yy, xx = np.indices((height, width))
        xy = np.stack([yy.ravel(), xx.ravel()], axis=1)
        self.distance = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(2))
        self.sorted_distance = np.sort(self.distance, axis=1)
        self.random_cache = {}

    def random_nearest(self, k):
        if k not in self.random_cache:
            survival = no_hit_probability(self.n, np.arange(1, self.n), k)
            # E[min D] = sum (consecutive sorted distance gaps) * P(no hit yet).
            mean_distance = np.diff(self.sorted_distance, axis=1) @ survival
            near1 = 1 - no_hit_probability(self.n, (self.distance <= 1).sum(1), k)
            near2 = 1 - no_hit_probability(self.n, (self.distance <= 2).sum(1), k)
            self.random_cache[k] = mean_distance, near1, near2
        return self.random_cache[k]

    def compare(self, a, b):
        directed, random_distances, random_near1, random_near2 = [], [], [], []
        expected_common = 0.0
        for frame in (0, 1):
            aa = a[a // self.n == frame] % self.n
            bb = b[b // self.n == frame] % self.n
            if not len(aa) or not len(bb):
                raise ValueError("Nearest-neighbour statistics require medoids in both frames")
            d = self.distance[np.ix_(aa, bb)]
            directed.extend([d.min(1), d.min(0)])
            expected_common += len(aa) * len(bb) / self.n
            for source, target in ((aa, bb), (bb, aa)):
                d0, n1, n2 = self.random_nearest(len(target))
                random_distances.extend(d0[source])
                random_near1.extend(n1[source])
                random_near2.extend(n2[source])
        nearest = np.concatenate(directed)
        intersection = len(np.intersect1d(a, b))
        return {
            "intersection_count": intersection,
            "exact_overlap": 2 * intersection / (len(a) + len(b)),
            "jaccard": intersection / (len(a) + len(b) - intersection),
            "within_1_patch": float((nearest <= 1).mean()),
            "within_2_patches": float((nearest <= 2).mean()),
            "nearest_distance_patches": float(nearest.mean()),
            "random_exact_overlap": 2 * expected_common / (len(a) + len(b)),
            "random_within_1_patch": float(np.mean(random_near1)),
            "random_within_2_patches": float(np.mean(random_near2)),
            "random_nearest_distance_patches": float(np.mean(random_distances)),
        }


def permutation_baseline(a, b, n, repetitions, rng):
    """Apply the same within-frame location permutation to every target head.

    This preserves all cross-head intersections in the target layer and every
    head's per-frame counts, and destroys alignment with the source layer.
    Each replicate refits the global one-to-one optimal head assignment.
    """
    masks = []
    for sets in (a, b):
        mask = np.zeros((len(sets), 2 * n), dtype=np.float32)
        for head, ids in enumerate(sets):
            mask[head, ids] = 1
        masks.append(mask)
    k = len(a[0])
    samples = {key: [] for key in ("same_head", "all_pairs", "row_best", "one_to_one")}
    for _ in range(repetitions):
        permutation = np.concatenate([rng.permutation(n), n + rng.permutation(n)])
        overlap = (masks[0] @ masks[1][:, permutation].T) / k
        source, target = linear_sum_assignment(-overlap)
        samples["same_head"].append(float(np.diag(overlap).mean()))
        samples["all_pairs"].append(float(overlap.mean()))
        samples["row_best"].append(float(overlap.max(1).mean()))
        samples["one_to_one"].append(float(overlap[source, target].mean()))
    return {key: np.asarray(values) for key, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/value_medoids/chess_f000000_f000020"))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    height, width = metadata["patch_grid"]
    comparison = PositionComparison(height, width)
    destination = args.input_dir / "cross_layer_positions"
    destination.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    all_rows, summaries, matches, matrices, head_identity_tests = [], [], [], {}, []
    label_rng = np.random.default_rng(args.seed + 1)
    for ratio in (0.01, 0.05):
        sets = [[np.load(args.input_dir / f"keep_{ratio:.0%}" / f"layer_{layer:02d}" / f"head_{head:02d}.npz")["medoids"]
                 for head in range(16)] for layer in range(3)]
        for layer_a, layer_b in combinations(range(3), 2):
            pair_rows = []
            for head_a in range(16):
                for head_b in range(16):
                    row = {"retention": ratio, "source_layer": layer_a, "target_layer": layer_b,
                           "source_head": head_a, "target_head": head_b,
                           **comparison.compare(sets[layer_a][head_a], sets[layer_b][head_b])}
                    pair_rows.append(row)
            all_rows.extend(pair_rows)
            matrix = np.array([r["exact_overlap"] for r in pair_rows]).reshape(16, 16)
            # Keep all medoid positions intact and shuffle only target head
            # labels to test whether matching identical head numbers is special.
            identity_null = np.array([matrix[np.arange(16), label_rng.permutation(16)].mean()
                                      for _ in range(10000)])
            identity_observed = float(np.diag(matrix).mean())
            head_identity_tests.append({
                "retention": ratio, "source_layer": layer_a, "target_layer": layer_b,
                "same_head_overlap": identity_observed,
                "arbitrary_head_overlap_mean": float(identity_null.mean()),
                "p_same_head_better_than_random_head_labels": float((1 + (identity_null >= identity_observed - 1e-8).sum()) / 10001),
                "permutations": 10000,
            })
            rows, columns = linear_sum_assignment(-matrix)
            matrices[f"keep_{ratio:.0%}_L{layer_a}_L{layer_b}"] = matrix
            selectors = {
                "same_head": np.arange(16) * 17,
                "all_pairs": np.arange(256),
                "row_best": np.arange(16) * 16 + matrix.argmax(1),
                "one_to_one": rows * 16 + columns,
            }
            null = permutation_baseline(sets[layer_a], sets[layer_b], comparison.n, args.permutations, rng)
            for name, ids in selectors.items():
                selected = [pair_rows[int(index)] for index in ids]
                summary = {"retention": ratio, "source_layer": layer_a, "target_layer": layer_b,
                           "head_comparison": name}
                for key in selected[0]:
                    if key not in ("retention", "source_layer", "target_layer", "source_head", "target_head"):
                        summary[key] = float(np.mean([r[key] for r in selected]))
                samples = null[name]
                summary.update(
                    permutation_overlap_mean=float(samples.mean()),
                    permutation_overlap_q025=float(np.quantile(samples, 0.025)),
                    permutation_overlap_q975=float(np.quantile(samples, 0.975)),
                    permutation_p_greater_equal=float((1 + (samples >= summary["exact_overlap"] - 1e-8).sum()) / (1 + len(samples))),
                )
                summaries.append(summary)
            for row, column in zip(rows, columns):
                matches.append({**pair_rows[int(row * 16 + column)], "assignment": "one_to_one_max_exact_overlap"})
            print(f"保留 {ratio:.0%}, L{layer_a}→L{layer_b}: 同编号重合={np.diag(matrix).mean():.2%}, "
                  f"一对一匹配={matrix[rows, columns].mean():.2%}, "
                  f"随机匹配={null['one_to_one'].mean():.2%}", flush=True)
    for name, records in (("all_head_pairs.csv", all_rows), ("summary.csv", summaries), ("one_to_one_head_matches.csv", matches),
                          ("head_identity_tests.csv", head_identity_tests)):
        with (destination / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    np.savez_compressed(destination / "overlap_matrices.npz", **matrices)
    report = {"input_metadata": metadata, "permutations": args.permutations, "seed": args.seed,
              "scope": "same image, same patch coordinates across global layers; no cross-image matching",
              "exact_overlap_definition": "2*intersection/(K_source+K_target); equal K gives intersection/K",
              "spatial_metric": "symmetric nearest-neighbour Euclidean distance in patch-row/column units, matching within each frame only",
              "permutation_null": "shared target-head location permutation within each frame; preserves target-layer head overlap structure and per-frame counts; refit head matching in every permutation",
              "caution": "single sequence, two frames; permutation p-values conditional on these observed sets, uncorrected for multiple layer-pair tests",
              "summary": summaries, "head_identity_tests": head_identity_tests}
    (destination / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for row, ratio in enumerate((0.01, 0.05)):
        for column, (a, b) in enumerate(combinations(range(3), 2)):
            matrix = matrices[f"keep_{ratio:.0%}_L{a}_L{b}"]
            axis = axes[row, column]
            image = axis.imshow(100 * matrix, vmin=0, vmax=20 if ratio == 0.01 else 25, cmap="viridis")
            ii, jj = linear_sum_assignment(-matrix)
            axis.scatter(jj, ii, marker="s", facecolors="none", edgecolors="red", s=85, linewidths=1.2)
            axis.set_xticks(range(16)); axis.set_yticks(range(16))
            axis.set_xlabel(f"第 {b + 1} 层的头编号")
            axis.set_ylabel(f"第 {a + 1} 层的头编号")
            axis.set_title(f"保留 {ratio:.0%}：第 {a + 1} 层 → 第 {b + 1} 层")
            figure.colorbar(image, ax=axis, shrink=0.8, label="完全相同位置的比例（%）")
    figure.suptitle("跨层代表性 Value 的位置重合：红框表示最优一对一头匹配", fontsize=17)
    figure.savefig(destination / "overlap_heatmaps.png", dpi=170)
    plt.close(figure)
    # A separate summary chart makes distance/overlap scales and the effect of
    # head selection clear without inferring persistence from a heatmap alone.
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, ratio in zip(axes, (0.01, 0.05)):
        x = np.arange(3)
        for offset, name, label, color in ((-0.19, "same_head", "同编号头", "#277da8"),
                                           (0.19, "one_to_one", "最优一对一头匹配", "#e68a24")):
            points = [next(r for r in summaries if r["retention"] == ratio and r["source_layer"] == a and r["target_layer"] == b and r["head_comparison"] == name)
                      for a, b in combinations(range(3), 2)]
            axis.bar(x + offset, [100*r["exact_overlap"] for r in points], width=0.35, label=label, color=color)
            baseline = np.array([100*r["permutation_overlap_mean"] for r in points])
            limits = np.array([[100*r["permutation_overlap_q025"] for r in points], [100*r["permutation_overlap_q975"] for r in points]])
            axis.errorbar(x + offset, baseline, yerr=np.maximum(0, np.stack([baseline-limits[0], limits[1]-baseline])),
                          fmt="k_", markersize=15, capsize=3, label="随机位置基线（95%区间）" if name == "same_head" else None)
        axis.set_xticks(x, ["第1层→第2层", "第1层→第3层", "第2层→第3层"])
        axis.set_ylabel("完全相同位置的比例（%）")
        axis.set_title(f"保留 {ratio:.0%}")
        axis.legend(fontsize=9)
    figure.savefig(destination / "overlap_vs_random.png", dpi=180)
    plt.close(figure)
    strongest = max((r for r in all_rows if r["retention"] == 0.05 and r["source_layer"] == 1 and r["target_layer"] == 2),
                    key=lambda row: row["exact_overlap"])
    source_head, target_head = strongest["source_head"], strongest["target_head"]
    source_ids = np.load(args.input_dir / "keep_5%" / "layer_01" / f"head_{source_head:02d}.npz")["medoids"]
    target_ids = np.load(args.input_dir / "keep_5%" / "layer_02" / f"head_{target_head:02d}.npz")["medoids"]
    render_overlap_example(metadata, source_ids, target_ids, source_head, target_head, destination)


if __name__ == "__main__":
    main()
