#!/usr/bin/env python3
"""Equal-budget medoid vs spatial-uniform local attention fidelity experiment.

Primary uniform baseline uses the same per-frame counts as each medoid set.
Uniform sampling partitions each image into rectangular cells and selects one
patch per cell. The centre is fixed without looking at Q/K/V. Four additional
within-cell phases provide an explicitly labelled post-hoc oracle baseline.
All candidates retain special keys; standard softmax is renormalised over
selected keys, with no medoid cluster-size weighting. The dense forward is
never modified, so errors are local, not accumulated across layers.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from scripts.visualize_vggt_value_medoids import pairwise_distances


PHASES = ((0.5, 0.5), (0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75))


def uniform_cells(height, width, count, phase=(0.5, 0.5)):
    """Exactly count unique spatially stratified patches, independent of V/Q/K."""
    if not 0 <= count <= height * width:
        raise ValueError("Invalid per-frame budget")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    bands = min(height, max(int(np.ceil(count / width)), round(np.sqrt(count * height / width))))
    band_counts = np.diff(np.floor(np.linspace(0, count, bands + 1)).astype(int))
    row_edges = np.floor(np.concatenate([[0], np.cumsum(band_counts)]) * height / count).astype(int)
    # With our geometry and ~10% budgets there are several rows per band.
    if np.any(np.diff(row_edges) <= 0):
        raise ValueError("Budget cannot form nonempty rectangular row bands")
    selected = []
    for band, columns in enumerate(band_counts):
        col_edges = np.floor(np.linspace(0, width, int(columns) + 1)).astype(int)
        row = min(row_edges[band+1]-1, row_edges[band] + int(phase[0] * (row_edges[band+1]-row_edges[band])))
        for left, right in zip(col_edges[:-1], col_edges[1:]):
            col = min(right-1, left + int(phase[1] * (right-left)))
            selected.append(int(row * width + col))
    assert len(selected) == count and len(set(selected)) == count
    return np.asarray(selected, dtype=np.int64)


def joint_uniform(height, width, counts, phase=(0.5, 0.5)):
    return np.concatenate([frame * height * width + uniform_cells(height, width, count, phase)
                           for frame, count in enumerate(counts)])


def fidelity(logits, values, selected_global, dense_target, probabilities):
    selected_logits = logits[:, selected_global]
    output = selected_logits.softmax(-1) @ values[selected_global]
    squared_error = (output - dense_target).square().sum(dtype=torch.float64)
    energy = dense_target.square().sum(dtype=torch.float64).clamp_min(1e-12)
    return {
        "attention_output_nmse": float(squared_error / energy),
        "attention_output_cosine": float(F.cosine_similarity(output, dense_target, dim=-1).mean()),
        "dense_attention_mass_retained": float(probabilities[:, selected_global].sum(-1).mean()),
    }


class Profiler:
    def __init__(self, args, metadata):
        self.args, self.metadata = args, metadata
        self.rows = []
        self.cache = np.load(args.input_dir / "values.npz")
        self.out = args.input_dir / "vs_uniform_10pct"
        self.out.mkdir(exist_ok=True)

    def hook(self, layer):
        def capture(module, inputs, kwargs, _output):
            x = inputs[0]
            qkv = module.qkv(x).reshape(1, x.shape[1], 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
            query, key = module.q_norm(query), module.k_norm(key)
            if module.rope is not None:
                query, key = module.rope(query, kwargs.get("pos")), module.rope(key, kwargs.get("pos"))
            with torch.autocast("cuda", enabled=False):
                self.evaluate(layer, query[0].float(), key[0].float(), value[0].float(), module.scale)
        return capture

    def evaluate(self, layer, query, key, values, scale):
        height, width = self.metadata["patch_grid"]
        per_frame = height * width
        frame_tokens = per_frame + 5
        device = query.device
        special = torch.tensor([frame * frame_tokens + s for frame in (0, 1) for s in range(5)], device=device)
        patch = torch.tensor([frame * frame_tokens + 5 + p for frame in (0, 1) for p in range(per_frame)], device=device)
        assert query.shape[1] == 2 * frame_tokens
        for head in range(16):
            head_file = self.args.input_dir / "keep_10%" / f"layer_{layer:02d}" / f"head_{head:02d}.npz"
            medoids = np.load(head_file)["medoids"]
            patch_values = values[head, patch]
            # Verify the dense forward exactly reproduces the V used to select
            # the saved medoids, preventing mismatched images/layers/precision.
            np.testing.assert_array_equal(patch_values.cpu().numpy(), self.cache[f"layer_{layer}"][head])
            distance = pairwise_distances(patch_values)
            counts = [int((medoids // per_frame == f).sum()) for f in (0, 1)]
            balanced = [len(medoids)//2, len(medoids)//2]
            balanced[head % 2] += len(medoids) % 2
            supports = {"medoids": medoids, "uniform_matched_center": joint_uniform(height, width, counts)}
            for phase_id, phase in enumerate(PHASES[1:], 1):
                supports[f"uniform_matched_phase{phase_id}"] = joint_uniform(height, width, counts, phase)
            supports["uniform_balanced_center"] = joint_uniform(height, width, balanced)
            logits = (query[head, patch] @ key[head].T) * scale
            probabilities = logits.softmax(-1)
            dense_target = probabilities @ values[head]
            patch_mass = probabilities[:, patch].sum(-1).clamp_min(1e-12)
            saved = {}
            for method, support in supports.items():
                indices = torch.as_tensor(support, device=device, dtype=torch.long)
                global_support = torch.cat([special, patch[indices]])
                metrics = fidelity(logits, values[head], global_support, dense_target, probabilities)
                metrics["dense_patch_attention_fraction_retained"] = float((probabilities[:, patch[indices]].sum(-1) / patch_mass).mean())
                nearest = distance[:, indices].amin(1)
                metrics["value_mean_nearest_l2"] = float(nearest.mean())
                metrics["value_nearest_reconstruction_nmse"] = float(nearest.square().sum(dtype=torch.float64) / patch_values.square().sum(dtype=torch.float64).clamp_min(1e-12))
                row = {"layer": layer, "head": head, "method": method,
                       "patch_keys": len(support), "special_keys": len(special), "patch_queries": len(patch),
                       "frame_0_keys": int((support // per_frame == 0).sum()),
                       "frame_20_keys": int((support // per_frame == 1).sum()), **metrics}
                self.rows.append(row)
                saved[method] = support
            np.savez_compressed(self.out / f"layer_{layer:02d}_head_{head:02d}_supports.npz", **saved)
            recent = self.rows[-len(supports):]
            medoid_score, uniform_score = recent[0]["attention_output_nmse"], recent[1]["attention_output_nmse"]
            print(f"第{layer+1}层 H{head:02d}: medoid NMSE={medoid_score:.6g}, uniform={uniform_score:.6g}, "
                  f"较优={'medoid' if medoid_score < uniform_score else 'uniform'}", flush=True)


def evaluate(args):
    from scripts.overlay_vggt_query_attention import build_model, run_to_global_layer
    from saf3r.utils.data_utils import load_and_process_images_vggt
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    images = load_and_process_images_vggt(metadata["image_paths"]).to(args.device)
    model = build_model(Path(metadata["checkpoint"]), args.device)
    profiler = Profiler(args, metadata)
    handles = [model.aggregator.global_blocks[layer].attn.register_forward_hook(profiler.hook(layer), with_kwargs=True) for layer in args.layers]
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            run_to_global_layer(model, images.unsqueeze(0), max(args.layers))
    finally:
        for handle in handles:
            handle.remove()
    for layer in args.layers:
        records = [r for r in profiler.rows if r["layer"] == layer]
        assert len(records) == 16 * 7
        with (profiler.out / f"layer_{layer:02d}_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)


def render(args):
    from scripts.visualize_vggt_value_medoids import label, overlay
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    root = args.input_dir / "vs_uniform_10pct"
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    n = int(np.prod(metadata["patch_grid"]))
    rows = []
    for layer in args.layers:
        with (root / f"layer_{layer:02d}_metrics.csv").open() as handle:
            for row in csv.DictReader(handle):
                converted = {k: (v if k == "method" else float(v)) for k, v in row.items()}
                converted["layer"], converted["head"] = int(converted["layer"]), int(converted["head"])
                rows.append(converted)
    comparisons, pages, links = [], [], []
    for layer in args.layers:
        thumbnails = []
        for head in range(16):
            head_rows = {r["method"]: r for r in rows if r["layer"] == layer and r["head"] == head}
            medoid, uniform = head_rows["medoids"], head_rows["uniform_matched_center"]
            phases = [r for name, r in head_rows.items() if name.startswith("uniform_matched")]
            oracle = min(phases, key=lambda r: r["attention_output_nmse"])
            comparisons.append({"layer": layer, "head": head,
                                "medoid_output_nmse": medoid["attention_output_nmse"],
                                "uniform_output_nmse": uniform["attention_output_nmse"],
                                "medoid_wins": medoid["attention_output_nmse"] < uniform["attention_output_nmse"],
                                "relative_nmse_reduction": 1-medoid["attention_output_nmse"]/uniform["attention_output_nmse"],
                                "best_uniform_phase_method": oracle["method"],
                                "best_uniform_phase_output_nmse": oracle["attention_output_nmse"],
                                "medoid_beats_best_uniform_phase": medoid["attention_output_nmse"] < oracle["attention_output_nmse"],
                                "medoid_value_mean_l2": medoid["value_mean_nearest_l2"],
                                "uniform_value_mean_l2": uniform["value_mean_nearest_l2"]})
            supports = np.load(root / f"layer_{layer:02d}_head_{head:02d}_supports.npz")
            width, panel_height = sum(im.width for im in originals), originals[0].height + 40
            image = Image.new("RGB", (width, 100+2*panel_height), "white")
            draw = ImageDraw.Draw(image)
            label(draw, (12, 4), f"第{layer+1}层（L{layer:02d}）H{head:02d}：medoid 对比均匀采样，各207点", 25)
            label(draw, (12, 43), "相同每帧点数；黄点=选中patch中心，青框=patch范围；误差参照dense输出", 19)
            for method_index, (method, method_name) in enumerate((("medoids", "k-medoids"), ("uniform_matched_center", "空间均匀采样"))):
                record = head_rows[method]
                y = 100+method_index*panel_height
                label(draw, (12, y), f"{method_name}：输出NMSE={record['attention_output_nmse']:.5f}；Value平均距离={record['value_mean_nearest_l2']:.4f}", 20)
                x = 0
                for frame, original in enumerate(originals):
                    selected = supports[method]
                    local = selected[selected // n == frame] % n
                    image.paste(overlay(original, local, metadata["patch_grid"]), (x, y+35))
                    x += original.width
            filename = f"layer_{layer:02d}_head_{head:02d}_comparison.png"
            image.save(root / filename)
            pages.append(image)
            thumbnails.append(image.resize((640, image.height//2), Image.Resampling.LANCZOS))
            links.append(f'<a href="{filename}"><img loading="lazy" src="{filename}" alt="L{layer:02d} H{head:02d}"></a>')
        tw, th = thumbnails[0].size
        sheet = Image.new("RGB", (4*tw, 4*th+65), "#eeeeee")
        label(ImageDraw.Draw(sheet), (15, 10), f"第{layer+1}层：Value medoid 与空间均匀采样，每头同预算207点", 29)
        for head, thumbnail in enumerate(thumbnails):
            sheet.paste(thumbnail, ((head%4)*tw, 65+(head//4)*th))
        sheet.save(root / f"overview_layer_{layer:02d}.jpg", quality=93)
    with (root / "head_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    summary = {"input_metadata": metadata,
               "protocol": "Local dense-QKV head outputs, all 2072 patch queries, 207 patch keys plus 10 special keys. Ordinary sparse softmax, no cluster weighting. FP32 evaluation with TF32 disabled; dense trunk BF16.",
               "primary_uniform": "per-frame medoid-count-matched rectangular strata, one patch per cell at fixed centre; no Q/K/V observation",
               "additional_baselines": "balanced per-frame count centre; four quarter-cell phases. Best of five matched phases is a post-hoc output-based oracle, not an executable cheap selector.",
               "per_layer": []}
    for layer in args.layers:
        layer_rows = [r for r in rows if r["layer"] == layer]
        head_rows = [r for r in comparisons if r["layer"] == layer]
        record = {"layer": layer,
                  "medoid_better_heads": [r["head"] for r in head_rows if r["medoid_wins"]],
                  "uniform_better_heads": [r["head"] for r in head_rows if not r["medoid_wins"]],
                  "medoid_beats_best_uniform_phase_heads": [r["head"] for r in head_rows if r["medoid_beats_best_uniform_phase"]]}
        record["uniform_best_of_five_phases_oracle_mean_nmse"] = float(np.mean([r["best_uniform_phase_output_nmse"] for r in head_rows]))
        for method in ("medoids", "uniform_matched_center", "uniform_balanced_center"):
            selected = [r for r in layer_rows if r["method"] == method]
            record[method] = {key: float(np.mean([r[key] for r in selected])) for key in ("attention_output_nmse", "attention_output_cosine", "dense_patch_attention_fraction_retained", "value_mean_nearest_l2")}
        summary["per_layer"].append(record)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
    figure, axes = plt.subplots(3, 1, figsize=(15, 12), constrained_layout=True)
    for axis, layer in zip(axes, args.layers):
        selected = [r for r in comparisons if r["layer"] == layer]
        x = np.arange(16)
        axis.bar(x-0.18, [r["medoid_output_nmse"] for r in selected], width=0.34, label="k-medoids", color="#228ba1")
        axis.bar(x+0.18, [r["uniform_output_nmse"] for r in selected], width=0.34, label="均匀采样（固定中心）", color="#ef9a34")
        axis.scatter(x, [r["best_uniform_phase_output_nmse"] for r in selected], marker="_", s=120,
                     color="black", label="5种均匀偏移中事后最佳值")
        axis.set_yscale("log")
        axis.set_xticks(x, [f"H{head:02d}" for head in range(16)])
        axis.set_ylabel("输出 NMSE（对数轴，越低越好）")
        axis.set_title(f"第{layer+1}个全局层（L{layer:02d}）：同预算207个patch keys + 全部特殊keys")
        axis.legend(fontsize=10)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(root / "attention_nmse_by_head.png", dpi=180)
    plt.close(figure)
    pages[0].save(root / "all_heads_comparison.pdf", save_all=True, append_images=pages[1:], resolution=120)
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>代表性Value与均匀采样比较</title>'
            '<style>body{max-width:1500px;margin:30px auto;font-family:system-ui;padding:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}img{width:100%}</style>'
            '<h1>Value k-medoids 与空间均匀采样</h1><p>Chess 第0、20帧；第12–14层（L11–L13）。'
            '每头207个点，两方法每帧点数完全一致。上排为medoids，下排为均匀采样。输出误差由相同dense QKV重算，特殊keys始终保留。</p>'
            '<p>均匀采样按矩形分区各取中心patch；k-medoids只优化V代表性，不利用Q或attention权重。</p>'
            '<p><a href="all_heads_comparison.pdf">48页完整PDF</a> · <a href="head_comparison.csv">逐头误差对比CSV</a></p><div class="grid">'
            + "".join(links) + '</div></html>')
    (root / "index.html").write_text(html)
    print(json.dumps(summary["per_layer"], ensure_ascii=False, indent=2), flush=True)


def positions(args):
    """Compare saved point locations only; do not rerun a model or clustering."""
    from scipy.optimize import linear_sum_assignment
    from scripts.analyze_value_medoid_positions import PositionComparison
    from scripts.visualize_vggt_value_medoids import label, patch_box

    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    source = args.input_dir / "vs_uniform_10pct"
    root = source / "positions"
    root.mkdir(exist_ok=True)
    height, width = metadata["patch_grid"]
    n = height * width
    comparison = PositionComparison(height, width)
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    rows, pages, gallery, phase_rows = [], [], [], []

    def region_mass(ids):
        frame, patch = ids // n, ids % n
        regions = frame*16 + ((patch//width)*4//height)*4 + (patch%width)*4//width
        return np.bincount(regions, minlength=32) / len(ids)

    for layer in args.layers:
        thumbnails = []
        for head in range(16):
            saved = np.load(source / f"layer_{layer:02d}_head_{head:02d}_supports.npz")
            a, b = saved["medoids"], saved["uniform_matched_center"]
            measured = comparison.compare(a, b)
            matched_distances, forward, backward = [], [], []
            for frame in (0, 1):
                aa, bb = a[a//n == frame] % n, b[b//n == frame] % n
                assert len(aa) == len(bb)
                d = comparison.distance[np.ix_(aa, bb)]
                forward.extend(d.min(1)); backward.extend(d.min(0))
                ii, jj = linear_sum_assignment(d)
                matched_distances.extend(d[ii, jj])
            row = {"layer": layer, "head": head, "k": len(a),
                   **{key: value for key, value in measured.items() if not key.startswith("random_")},
                   "medoids_to_uniform_within_1_patch": float((np.asarray(forward) <= 1).mean()),
                   "medoids_to_uniform_within_2_patches": float((np.asarray(forward) <= 2).mean()),
                   "uniform_to_medoids_within_1_patch": float((np.asarray(backward) <= 1).mean()),
                   "uniform_to_medoids_within_2_patches": float((np.asarray(backward) <= 2).mean()),
                   "optimal_one_to_one_distance_patches": float(np.mean(matched_distances)),
                   "region_4x4_total_variation": float(0.5*np.abs(region_mass(a)-region_mass(b)).sum())}
            rows.append(row)
            # Changing only a uniform grid's phase can change every selected
            # patch. Keep this positional control so exact overlap alone is
            # not mistaken for a difference in large-scale spatial coverage.
            for phase_id in range(1, 5):
                alternate = saved[f"uniform_matched_phase{phase_id}"]
                phase_metrics = comparison.compare(b, alternate)
                displacements = []
                for frame in (0, 1):
                    base = b[b//n == frame] % n
                    shifted = alternate[alternate//n == frame] % n
                    d = comparison.distance[np.ix_(base, shifted)]
                    ii, jj = linear_sum_assignment(d)
                    displacements.extend(d[ii, jj])
                phase_rows.append({"layer": layer, "head": head, "phase": phase_id,
                                   **{key: phase_metrics[key] for key in ("exact_overlap", "within_1_patch", "within_2_patches")},
                                   "optimal_one_to_one_distance_patches": float(np.mean(displacements))})
            common = np.intersect1d(a, b)
            only_a, only_b = np.setdiff1d(a, b), np.setdiff1d(b, a)
            image = Image.new("RGB", (sum(im.width for im in originals), max(im.height for im in originals)+145), "white")
            draw = ImageDraw.Draw(image)
            label(draw, (12, 5), f"第{layer+1}层（L{layer:02d}）H{head:02d}：k-medoids 与均匀采样的位置对照", 24)
            label(draw, (12, 43), "绿色=共同位置；橙色=仅medoids；青色=仅均匀采样（固定格内中心）", 20)
            label(draw, (12, 75), f"相同位置 {len(common)}/{len(a)}={row['exact_overlap']:.2%}；双向1patch内 {row['within_1_patch']:.2%}；2patch内 {row['within_2_patches']:.2%}", 20)
            offset = 0
            for frame, original in enumerate(originals):
                marks = Image.new("RGBA", original.size)
                marker = ImageDraw.Draw(marks)
                for selected, color, radius in ((only_a, (255, 155, 0, 255), 3),
                                                (only_b, (0, 230, 255, 255), 3),
                                                (common, (70, 255, 65, 255), 5)):
                    for patch in selected[selected//n == frame] % n:
                        patch_row, patch_col = divmod(int(patch), width)
                        box = patch_box(original, patch_row, patch_col, [height, width])
                        x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
                        marker.rectangle(box, outline=color, width=1)
                        marker.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline="black")
                marked = Image.alpha_composite(original.convert("RGBA"), marks).convert("RGB")
                image.paste(marked, (offset, 145))
                count = int((common//n == frame).sum())
                label(draw, (offset+12, 111), f"原图帧 {metadata['frame_indices'][frame]}：共同位置{count}个", 19)
                offset += original.width
            filename = f"layer_{layer:02d}_head_{head:02d}_positions.png"
            image.save(root / filename)
            pages.append(image)
            thumbnails.append(image.resize((640, image.height//2), Image.Resampling.LANCZOS))
            gallery.append(f'<a href="{filename}"><img loading="lazy" src="{filename}"></a>')
        tw, th = thumbnails[0].size
        overview = Image.new("RGB", (4*tw, 4*th+65), "#eeeeee")
        label(ImageDraw.Draw(overview), (15, 10), f"第{layer+1}层：绿=两方法共同点，橙=仅medoids，青=仅均匀采样", 29)
        for head, thumbnail in enumerate(thumbnails):
            overview.paste(thumbnail, ((head%4)*tw, 65+(head//4)*th))
        overview.save(root / f"overview_layer_{layer:02d}.jpg", quality=93)
    with (root / "per_head_positions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (root / "uniform_phase_reference.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phase_rows[0]))
        writer.writeheader(); writer.writerows(phase_rows)
    summary = {"input_metadata": metadata,
               "comparison": "same layer/head: saved k-medoids versus fixed-centre spatial uniform selection, identical per-frame counts; no value or attention-output comparisons here",
               "distance": "Euclidean patch coordinates, same image only; radius 1 excludes diagonal distance sqrt(2)",
               "neighborhood_hit": "symmetric nearest-neighbour coverage; multiple medoids may share one nearest uniform point",
               "one_to_one_distance": "minimum total Euclidean displacement under bijective medoid-to-uniform assignment, solved separately in each frame",
               "region_total_variation": "half L1 distance between normalized point-count histograms over two images times 4x4 rectangular image regions; unadjusted descriptive statistic",
               "per_layer": [], "uniform_phase_reference_per_layer": []}
    for layer in args.layers:
        selected = [r for r in rows if r["layer"] == layer]
        summary["per_layer"].append({"layer": layer, **{key: float(np.mean([r[key] for r in selected]))
                                                      for key in selected[0] if key not in ("layer", "head", "k")}})
        phase_selected = [r for r in phase_rows if r["layer"] == layer]
        summary["uniform_phase_reference_per_layer"].append({
            "layer": layer, **{key: float(np.mean([r[key] for r in phase_selected])) for key in
                               ("exact_overlap", "within_1_patch", "within_2_patches", "optimal_one_to_one_distance_patches")}})
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    pages[0].save(root / "all_heads_positions.pdf", save_all=True, append_images=pages[1:], resolution=120)
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>medoids与均匀采样的位置差异</title>'
                '<style>body{max-width:1500px;margin:30px auto;padding:12px;font-family:system-ui}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}img{width:100%}</style>'
                '<h1>k-medoids与均匀采样：位置差异</h1><p>第12–14层，每头207点；相同每帧点数。'
                '绿点为共同选中位置，橙点仅medoids选中，青点仅空间均匀采样选中。只比较同一帧内的空间坐标。</p>'
                '<p>1/2 patch邻域比例是双向最近邻命中；原图约17/34像素。较高邻域比例也受到均匀网格覆盖整图的影响。</p>'
                '<p><a href="all_heads_positions.pdf">48页原图叠加PDF</a> · <a href="per_head_positions.csv">逐头位置统计CSV</a></p>'
                '<div class="grid">'+"".join(gallery)+'</div></html>')
    (root / "index.html").write_text(document)
    print(json.dumps(summary["per_layer"], ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate", "render", "positions"))
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/value_medoids/chess_f000000_f000020_layers_12_14"))
    parser.add_argument("--layers", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    {"evaluate": evaluate, "render": render, "positions": positions}[args.stage](args)


if __name__ == "__main__":
    main()
