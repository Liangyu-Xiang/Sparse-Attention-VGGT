#!/usr/bin/env python3
"""Map every patch value to its nearest 1%-retention medoid on original RGB.

Assignments are the exact saved PAM value-space assignments, never spatial
Voronoi regions. Colour/cluster IDs correspond only within a head across its
two images. Each medoid gets its own original-image region highlight.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from scripts.visualize_vggt_value_medoids import label, patch_box


PALETTE = np.array([
    [31,119,180], [255,127,14], [44,160,44], [214,39,40], [148,103,189],
    [140,86,75], [227,119,194], [127,127,127], [188,189,34], [23,190,207],
    [174,199,232], [255,187,120], [152,223,138], [255,152,150], [197,176,213],
    [196,156,148], [247,182,210], [199,199,199], [219,219,141], [158,218,229],
    [250,210,25],
], dtype=np.uint8)


def check_assignment(values, medoids, assignments, saved_distances):
    if assignments.shape != (len(values),) or len(np.unique(medoids)) != len(medoids):
        raise ValueError("Invalid medoid or assignment dimensions")
    if assignments.min() < 0 or assignments.max() >= len(medoids):
        raise ValueError("Invalid cluster ID")
    # Independent direct differences check the cached norm/GEMM computation.
    direct = np.linalg.norm(values[:, None, :].astype(np.float64) - values[medoids][None, :, :], axis=2)
    selected = direct[np.arange(len(values)), assignments]
    tolerance = 2e-4 * max(1.0, float(direct.max()))
    if not np.all(selected <= direct.min(1) + tolerance):
        raise ValueError("An assignment does not belong to a nearest value medoid")
    np.testing.assert_allclose(saved_distances, selected, rtol=2e-4, atol=tolerance)
    return np.bincount(assignments, minlength=len(medoids))


def pixel_labels(labels, original):
    return np.asarray(Image.fromarray(labels.astype(np.uint8)).resize(original.size, Image.Resampling.NEAREST))


def draw_medoid(draw, original, patch, grid, number, color):
    row, col = divmod(int(patch), grid[1])
    box = patch_box(original, row, col, grid)
    x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
    angles = np.arange(10)*np.pi/5-np.pi/2
    radii = np.where(np.arange(10)%2 == 0, 10, 4)
    points = list(zip(x+radii*np.cos(angles), y+radii*np.sin(angles)))
    draw.polygon(points, fill=color, outline="white", width=2)
    tag_x = min(max(0, x+8), original.width-35)
    tag_y = min(max(0, y-21), original.height-22)
    draw.rounded_rectangle((tag_x, tag_y, tag_x+34, tag_y+21), radius=3, fill="white", outline="black")
    label(draw, (tag_x+2, tag_y), f"C{number:02d}", 14)


def pair_canvas(originals, title, subtitle, frames, legend=None):
    legend_height = 105 if legend is not None else 0
    image = Image.new("RGB", (sum(im.width for im in originals), 115+max(im.height for im in originals)+legend_height), "white")
    draw = ImageDraw.Draw(image)
    label(draw, (12, 5), title, 24)
    label(draw, (12, 42), subtitle, 18)
    offset = 0
    for original, frame, panel in zip(originals, frames, frames.panels):
        label(draw, (offset+12, 78), f"原图帧 {frame}", 20)
        image.paste(panel, (offset, 115))
        offset += original.width
    if legend is not None:
        top = 115+max(im.height for im in originals)
        cell_width = image.width//7
        for index, count in enumerate(legend):
            xx, yy = (index%7)*cell_width+8, top+(index//7)*34+6
            draw.rectangle((xx, yy+4, xx+19, yy+23), fill=tuple(PALETTE[index].tolist()), outline="black")
            label(draw, (xx+25, yy), f"C{index+1:02d}: {count}", 18)
    return image


class FramePanels(list):
    def __init__(self, frame_indices, panels):
        super().__init__(frame_indices)
        self.panels = panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/value_medoids/chess_f000000_f000020"))
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    height, width = metadata["patch_grid"]
    n = height*width
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    for original in originals:
        resized_h = round(original.height*(518/original.width)/14)*14
        if width*14 != 518 or height*14 != resized_h:
            raise ValueError("This region renderer requires uncropped width-518 two-frame inputs")
    out = args.input_dir / "value_clusters_1pct"
    out.mkdir(exist_ok=True)
    values_cache = np.load(args.input_dir / "values.npz")
    summaries, cluster_records, overview_pages, index_sections = [], [], [], []
    patch_fields = ["layer", "head", "frame", "patch_row", "patch_col", "cluster_id", "medoid_joint_patch_index", "distance_to_medoid", "is_medoid"]
    with (out / "all_patch_classes.csv").open("w", newline="") as patch_handle:
        patch_writer = csv.DictWriter(patch_handle, fieldnames=patch_fields)
        patch_writer.writeheader()
        for layer in args.layers:
            thumbnails, head_links = [], []
            for head in range(16):
                source = args.input_dir / "keep_1%" / f"layer_{layer:02d}" / f"head_{head:02d}.npz"
                saved = np.load(source)
                medoids, assignments = saved["medoids"], saved["assignments"]
                distances = saved["distances"]
                counts = check_assignment(values_cache[f"layer_{layer}"][head], medoids, assignments, distances)
                assert len(medoids) == 21 and len(assignments) == 2*n
                np.testing.assert_array_equal(counts, saved["cluster_sizes"])
                assert (counts > 0).all()
                assert np.array_equal(assignments[medoids], np.arange(len(medoids)))
                directory = out / f"layer_{layer:02d}" / f"head_{head:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                grid_labels = assignments.reshape(2, height, width)
                np.savez_compressed(directory / "patch_labels.npz", cluster_ids=grid_labels.astype(np.uint8)+1,
                                    medoids=medoids, palette_rgb=PALETTE)
                pix = [pixel_labels(labels, original) for labels, original in zip(grid_labels, originals)]
                colour_panels, representatives, pure_panels = [], [], []
                for frame, original in enumerate(originals):
                    coloured = Image.fromarray(PALETTE[pix[frame]])
                    pure_panels.append(coloured)
                    coloured = Image.blend(original, coloured, 0.55)
                    representatives.append(original.copy())
                    for cluster_index, medoid in enumerate(medoids):
                        if medoid//n != frame:
                            continue
                        color = tuple(PALETTE[cluster_index].tolist())
                        for panel in (coloured, representatives[frame]):
                            draw_medoid(ImageDraw.Draw(panel), original, medoid%n, [height, width], cluster_index+1, color)
                    colour_panels.append(coloured)
                title = f"第{layer+1}层（L{layer:02d}）H{head:02d}：保留1%，21个代表token与全部patch分类"
                subtitle = "同色=同一value簇；星号+C编号=该簇medoid；下方数字=簇内patch数；颜色仅在本头两帧间对应"
                combined = pair_canvas(originals, title, subtitle, FramePanels(metadata["frame_indices"], colour_panels), counts)
                combined.save(directory / "cluster_map.png")
                representative_map = pair_canvas(originals, title, "21个medoid对应C01-C21；同一编号在区域图和单簇高亮图中对应", FramePanels(metadata["frame_indices"], representatives), counts)
                representative_map.save(directory / "representatives.png")
                pure = pair_canvas(originals, title, "纯簇标签图：每个patch恰好分到一个最近value medoid；未做空间平滑", FramePanels(metadata["frame_indices"], pure_panels), counts)
                pure.save(directory / "labels_only.png")
                overview_pages.append(combined)
                thumbnails.append(combined.resize((640, combined.height//2), Image.Resampling.LANCZOS))
                medoid_lookup = set(medoids.tolist())
                for index, (assignment, distance) in enumerate(zip(assignments, distances)):
                    frame, patch = divmod(index, n)
                    patch_row, patch_col = divmod(patch, width)
                    patch_writer.writerow({"layer": layer, "head": head, "frame": metadata["frame_indices"][frame],
                                           "patch_row": patch_row, "patch_col": patch_col, "cluster_id": int(assignment)+1,
                                           "medoid_joint_patch_index": int(medoids[assignment]), "distance_to_medoid": float(distance),
                                           "is_medoid": index in medoid_lookup})
                cards = []
                for cluster_index, medoid in enumerate(medoids):
                    frame_medoid, local_medoid = divmod(int(medoid), n)
                    medoid_row, medoid_col = divmod(local_medoid, width)
                    colour = tuple(PALETTE[cluster_index].tolist())
                    frame_counts = [int((grid_labels[frame] == cluster_index).sum()) for frame in (0, 1)]
                    record = {"layer": layer, "head": head, "cluster_id": cluster_index+1,
                              "colour_hex": "#"+"".join(f"{c:02x}" for c in colour),
                              "patch_count": int(counts[cluster_index]), "fraction_of_all_patches": float(counts[cluster_index]/(2*n)),
                              "frame_0_patch_count": frame_counts[0], "frame_20_patch_count": frame_counts[1],
                              "medoid_frame": metadata["frame_indices"][frame_medoid], "medoid_patch_row": medoid_row, "medoid_patch_col": medoid_col,
                              "mean_value_distance": float(distances[assignments == cluster_index].mean())}
                    cluster_records.append(record)
                    panels = []
                    for frame, original in enumerate(originals):
                        mask = Image.fromarray(((pix[frame] == cluster_index)*255).astype(np.uint8))
                        panel = Image.composite(original, ImageEnhance.Brightness(original).enhance(0.16), mask)
                        if frame == frame_medoid:
                            draw_medoid(ImageDraw.Draw(panel), original, local_medoid, [height, width], cluster_index+1, colour)
                        panels.append(panel)
                    region = pair_canvas(originals,
                                         f"第{layer+1}层 H{head:02d}，C{cluster_index+1:02d}：这个代表token覆盖的区域",
                                         f"亮区为该簇的全部patch：共{counts[cluster_index]}个（{counts[cluster_index]/(2*n):.1%}）；帧0有{frame_counts[0]}个，帧20有{frame_counts[1]}个；星号是medoid",
                                         FramePanels(metadata["frame_indices"], panels))
                    region.save(directory / f"cluster_{cluster_index+1:02d}.jpg", quality=91)
                    crop_box = patch_box(originals[frame_medoid], medoid_row, medoid_col, [height, width])
                    crop = originals[frame_medoid].crop(tuple(round(x) for x in crop_box)).resize((84, 84), Image.Resampling.NEAREST)
                    crop.save(directory / f"medoid_{cluster_index+1:02d}.png")
                    cards.append(f'<section><h3 style="border-left:16px solid {record["colour_hex"]};padding-left:8px">C{cluster_index+1:02d} · {counts[cluster_index]} patches</h3>'
                                 f'<p><img class="crop" src="medoid_{cluster_index+1:02d}.png">代表点：帧{record["medoid_frame"]}，patch({medoid_row},{medoid_col})；'
                                 f'帧0覆盖{frame_counts[0]}个，帧20覆盖{frame_counts[1]}个。</p>'
                                 f'<a href="cluster_{cluster_index+1:02d}.jpg"><img loading="lazy" src="cluster_{cluster_index+1:02d}.jpg"></a></section>')
                doc = (f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>L{layer:02d} H{head:02d} Value簇区域</title>'
                       '<style>body{max-width:1500px;margin:25px auto;padding:12px;font-family:system-ui;line-height:1.6}img{width:100%}.crop{width:84px;float:left;margin-right:12px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}section{border:1px solid #ddd;padding:10px}a{color:#14609a}</style>'
                       f'<h1>第{layer+1}层 H{head:02d}：21个代表token各自表示哪些区域</h1>'
                       '<p>全部2072个patch按原始value欧氏距离归到最近medoid。颜色/编号仅在本头两帧内对应，不是物体类别；同簇可跨帧且不连续。'
                       '亮区是该代表点的全部成员，暗区是其他簇。点击图片放大。</p>'
                       '<p><a href="../../index.html">返回全部头</a> · <a href="representatives.png">仅看21个代表点</a> · <a href="labels_only.png">纯标签图</a> · <a href="patch_labels.npz">patch分类数组</a></p>'
                       '<img src="cluster_map.png"><h2>逐代表点查看覆盖区域</h2><div class="cards">'+"".join(cards)+'</div></html>')
                (directory / "index.html").write_text(doc)
                relative = directory.relative_to(out)
                head_links.append(f'<a href="{relative}/index.html"><img loading="lazy" src="{relative}/cluster_map.png" alt="H{head:02d}"><p>第{layer+1}层 H{head:02d}：查看21个代表点的覆盖区域</p></a>')
                record = {"layer": layer, "head": head, "n_patches": int(len(assignments)), "n_clusters": int(len(medoids)),
                          "cluster_size_min": int(counts.min()), "cluster_size_max": int(counts.max()),
                          "cross_frame_clusters": int(sum((grid_labels[0] == c).any() and (grid_labels[1] == c).any() for c in range(21)))}
                summaries.append(record)
                print(f"第{layer+1}层 H{head:02d}：2072个patch → 21簇；跨帧簇{record['cross_frame_clusters']}；簇大小{counts.min()}–{counts.max()}", flush=True)
            tw, th = thumbnails[0].size
            sheet = Image.new("RGB", (4*tw, 4*th+65), "#eeeeee")
            label(ImageDraw.Draw(sheet), (15, 10), f"第{layer+1}层：1% medoids对应的全部patch分类，颜色仅在每个头内有效", 28)
            for head, thumbnail in enumerate(thumbnails):
                sheet.paste(thumbnail, ((head%4)*tw, 65+(head//4)*th))
            sheet.save(out / f"overview_layer_{layer:02d}.jpg", quality=94)
            index_sections.append(f'<h2>第{layer+1}个全局层（L{layer:02d}）</h2><a href="overview_layer_{layer:02d}.jpg">16头总览</a><div class="grid">'+"".join(head_links)+'</div>')
    for filename, records in (("cluster_regions.csv", cluster_records), ("head_summary.csv", summaries)):
        with (out / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    overview_pages[0].save(out / "all_heads_cluster_maps.pdf", save_all=True, append_images=overview_pages[1:], resolution=120)
    (out / "summary.json").write_text(json.dumps({"input_metadata": metadata, "layers": args.layers,
        "retention": 0.01, "medoids_per_head": 21, "patches_per_head": 2*n,
        "assignment": "saved exact nearest raw-value Euclidean medoid membership, independently verified against direct FP64 value differences; ties follow saved PAM assignments",
        "display_cluster_ids": "C01–C21 correspond to zero-based saved assignment IDs +1; labels and colours have no meaning across heads/layers",
        "spatial_interpretation": "patch masks are projected from value-space memberships, not spatial Voronoi regions or semantic object classes",
        "heads": summaries}, ensure_ascii=False, indent=2)+"\n")
    layer_title = "、".join(str(layer+1) for layer in args.layers)
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>1%代表token与全部patch分类</title>'
                '<style>body{max-width:1500px;margin:25px auto;padding:12px;font-family:system-ui;line-height:1.7}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%}a{color:#14609a}</style>'
                f'<h1>第{layer_title}层：1%代表token与全部patch的Value分类</h1><p>Chess第0、20帧。每头从2072个patch选择21个medoids，全部patch按原始V的欧氏距离分到最近medoid。'
                '同色代表同一个簇，星号和C编号标记代表token。颜色仅在单个头的两帧之间对应；这不是预设物体类别的语义分割。</p>'
                '<p>点击任意头，可查看仅代表点、完整分类、纯标签图，以及21个代表点各自覆盖的区域。亮区表示该代表点的成员。</p>'
                '<p><a href="all_heads_cluster_maps.pdf">48页全头分类PDF</a> · <a href="cluster_regions.csv">1008个簇的统计</a> · <a href="all_patch_classes.csv">全部patch的分类CSV</a></p>'
                +"".join(index_sections)+'</html>')
    (out / "index.html").write_text(document)


if __name__ == "__main__":
    main()
