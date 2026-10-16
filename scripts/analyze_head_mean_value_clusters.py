#!/usr/bin/env python3
"""Average raw V over heads, then independently cluster each layer's patches.

This is coordinate-wise averaging of 16 raw 64D head values, NOT averaging
cluster labels, attention probabilities, or projected multi-head outputs.
Head feature bases are not aligned; coordinate averaging is basis dependent.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

from scripts.visualize_vggt_value_medoids import pairwise_distances, pam, label
from scripts.visualize_value_cluster_regions import (
    PALETTE, FramePanels, check_assignment, draw_medoid, pair_canvas, pixel_labels,
)
from scripts.analyze_value_cluster_evolution import (
    all_contingencies, cluster_change, partition_scores,
    null_best_scores, save_alluvial, save_aligned_maps, save_split_example,
)


def average_heads(values):
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[0] < 1:
        raise ValueError("Expected raw V with shape [heads, patches, dimensions]")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite raw V")
    return values.mean(axis=0, dtype=np.float32)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_layer(metadata, layer, result, directory):
    medoids, assignments, distances = (result[key] for key in ("medoids", "assignments", "distances"))
    grid = metadata["patch_grid"]
    n = int(np.prod(grid))
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    counts = np.bincount(assignments, minlength=len(medoids))
    grid_labels = assignments.reshape(2, *grid)
    pix = [pixel_labels(lab, original) for lab, original in zip(grid_labels, originals)]
    colour_panels, representatives = [], []
    for frame, original in enumerate(originals):
        coloured = Image.blend(original, Image.fromarray(PALETTE[pix[frame]]), .55)
        representative = original.copy()
        for cluster, medoid in enumerate(medoids):
            if medoid//n == frame:
                for panel in (coloured, representative):
                    draw_medoid(ImageDraw.Draw(panel), original, medoid%n, grid,
                                cluster+1, tuple(PALETTE[cluster].tolist()))
        colour_panels.append(coloured)
        representatives.append(representative)
    title = f"第{layer+1}层（L{layer:02d}）：16头原始V平均后，1% medoids与全部patch分类"
    subtitle = "21个真实patch代表2072个平均V；星号+C编号=medoid；颜色/编号不直接跨层对应"
    pair_canvas(originals,title,subtitle,FramePanels(metadata["frame_indices"],colour_panels),counts).save(directory/"cluster_map.png")
    pair_canvas(originals,title,"仅标出21个代表点；统计数字为每簇patch数",FramePanels(metadata["frame_indices"],representatives),counts).save(directory/"representatives.png")
    pure = [Image.fromarray(PALETTE[p]) for p in pix]
    pair_canvas(originals,title,"每个patch归到最近平均V medoid；未做空间平滑",FramePanels(metadata["frame_indices"],pure),counts).save(directory/"labels_only.png")
    cards, records = [], []
    for cluster, medoid in enumerate(medoids):
        selected_counts = [int((grid_labels[f]==cluster).sum()) for f in (0,1)]
        medoid_frame, medoid_patch = divmod(int(medoid), n)
        row, col = divmod(medoid_patch, grid[1])
        panels = []
        for frame, original in enumerate(originals):
            mask = Image.fromarray(((pix[frame]==cluster)*255).astype(np.uint8))
            panel = Image.composite(original,ImageEnhance.Brightness(original).enhance(.16),mask)
            if frame==medoid_frame:
                draw_medoid(ImageDraw.Draw(panel),original,medoid_patch,grid,cluster+1,tuple(PALETTE[cluster].tolist()))
            panels.append(panel)
        pair_canvas(originals,f"第{layer+1}层头平均V，C{cluster+1:02d}的成员区域",
                    f"共{counts[cluster]}个patch；帧0有{selected_counts[0]}个，帧20有{selected_counts[1]}个；亮区=全部成员，星号=medoid",
                    FramePanels(metadata["frame_indices"],panels)).save(directory/f"cluster_{cluster+1:02d}.jpg",quality=92)
        cards.append(f'<section><h3>C{cluster+1:02d}：{counts[cluster]}个patch</h3>'
                     f'<p>代表点：帧{metadata["frame_indices"][medoid_frame]}，patch({row},{col})</p>'
                     f'<img loading="lazy" src="cluster_{cluster+1:02d}.jpg"></section>')
        records.append({"layer":layer,"cluster_id":cluster+1,"patch_count":int(counts[cluster]),
                        "frame_0_patch_count":selected_counts[0],"frame_20_patch_count":selected_counts[1],
                        "medoid_frame":metadata["frame_indices"][medoid_frame],"medoid_patch_row":row,"medoid_patch_col":col,
                        "mean_value_distance":float(distances[assignments==cluster].mean())})
    (directory/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f'<title>第{layer+1}层：头平均V聚类</title>'+STYLE+
        f'<h1>第{layer+1}层：先平均16头V，再选择21个medoids</h1>'
        '<p>所有patch按平均V的欧氏距离分类；不是平均原有簇编号。亮区是该medoid代表的全部patch。</p>'
        '<p><a href="../index.html">返回逐层结果</a> · <a href="representatives.png">仅看代表点</a> · '
        '<a href="labels_only.png">纯分类图</a></p><img src="cluster_map.png"><div class="grid">'+"".join(cards)+'</div></html>')
    return records


def render_layerwise(metadata, layers, results, mappings, destination):
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    h = originals[0].height
    grid = metadata["patch_grid"]
    n = int(np.prod(grid))
    canvas = Image.new("RGB", (sum(im.width for im in originals), 110+len(layers)*(h+50)), "white")
    draw = ImageDraw.Draw(canvas)
    label(draw,(12,5),"前三层：每层先平均16头原始V，再做1% k-medoids",26)
    label(draw,(12,42),"左=帧0，右=帧20；颜色逐层按最大交集对齐，便于观察；同色不是簇不变的保证",19)
    colour_lookup = np.arange(len(results[0]["medoids"]))
    for pos,(layer,result) in enumerate(zip(layers,results)):
        if pos:
            next_lookup = np.empty_like(colour_lookup)
            next_lookup[mappings[pos-1]] = colour_lookup
            colour_lookup = next_lookup
        y = 90+pos*(h+50)
        label(draw,(12,y),f"第{layer+1}层（L{layer:02d}）；星号+C编号=本层代表token；本层所有2072个patch均已分类",22)
        labels = result["assignments"].reshape(2,*grid)
        offset = 0
        for frame,original in enumerate(originals):
            panel = Image.blend(original,Image.fromarray(PALETTE[colour_lookup[pixel_labels(labels[frame],original)]]),.55)
            for cluster,medoid in enumerate(result["medoids"]):
                if medoid//n==frame:
                    draw_medoid(ImageDraw.Draw(panel),original,medoid%n,grid,cluster+1,
                                tuple(PALETTE[colour_lookup[cluster]].tolist()))
            canvas.paste(panel,(offset,y+35))
            offset += original.width
    canvas.save(destination)


STYLE = '<style>body{max-width:1450px;margin:25px auto;padding:12px;font-family:system-ui;line-height:1.7}img{max-width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}a{color:#14609a}</style>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("outputs/value_medoids/chess_f000000_f000020"))
    parser.add_argument("--layers",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--max-swaps",type=int,default=1000)
    args = parser.parse_args()
    if len(args.layers)<2 or args.layers!=sorted(set(args.layers)):
        parser.error("Provide at least two unique layers in increasing order")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.input_dir/"head_mean_1pct"
    root.mkdir(exist_ok=True)
    metadata = json.loads((args.input_dir/"metadata.json").read_text())
    cache = np.load(args.input_dir/"values.npz")
    results, layer_rows, cluster_rows, patch_rows, mean_values = [], [], [], [], {}
    n = metadata["values_per_head"]
    k = round(.01*n)
    if k!=21:
        raise ValueError("This first visualization uses the existing two-frame/21-medoid setup")
    for layer in args.layers:
        raw = cache[f"layer_{layer}"]
        if raw.shape[0]!=16 or raw.shape[1]!=n:
            raise ValueError("Expected all 16 heads on the original patch domain")
        values = average_heads(raw)
        mean_values[f"layer_{layer}"] = values
        distance = pairwise_distances(torch.from_numpy(values).to(args.device))
        start = time.perf_counter()
        with torch.inference_mode():
            starts = [pam(distance,k,method,20261015+100*layer,args.max_swaps) for method in ("build","sample")]
        result = min(starts,key=lambda item:item["objective"])
        check_assignment(values,result["medoids"],result["assignments"],result["distances"])
        if not result["converged"]:
            raise RuntimeError(f"Layer {layer}: selected PAM start did not reach single-swap local optimum")
        results.append(result)
        directory = root/f"layer_{layer:02d}"
        directory.mkdir(exist_ok=True)
        counts = np.bincount(result["assignments"],minlength=k)
        np.savez_compressed(directory/"clusters.npz",**{key:result[key] for key in ("medoids","assignments","distances")},cluster_sizes=counts)
        stats = {key:value for key,value in result.items() if key not in ("medoids","assignments","distances")}
        stats["starts"] = [{key:value for key,value in run.items() if key not in ("medoids","assignments","distances")} for run in starts]
        row = {"layer":layer,"heads_averaged":len(raw),"patch_count":n,"medoid_count":k,"actual_retention":k/n,
               "objective":result["objective"],"mean_distance":float(result["distances"].mean()),
               "cluster_size_min":int(counts.min()),"cluster_size_max":int(counts.max()),
               "medoids_frame_0":int((result["medoids"]<(n//2)).sum()),"medoids_frame_20":int((result["medoids"]>=(n//2)).sum()),
               "converged":result["converged"],"swaps":result["swaps"],"elapsed_seconds":time.perf_counter()-start}
        layer_rows.append(row)
        (directory/"clustering.json").write_text(json.dumps({**row,**stats},indent=2)+"\n")
        cluster_rows.extend(render_layer(metadata,layer,result,directory))
        medoid_lookup = set(result["medoids"].tolist())
        for index,cluster in enumerate(result["assignments"]):
            frame,patch = divmod(index,n//2)
            r,c = divmod(patch,metadata["patch_grid"][1])
            patch_rows.append({"layer":layer,"frame":metadata["frame_indices"][frame],"patch_row":r,"patch_col":c,
                               "cluster_id":int(cluster)+1,"medoid_joint_patch_index":int(result["medoids"][cluster]),
                               "distance_to_medoid":float(result["distances"][index]),"is_medoid":index in medoid_lookup})
        print(json.dumps(row,ensure_ascii=False),flush=True)
    np.savez_compressed(root/"mean_values.npz",**mean_values)
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    plt.rcParams["font.family"]=["DejaVu Sans","Droid Sans Fallback"]
    rng = np.random.default_rng(20261015)
    relations, mappings, flow_rows = [], [], []
    for pos in range(len(results)-1):
        source,target = args.layers[pos:pos+2]
        a,b = (results[j]["assignments"] for j in (pos,pos+1))
        counts = all_contingencies(a[None],b[None],k)[0,0]
        ari,nmi = partition_scores(counts)
        change = cluster_change(counts)
        mapping = np.asarray(change["source_to_target_cluster_mapping"])
        mappings.append(mapping)
        null,_ = null_best_scores(a[None],b[None],k,500,rng)
        relation = {"source_layer":source,"target_layer":target,"ari":float(ari),"nmi":float(nmi),
                    **{key:value for key,value in change.items() if key!="source_to_target_cluster_mapping"},
                    "random_ari_mean":float(null.mean()),"random_ari_q025":float(np.quantile(null,.025)),"random_ari_q975":float(np.quantile(null,.975))}
        relations.append(relation)
        directory = root/f"L{source:02d}_to_L{target:02d}"
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/"transitions.npz",counts=counts,source_labels=a,target_labels=b,source_to_target_cluster_mapping=mapping)
        title = f"第{source+1}层 到 第{target+1}层：16头平均V的簇变化"
        save_alluvial(counts,mapping,title,directory/"cluster_flows.png")
        save_aligned_maps(metadata,a,b,mapping,title,float(ari),directory/"aligned_regions.png")
        example = save_split_example(metadata,a,b,counts,title,directory/"split_example.png")
        (directory/"example.json").write_text(json.dumps(example,ensure_ascii=False,indent=2)+"\n")
        for i,j in zip(*np.nonzero(counts)):
            flow_rows.append({"source_layer":source,"target_layer":target,"source_cluster":int(i)+1,"target_cluster":int(j)+1,
                              "patch_count":int(counts[i,j]),"fraction_of_source":float(counts[i,j]/counts[i].sum()),
                              "fraction_of_target":float(counts[i,j]/counts[:,j].sum())})
        print(json.dumps(relation,ensure_ascii=False),flush=True)
    render_layerwise(metadata,args.layers,results,mappings,root/"layerwise_aligned_regions.png")
    for filename,rows in (("layer_summary.csv",layer_rows),("cluster_regions.csv",cluster_rows),("all_patch_classes.csv",patch_rows),
                          ("adjacent_layers.csv",relations),("cluster_flows.csv",flow_rows)):
        write_csv(root/filename,rows)
    summary = {"input_metadata":metadata,"layers":args.layers,"aggregation":"arithmetic mean over head axis of raw V, float32; no L2 normalization or feature-basis alignment",
               "operation_order":"mean raw V -> independent layerwise k-medoids -> nearest-medoid assignments; never mean cluster IDs",
               "heads_averaged":16,"retention":.01,"actual_retention":k/n,"clusters_per_layer":k,
               "caveat":"Heads have independently learned feature bases; coordinate-wise mean is basis dependent and is not the mean attention probability, projected output, or consensus partition.",
               "optimizer":"two PAM starts with exhaustive best single swaps; single-swap local optimum, not certified global optimum",
               "null":"500 shared within-frame position permutations, preserving cluster sizes per frame",
               "layers_summary":layer_rows,"adjacent_layers":relations}
    (root/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    table = ''.join(f'<tr><td>第{r["source_layer"]+1}→{r["target_layer"]+1}层</td><td>{r["ari"]:.4f}</td><td>{r["nmi"]:.4f}</td><td>{r["aligned_same_fraction"]:.2%}</td></tr>' for r in relations)
    layer_links = ''.join(f'<p><a href="layer_{layer:02d}/index.html">第{layer+1}层：查看21个代表token各自覆盖的区域</a></p><img src="layer_{layer:02d}/cluster_map.png">' for layer in args.layers)
    relation_links = ''.join(f'<h3>第{s+1}→{t+1}层</h3><a href="L{s:02d}_to_L{t:02d}/cluster_flows.png">全部簇流量图</a><img src="L{s:02d}_to_L{t:02d}/split_example.png">' for s,t in zip(args.layers[:-1],args.layers[1:]))
    (root/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>前三层：先平均16头V再聚类</title>'+STYLE+
        '<h1>前三层：先平均16头原始Value，再进行1% k-medoids</h1>'
        '<p>相同Chess第0、20帧。每层把同一个patch的16个64维V逐坐标取算术平均，再独立选21个medoids，分类全部2072个patch。不是平均原有簇编号或头间统计。</p>'
        '<p>注意：不同头的特征基底未对齐，原始V直接平均具有基底依赖性；这不是平均注意力权重、拼接后的输出或多头共识聚类。前三层独立聚类，簇编号不直接对应；总览颜色仅按最大交集逐层对齐。</p>'
        '<p><a href="layer_summary.csv">逐层统计</a> · <a href="all_patch_classes.csv">全部patch分类</a> · <a href="adjacent_layers.csv">跨层统计</a> · <a href="cluster_flows.csv">全部簇流向</a></p>'
        '<table><tr><th>相邻层</th><th>ARI</th><th>NMI</th><th>最大交集标签对齐后同归属比例</th></tr>'+table+'</table>'
        '<img src="layerwise_aligned_regions.png"><h2>各层原始簇编号与逐代表点区域</h2>'+layer_links+
        '<h2>相邻层的单簇变化图解</h2><p>每组展示最大的明显分流源簇，上排为原区域，下排为同位置patch在下一层的去向；完整流量另见桑基图。</p>'+relation_links+'</html>')


if __name__ == "__main__":
    main()
