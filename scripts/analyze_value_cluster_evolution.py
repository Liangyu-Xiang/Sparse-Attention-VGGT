#!/usr/bin/env python3
"""Compare whole-patch value partitions across adjacent VGGT layers.

Heads are matched using label-invariant ARI across all 16x16 head pairs.
Cluster labels are never assumed aligned. Exact contingency counts track the
SAME image/patch token across depth; alluvial flow means cluster membership,
not attention edges or motion. Cluster colours in paired maps are aligned by
maximum overlap solely for readable visualisation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from scripts.visualize_vggt_value_medoids import label
from scripts.visualize_value_cluster_regions import PALETTE, pixel_labels


def all_contingencies(a, b, k):
    if a.shape[1] != b.shape[1]:
        raise ValueError("Partitions must describe identical patch-token domains")
    keys = a[:, None, :]*k + b[None, :, :]
    offsets = np.arange(len(a)*len(b)).reshape(len(a), len(b), 1)*k*k
    return np.bincount((keys+offsets).ravel(), minlength=len(a)*len(b)*k*k).reshape(len(a), len(b), k, k)


def partition_scores(counts):
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.sum(axis=(-2, -1))
    row, column = counts.sum(-1), counts.sum(-2)
    pairs = n*(n-1)/2
    same_both = (counts*(counts-1)/2).sum(axis=(-2, -1))
    same_a = (row*(row-1)/2).sum(-1)
    same_b = (column*(column-1)/2).sum(-1)
    expected = same_a*same_b / np.maximum(pairs, 1)
    denominator = (same_a+same_b)/2-expected
    ari = np.divide(same_both-expected, denominator, out=np.ones_like(n), where=np.abs(denominator)>1e-12)
    probabilities = counts/np.maximum(n[..., None, None], 1)
    independent = row[..., :, None]*column[..., None, :]/np.maximum(n[..., None, None]**2, 1)
    mi = (probabilities*np.log(np.maximum(probabilities, 1e-300)/np.maximum(independent, 1e-300))).sum(axis=(-2, -1))
    pa, pb = row/n[..., None], column/n[..., None]
    ha = -(pa*np.log(np.maximum(pa, 1e-300))).sum(-1)
    hb = -(pb*np.log(np.maximum(pb, 1e-300))).sum(-1)
    nmi = np.divide(2*mi, ha+hb, out=np.ones_like(n), where=ha+hb>1e-12)
    return ari, nmi


def cluster_change(counts, threshold=0.20):
    row, column = counts.sum(1), counts.sum(0)
    forward = counts/row[:, None]
    backward = counts/column[None, :]
    ii, jj = linear_sum_assignment(-counts)
    return {
        "aligned_same_fraction": float(counts[ii, jj].sum()/counts.sum()),
        "source_dominant_child_fraction": float(counts.max(1).sum()/counts.sum()),
        "target_dominant_parent_fraction": float(counts.max(0).sum()/counts.sum()),
        "split_source_clusters": int(((forward >= threshold).sum(1) >= 2).sum()),
        "merged_target_clusters": int(((backward >= threshold).sum(0) >= 2).sum()),
        "source_to_target_cluster_mapping": jj.tolist(),
        "substantial_branch_threshold": threshold,
    }


def null_best_scores(source, target, k, repetitions, rng):
    n_per_frame = source.shape[1]//2
    row_best, bijective = [], []
    for _ in range(repetitions):
        permutation = np.concatenate([rng.permutation(n_per_frame), n_per_frame+rng.permutation(n_per_frame)])
        ari, _ = partition_scores(all_contingencies(source, target[:, permutation], k))
        ii, jj = linear_sum_assignment(-ari)
        row_best.append(float(ari.max(1).mean()))
        bijective.append(float(ari[ii, jj].mean()))
    return np.asarray(row_best), np.asarray(bijective)


def save_alluvial(counts, target_order, title, destination):
    k = len(counts)
    n = counts.sum()
    row_sizes, col_sizes = counts.sum(1)/n, counts.sum(0)/n
    gap = 0.006
    row_bottom = np.cumsum(np.r_[0, row_sizes[:-1]])+gap*np.arange(k)
    ordered_sizes = col_sizes[target_order]
    col_bottom_ordered = np.cumsum(np.r_[0, ordered_sizes[:-1]])+gap*np.arange(k)
    col_bottom = np.zeros(k)
    col_bottom[target_order] = col_bottom_ordered
    row_used, col_used = np.zeros(k), np.zeros(k)
    figure, axis = plt.subplots(figsize=(11, 13), constrained_layout=True)
    for i in range(k):
        for j in target_order:
            amount = counts[i, j]/n
            if not amount:
                continue
            y0, y1 = row_bottom[i]+row_used[i], col_bottom[j]+col_used[j]
            vertices = [(0.15,y0), (0.40,y0), (0.60,y1), (0.85,y1),
                        (0.85,y1+amount), (0.60,y1+amount), (0.40,y0+amount), (0.15,y0+amount), (0.15,y0)]
            codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                     MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
            axis.add_patch(PathPatch(MplPath(vertices, codes), facecolor=PALETTE[i]/255, alpha=0.45, edgecolor="none"))
            row_used[i] += amount
            col_used[j] += amount
    for i in range(k):
        axis.add_patch(Rectangle((0.12,row_bottom[i]),0.03,row_sizes[i],facecolor=PALETTE[i]/255))
        axis.text(0.105,row_bottom[i]+row_sizes[i]/2,f"C{i+1:02d} ({counts[i].sum()})",ha="right",va="center",fontsize=9)
        j = int(target_order[i])
        axis.add_patch(Rectangle((0.85,col_bottom[j]),0.03,col_sizes[j],facecolor=PALETTE[i]/255))
        axis.text(0.895,col_bottom[j]+col_sizes[j]/2,f"C{j+1:02d} ({counts[:,j].sum()})",ha="left",va="center",fontsize=9)
    axis.set_xlim(-0.05,1.08); axis.set_ylim(1+gap*k,-0.025)
    axis.axis("off")
    axis.set_title(title+"\n全部2072个同位置patch的簇归属流向；带宽=patch数",fontsize=13)
    figure.savefig(destination,dpi=150)
    plt.close(figure)


def save_aligned_maps(metadata, a, b, mapping, title, ari, destination):
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    height, width = metadata["patch_grid"]
    inverse = np.empty_like(mapping)
    inverse[mapping] = np.arange(len(mapping))
    aligned_b = inverse[b]
    canvas = Image.new("RGB",(sum(im.width for im in originals), 115+2*(originals[0].height+35)),"white")
    draw = ImageDraw.Draw(canvas)
    label(draw,(12,5),title,24)
    label(draw,(12,43),f"ARI={ari:.3f}；目标层颜色按最大交集对齐；同色便于对照，簇关系以流量图为准",19)
    for r, (labels, text) in enumerate(((a,"前一层"),(aligned_b,"后一层（颜色对齐后）"))):
        y = 90+r*(originals[0].height+35)
        label(draw,(12,y),text+"；左=帧0，右=帧20",20)
        offset=0
        for frame, original in enumerate(originals):
            pix = pixel_labels(labels.reshape(2,height,width)[frame],original)
            colour = Image.fromarray(PALETTE[pix])
            canvas.paste(Image.blend(original,colour,0.55),(offset,y+30))
            offset += original.width
    canvas.save(destination)


def save_split_example(metadata, a, b, counts, title, destination):
    """Show the largest substantially split source cluster on the SAME patches."""
    sizes = counts.sum(1)
    branches = ((counts / sizes[:, None]) >= .20).sum(1)
    candidates = np.flatnonzero(branches >= 2)
    source = int(candidates[np.argmax(sizes[candidates])]) if len(candidates) else int(sizes.argmax())
    children = np.flatnonzero(counts[source])
    children = children[np.argsort(counts[source, children])[::-1]]
    originals = [Image.open(path).convert("RGB") for path in metadata["image_paths"]]
    height, width = metadata["patch_grid"]
    legend_y = 120+2*(originals[0].height+35)
    canvas = Image.new("RGB", (sum(im.width for im in originals), legend_y+32*((len(children)+4)//5)), "white")
    draw = ImageDraw.Draw(canvas)
    label(draw, (12,5), title.replace("→", "到"), 24)
    label(draw, (12,42), f"追踪源簇C{source+1:02d}的{sizes[source]}个patch；下排只给这些位置按目标簇着色，其余保持暗色", 19)
    a_grid, b_grid = a.reshape(2,height,width), b.reshape(2,height,width)
    for row in range(2):
        y = 88+row*(originals[0].height+35)
        label(draw, (12,y), (f"源簇C{source+1:02d}的原图区域" if row==0 else "同一批patch在后一层的去向")+"；左=帧0，右=帧20",20)
        offset = 0
        for frame, original in enumerate(originals):
            selected = pixel_labels(a_grid[frame], original) == source
            foreground = original if row==0 else Image.blend(original,Image.fromarray(PALETTE[pixel_labels(b_grid[frame],original)]),.65)
            background = Image.fromarray((np.asarray(original)*.16).astype(np.uint8))
            panel = Image.composite(foreground, background, Image.fromarray(selected.astype(np.uint8)*255))
            canvas.paste(panel,(offset,y+30))
            offset += original.width
    for index,j in enumerate(children):
        xx,yy = 8+(index%5)*(canvas.width//5),legend_y+(index//5)*32
        draw.rectangle((xx,yy+3,xx+17,yy+21),fill=tuple(PALETTE[j].tolist()),outline="black")
        label(draw,(xx+24,yy),f"C{j+1:02d}: {counts[source,j]} ({counts[source,j]/sizes[source]:.1%})",17)
    canvas.save(destination)
    return {"source_cluster":source+1,"source_patch_count":int(sizes[source]),
            "destinations":[{"cluster":int(j)+1,"count":int(counts[source,j]),
                             "fraction":float(counts[source,j]/sizes[source])} for j in children]}


def render_examples(root, metadata, selected_rows):
    """Two illustrative strongest head pairs, explicitly not population averages."""
    examples = []
    for layer in (0,1):
        best = max((row for row in selected_rows if int(row["source_layer"])==layer),key=lambda row:float(row["ari"]))
        source, target = int(best["source_head"]), int(best["target_head"])
        pair_name = f"L{layer:02d}_H{source:02d}_to_L{layer+1:02d}_H{target:02d}"
        saved = np.load(root/pair_name/"transitions.npz")
        title = f"第{layer+1}层 H{source:02d} 到 第{layer+2}层 H{target:02d}：单簇区域的跨层变化"
        detail = save_split_example(metadata,saved["source_labels"],saved["target_labels"],saved["counts"],title,root/f"example_L{layer}_L{layer+1}_split.png")
        examples.append({"pair":pair_name,"selection":"highest ARI head pair, then largest source cluster with at least two 20% branches",**detail})
    (root/"illustrated_examples.json").write_text(json.dumps(examples,ensure_ascii=False,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("outputs/value_medoids/chess_f000000_f000020"))
    parser.add_argument("--null-repetitions",type=int,default=500)
    args=parser.parse_args()
    root=args.input_dir/"cross_layer_cluster_evolution_1pct"
    root.mkdir(exist_ok=True)
    metadata=json.loads((args.input_dir/"metadata.json").read_text())
    sets=[[np.load(args.input_dir/f"keep_1%/layer_{l:02d}/head_{h:02d}.npz")["assignments"] for h in range(16)] for l in range(3)]
    labels=[np.stack(layer).astype(np.int64) for layer in sets]
    k=21
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    plt.rcParams["font.family"]=["DejaVu Sans","Droid Sans Fallback"]
    rng=np.random.default_rng(20261015)
    matrices,selected_rows,flow_rows,summaries,sections={},[],[],[],[]
    pair_rows=[]
    figure,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    for source_layer in (0,1):
        target_layer=source_layer+1
        counts=all_contingencies(labels[source_layer],labels[target_layer],k)
        ari,nmi=partition_scores(counts)
        matrices[f"L{source_layer}_L{target_layer}_ari"]=ari
        matrices[f"L{source_layer}_L{target_layer}_nmi"]=nmi
        best=ari.argmax(1)
        hi,hj=linear_sum_assignment(-ari)
        null,null_bijective=null_best_scores(labels[source_layer],labels[target_layer],k,args.null_repetitions,rng)
        for a in range(16):
            for b in range(16):
                pair_rows.append({"source_layer":source_layer,"target_layer":target_layer,"source_head":a,"target_head":b,
                                  "ari":float(ari[a,b]),"nmi":float(nmi[a,b])})
        axis=axes[source_layer]
        artist=axis.imshow(ari,vmin=0,vmax=0.55,cmap="viridis")
        axis.scatter(best,np.arange(16),marker="s",facecolors="none",edgecolors="red",s=90)
        axis.set_xticks(range(16));axis.set_yticks(range(16))
        axis.set_xlabel(f"第{target_layer+1}层目标头");axis.set_ylabel(f"第{source_layer+1}层源头")
        axis.set_title(f"第{source_layer+1}→{target_layer+1}层：完整簇划分的ARI，红框为每行最佳")
        figure.colorbar(artist,ax=axis,shrink=0.8,label="ARI（1=划分相同，0≈随机）")
        cards=[]
        for source_head,target_head in enumerate(best.tolist()):
            c=counts[source_head,target_head]
            change=cluster_change(c)
            row={"source_layer":source_layer,"target_layer":target_layer,"source_head":source_head,"target_head":target_head,
                 "ari":float(ari[source_head,target_head]),"nmi":float(nmi[source_head,target_head]),
                 **{key:value for key,value in change.items() if key!="source_to_target_cluster_mapping"}}
            selected_rows.append(row)
            name=f"L{source_layer:02d}_H{source_head:02d}_to_L{target_layer:02d}_H{target_head:02d}"
            directory=root/name
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/"transitions.npz",counts=c,source_labels=labels[source_layer][source_head],
                                target_labels=labels[target_layer][target_head],source_to_target_cluster_mapping=change["source_to_target_cluster_mapping"])
            title=f"第{source_layer+1}层 H{source_head:02d} → 第{target_layer+1}层 H{target_head:02d}"
            mapping=np.asarray(change["source_to_target_cluster_mapping"])
            save_alluvial(c,mapping,title,directory/"cluster_flows.png")
            save_aligned_maps(metadata,labels[source_layer][source_head],labels[target_layer][target_head],mapping,title.replace("→", "到"),row["ari"],directory/"aligned_regions.png")
            transition=c[:,mapping]/c.sum(1,keepdims=True)
            fig,ax=plt.subplots(figsize=(10,9),constrained_layout=True)
            im=ax.imshow(100*transition,vmin=0,vmax=100,cmap="magma")
            ax.set_xticks(range(k),[f"C{j+1:02d}" for j in mapping],rotation=90)
            ax.set_yticks(range(k),[f"C{i+1:02d}" for i in range(k)])
            ax.set_xlabel("目标层簇（按最大交集排列）");ax.set_ylabel("源层簇")
            for i,j in zip(*np.where(transition>=0.1)):
                ax.text(j,i,f"{100*transition[i,j]:.0f}",ha="center",va="center",color="white",fontsize=7)
            ax.set_title(title+"\n每行表示源簇分流到目标簇的比例（%）")
            fig.colorbar(im,ax=ax,label="占源簇的百分比")
            fig.savefig(directory/"transition_heatmap.png",dpi=150);plt.close(fig)
            table_rows=[]
            for i in range(k):
                order=np.argsort(c[i])[::-1]
                destinations=[]
                for j in order:
                    if c[i,j]==0:continue
                    fraction=c[i,j]/c[i].sum()
                    flow_rows.append({"source_layer":source_layer,"source_head":source_head,"target_layer":target_layer,"target_head":target_head,
                                      "source_cluster":i+1,"target_cluster":int(j)+1,"patch_count":int(c[i,j]),
                                      "fraction_of_source":float(fraction),"fraction_of_target":float(c[i,j]/c[:,j].sum())})
                    if fraction>=.05:
                        target_url=f"../../value_clusters_1pct/layer_{target_layer:02d}/head_{target_head:02d}/cluster_{j+1:02d}.jpg"
                        destinations.append(f'<a href="{target_url}">C{j+1:02d}</a>：{c[i,j]}个（{fraction:.1%}）')
                source_url=f"../../value_clusters_1pct/layer_{source_layer:02d}/head_{source_head:02d}/cluster_{i+1:02d}.jpg"
                significant=int((c[i]/c[i].sum()>=.2).sum())
                table_rows.append(f'<tr><td><a href="{source_url}">C{i+1:02d}原图区域</a></td><td>{c[i].sum()}</td><td>'+"；".join(destinations)+f'</td><td>{significant}</td></tr>')
            doc=(f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{title}簇变化</title>'
                 '<style>body{max-width:1350px;margin:25px auto;padding:12px;font-family:system-ui;line-height:1.65}img{width:100%}.flow{max-width:900px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}a{color:#14609a}</style>'
                 f'<h1>{title}：同一批patch的簇归属变化</h1><p>该目标头从16个候选头中按ARI选取；ARI={row["ari"]:.3f}，NMI={row["nmi"]:.3f}。'
                 f'最佳一对一簇标签对齐后，{row["aligned_same_fraction"]:.1%}的patch颜色保持一致。'
                 f'有{row["split_source_clusters"]}/21个源簇分流至至少两个各占20%的目标簇；有{row["merged_target_clusters"]}/21个目标簇从至少两个各占20%的源簇接收成员。</p>'
                 '<p>前后层簇都是独立聚类结果，编号不直接对应。每条带追踪同一帧同一patch位置；颜色按最大交集对齐用于阅读。</p>'
                 '<p><a href="../index.html">返回全部头配对</a> · <a href="transitions.npz">完整转移计数与标签</a></p>'
                 '<img src="aligned_regions.png"><h2>簇分流/汇聚</h2><img class="flow" src="cluster_flows.png">'
                 '<h2>完整转移矩阵</h2><img src="transition_heatmap.png"><h2>逐源簇追踪区域</h2>'
                 '<p>点击源簇或目标簇可看原图区域。此表列出占源簇≥5%的去向；完整非零流量保存在总CSV和NPZ中。</p>'
                 '<table><tr><th>源簇</th><th>patch数</th><th>主要目标簇</th><th>占比≥20%的子簇数</th></tr>'+"".join(table_rows)+'</table></html>')
            (directory/"index.html").write_text(doc)
            cards.append(f'<a href="{name}/index.html"><img loading="lazy" src="{name}/aligned_regions.png"><p>H{source_head:02d} → H{target_head:02d}，ARI={row["ari"]:.3f}，查看21簇分流与合并</p></a>')
        relevant=[r for r in selected_rows if r["source_layer"]==source_layer]
        summary={"source_layer":source_layer,"target_layer":target_layer,"head_matching":"row-best ARI; multiple source heads may choose same target",
                 "mean_best_ari":float(ari[np.arange(16),best].mean()),"mean_nmi_of_selected_pairs":float(nmi[np.arange(16),best].mean()),
                 "all_pairs_mean_ari":float(ari.mean()),"one_to_one_head_mean_ari":float(ari[hi,hj].mean()),
                 "random_refitted_best_ari_mean":float(null.mean()),"random_refitted_best_ari_q025":float(np.quantile(null,.025)),
                 "random_refitted_best_ari_q975":float(np.quantile(null,.975)),
                 "mean_aligned_same_fraction":float(np.mean([r["aligned_same_fraction"] for r in relevant])),
                 "mean_split_source_clusters":float(np.mean([r["split_source_clusters"] for r in relevant])),
                 "mean_merged_target_clusters":float(np.mean([r["merged_target_clusters"] for r in relevant]))}
        summaries.append(summary)
        sections.append(f'<h2>第{source_layer+1}→{target_layer+1}层</h2><p>平均最佳ARI={summary["mean_best_ari"]:.3f}，NMI={summary["mean_nmi_of_selected_pairs"]:.3f}，'
                        f'标签对齐后保持比例={summary["mean_aligned_same_fraction"]:.1%}。</p><div class="grid">'+"".join(cards)+'</div>')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    figure.savefig(root/"head_partition_similarity.png",dpi=180);plt.close(figure)
    for filename,records in (("all_head_pairs.csv",pair_rows),("selected_head_pairs.csv",selected_rows),("cluster_flows.csv",flow_rows)):
        with (root/filename).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(root/"head_similarity_matrices.npz",**matrices)
    (root/"summary.json").write_text(json.dumps({"input_metadata":metadata,"retention":.01,"clusters_per_head":21,
        "heads_assumed_aligned":False,"clusters_assumed_aligned":False,"head_matching":"all 16x16 ARI scores; row-best matching for detailed changes; independent of cluster label IDs",
        "metrics":"ARI adjusts pair agreement for chance; NMI uses arithmetic entropy normalization; both invariant to relabeling",
        "null":"shared within-frame target-label position permutations, preserving class sizes per frame and cross-head label relationships; best target refitted each time",
        "null_repetitions":args.null_repetitions,"split_merge_threshold":.20,
        "limitation":"descriptive for one sequence/two frames; heads selected and evaluated on same input; split/merge describes membership reassignments between independent fixed-K partitions",
        "adjacent_layers":summaries},ensure_ascii=False,indent=2)+"\n")
    render_examples(root, metadata, selected_rows)
    doc=('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>前三层Value簇的跨层演化</title>'
         '<style>body{max-width:1450px;margin:25px auto;padding:12px;font-family:system-ui;line-height:1.7}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%}a{color:#14609a}</style>'
         '<h1>前三层：21个Value簇的跨层归属变化</h1><p>同一两帧、每头2072个patch、21个medoids（约1%）。'
         '比较完整簇划分，按ARI在16×16头对间选最佳，不按头编号或簇编号对应。</p>'
         '<p><a href="../value_clusters_1pct/index.html">先看前三层全部patch分类与每个medoid覆盖区域</a> · '
         '<a href="selected_head_pairs.csv">32个最佳头配对统计</a> · <a href="cluster_flows.csv">全部簇流量CSV</a></p>'
         '<img src="head_partition_similarity.png"><h2>单簇区域变化示例</h2>'
         '<p>每组相邻层选ARI最高的头对，再选其中最大的明显分流源簇。仅作图解，不代表平均水平；统计见下方全部头对。</p>'
         '<img src="example_L0_L1_split.png"><img src="example_L1_L2_split.png">'+"".join(sections)+'</html>')
    (root/"index.html").write_text(doc)


if __name__=="__main__":
    main()
