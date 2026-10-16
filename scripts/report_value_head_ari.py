#!/usr/bin/env python3
"""Report ALL adjacent-layer head-pair ARIs, without averaging head values."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.metrics import adjusted_rand_score

def all_head_ari(source, target):
    """Exact ARI using only nonzero joint counts, with no H*H*K*K allocation."""
    source,target = np.asarray(source),np.asarray(target)
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1]:
        raise ValueError("Expected head-by-patch partitions on the same patch domain")
    if not np.issubdtype(source.dtype,np.integer) or not np.issubdtype(target.dtype,np.integer):
        raise ValueError("Cluster labels must be integers")
    if (source<0).any() or (target<0).any():
        raise ValueError("Cluster labels must be nonnegative")
    n = source.shape[1]
    scores = np.ones((len(source),len(target)),dtype=np.float64)
    if n<2:
        return scores
    def pairs(labels):
        sizes = np.unique(labels,return_counts=True)[1].astype(np.float64)
        return float((sizes*(sizes-1)/2).sum())
    source_pairs = [pairs(a) for a in source]
    target_pairs = [pairs(b) for b in target]
    base = int(target.max())+1
    total_pairs = n*(n-1)/2
    for i,a in enumerate(source):
        for j,b in enumerate(target):
            joint = np.unique(a.astype(np.int64)*base+b,return_counts=True)[1].astype(np.float64)
            same_both = (joint*(joint-1)/2).sum()
            expected = source_pairs[i]*target_pairs[j]/total_pairs
            denominator = (source_pairs[i]+target_pairs[j])/2-expected
            if abs(denominator)>1e-12:
                scores[i,j] = (same_both-expected)/denominator
    return scores


def match_next_to_previous(ari):
    """Columns are current heads; independently maximize over previous rows."""
    ari = np.asarray(ari,dtype=np.float64)
    if ari.ndim!=2 or min(ari.shape)<1 or not np.isfinite(ari).all():
        raise ValueError("Expected a finite nonempty previous-head by current-head ARI matrix")
    previous = ari.argmax(axis=0)
    return previous,ari[previous,np.arange(ari.shape[1])]


def adjacent_layer_pairs(num_layers):
    if not 2<=num_layers<=24:
        raise ValueError("Number of global layers must be between 2 and 24")
    return list(zip(range(num_layers-1),range(1,num_layers)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("outputs/value_medoids/chess_f000000_f000020"))
    parser.add_argument("--ratio",type=float,choices=(.01,.05,.10),default=.01,
                        help="Medoid fraction; all patch tokens participate in clustering")
    parser.add_argument("--num-layers",type=int,default=3)
    args = parser.parse_args()
    try:
        layer_pairs = adjacent_layer_pairs(args.num_layers)
    except ValueError as error:
        parser.error(str(error))
    ratio_label = f"{args.ratio:.0%}"
    keep = f"keep_{ratio_label}"
    root = args.input_dir/f"all_head_ari_{round(100*args.ratio)}pct"
    root.mkdir(exist_ok=True)
    metadata = json.loads((args.input_dir/"metadata.json").read_text())
    layers = range(args.num_layers)
    labels = [np.stack([np.load(args.input_dir/f"{keep}/layer_{l:02d}/head_{h:02d}.npz")["assignments"]
                        for h in range(16)]) for l in layers]
    k = len(np.load(args.input_dir/keep/"layer_00/head_00.npz")["medoids"])
    patches = labels[0].shape[1]
    frames = len(metadata["frame_indices"])
    assert patches==metadata["values_per_head"] and k==round(args.ratio*patches)
    for layer in layers:
        for head in range(16):
            saved = np.load(args.input_dir/f"{keep}/layer_{layer:02d}/head_{head:02d}.npz")
            assert len(saved["medoids"])==k and len(saved["assignments"])==patches
            assert saved["assignments"].min()>=0 and saved["assignments"].max()<k
            assert np.all(np.bincount(saved["assignments"],minlength=k)>0)
            stats = json.loads((args.input_dir/f"{keep}/layer_{layer:02d}/head_{head:02d}.json").read_text())
            if not stats["converged"]:
                raise RuntimeError(f"Unconverged clustering: layer {layer}, head {head}")
    matrices, records, summaries, matching_rows = {}, [], [], []
    for source,target in layer_pairs:
        ari = all_head_ari(labels[source],labels[target])
        # Independent reference computation for ALL 256 entries, not a sample.
        reference = np.array([[adjusted_rand_score(a,b) for b in labels[target]] for a in labels[source]])
        np.testing.assert_allclose(ari,reference,rtol=0,atol=1e-12)
        matrices[f"L{source:02d}_L{target:02d}"] = ari
        with (root/f"L{source:02d}_L{target:02d}_ari_matrix.csv").open("w",newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_head / target_head"]+[f"H{h:02d}" for h in range(16)])
            writer.writerows([[f"H{h:02d}",*row] for h,row in enumerate(ari)])
        for a in range(16):
            for b in range(16):
                records.append({"source_layer":source,"target_layer":target,"source_head":a,"target_head":b,"ari":float(ari[a,b])})
        best = np.unravel_index(ari.argmax(),ari.shape)
        previous,scores = match_next_to_previous(ari)
        for current_head,previous_head in enumerate(previous):
            matching_rows.append({"previous_layer":source,"current_layer":target,"current_head":current_head,
                                  "best_previous_head":int(previous_head),"ari":float(scores[current_head]),
                                  "same_index_ari_reference":float(ari[current_head,current_head]),
                                  "gain_over_same_index":float(scores[current_head]-ari[current_head,current_head])})
        summary = {"source_layer":source,"target_layer":target,"head_pairs":256,"all_pair_mean_ari":float(ari.mean()),
                   "all_pair_median_ari":float(np.median(ari)),"all_pair_min_ari":float(ari.min()),
                   "all_pair_max_ari":float(ari.max()),"best_source_head":int(best[0]),"best_target_head":int(best[1]),
                   "row_best_mean_ari":float(ari.max(1).mean()),
                   "per_current_head_best_previous_mean_ari":float(scores.mean()),
                   "best_previous_heads_by_current_head":previous.tolist(),
                   "matching_direction":"current-layer head independently chooses highest-ARI previous-layer head; reuse allowed",
                   "sklearn_max_absolute_difference":float(np.abs(reference-ari).max())}
        summaries.append(summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    plt.rcParams["font.family"]=["DejaVu Sans","Droid Sans Fallback"]
    figures = []
    colour_max = max(.5,np.ceil(max(ari.max() for ari in matrices.values())*10)/10)
    colour_min = min(0,min(ari.min() for ari in matrices.values()))
    for source,((name,ari),summary) in enumerate(zip(matrices.items(),summaries)):
        previous,scores = match_next_to_previous(ari)
        fig,ax = plt.subplots(figsize=(11.8,10.8),constrained_layout=True)
        im = ax.imshow(ari,vmin=colour_min,vmax=colour_max,cmap="viridis")
        ax.set_xticks(range(16),[f"H{h:02d}" for h in range(16)],rotation=45)
        ax.set_yticks(range(16),[f"H{h:02d}" for h in range(16)])
        ax.set_xlabel(f"第{source+2}层的头（列）",fontsize=13)
        ax.set_ylabel(f"第{source+1}层的头（行）",fontsize=13)
        ax.set_title(f"第{source+1}层 → 第{source+2}层：全部16×16头对的ARI\n"
                     f"{frames}帧输入，各头独立{ratio_label} Value聚类，{k}簇/{patches}个patch；未对头取平均\n"
                     f"粗体=每列最佳前层头；匹配后平均ARI={scores.mean():.4f}",fontsize=14)
        for a in range(16):
            for b in range(16):
                ax.text(b,a,f"{ari[a,b]:.3f}",ha="center",va="center",fontsize=8.5,
                        color="black" if ari[a,b]>(colour_min+.66*(colour_max-colour_min)) else "white",
                        fontweight="bold" if a==previous[b] else "normal")
        ax.scatter(np.arange(16),previous,marker="s",s=1250,facecolors="none",edgecolors="#ff5555",linewidths=1.5)
        ax.set_xticks(np.arange(-.5,16,1),minor=True)
        ax.set_yticks(np.arange(-.5,16,1),minor=True)
        ax.grid(which="minor",color="white",alpha=.17,linewidth=.5)
        ax.tick_params(which="minor",bottom=False,left=False)
        fig.colorbar(im,ax=ax,shrink=.8,label="ARI（相同划分=1，机会水平≈0；所有图采用相同色标）")
        fig.savefig(root/f"{name}_ari.png",dpi=160)
        figures.append(fig)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(root/"all_adjacent_head_ari.pdf") as pdf:
        for fig in figures:
            pdf.savefig(fig)
            plt.close(fig)
    with (root/"all_head_pairs_ari.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    with (root/"best_previous_heads.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(matching_rows[0]))
        writer.writeheader();writer.writerows(matching_rows)
    np.savez_compressed(root/"ari_matrices.npz",**matrices)
    (root/"summary.json").write_text(json.dumps({"layers":list(layers),"retention":args.ratio,"heads_per_layer":16,
        "actual_retention":k/patches,"ratio_semantics":"medoid fraction; all patch values participate in clustering and ARI",
        "ari_counting":"exact nonzero joint counts; no dense H*H*K*K contingency tensor",
        "clusters_per_head":k,"patches_per_head":patches,"num_frames":frames,"frame_indices":metadata["frame_indices"],
        "head_averaging":False,"head_index_correspondence_assumed":False,
        "matching_direction":"For each current-layer head (column), maximize ARI over all previous-layer heads (rows)",
        "one_to_one_constraint":False,"previous_head_reuse_allowed":True,
        "pair_count":len(records),"validation":f"all {len(records)} scores independently verified against sklearn adjusted_rand_score at atol=1e-12",
        "input_metadata":metadata,"scope":"layer IDs and head IDs are zero-based; all input frames are clustered jointly",
        "adjacent_layers":summaries},ensure_ascii=False,indent=2)+"\n")
    sections = ''.join(f'<h2>第{i+2}层每个头匹配第{i+1}层最佳头</h2><img src="{name}_ari.png"><p><a href="{name}_ari_matrix.csv">完整16×16数值CSV</a></p>'
                       for i,name in enumerate(matrices))
    (root/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>相邻层全部头对ARI</title>'
        '<style>body{max-width:1550px;margin:25px auto;font-family:system-ui;line-height:1.7}img{width:100%}a{color:#14609a}</style>'
        f'<h1>前{args.num_layers}层：相邻层全部注意力头的聚类相似度</h1>'
        f'<p>{frames}帧输入，每个头独立做{ratio_label} Value聚类（{k}个medoids），比较全部{patches}个patch的归属。'
        '未平均头的V；每张图有全部256个头对。粗体及红框表示每列最大值：每个后层头在前层全部16个头中独立选择最相似头。'
        '允许多个后层头选择同一个前层头，不按相同编号配对，不施加一对一约束。'
        'ARI不需要对齐簇标签，也不假设相同头编号具有对应关系；约0为机会水平，1为相同划分，可出现负值。</p>'
        f'<p><a href="best_previous_heads.csv">{len(matching_rows)}个后层头各自的最佳前层匹配</a> · '
        f'<a href="all_head_pairs_ari.csv">全部{len(records)}个ARI结果</a> · <a href="all_adjacent_head_ari.pdf">全部矩阵PDF</a></p>'
        +sections+'</html>')


if __name__=="__main__":
    main()
