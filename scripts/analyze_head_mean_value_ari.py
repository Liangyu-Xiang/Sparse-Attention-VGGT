#!/usr/bin/env python3
"""Mean raw V over heads BEFORE clustering, then compare adjacent layers' ARI."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from scripts.analyze_head_mean_value_clusters import average_heads
from scripts.report_value_head_ari import all_head_ari, adjacent_layer_pairs
from scripts.visualize_vggt_value_medoids import pairwise_distances, pam


def plot_layer_ari(rows,destination,frames,ratio):
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    with plt.rc_context({"font.family":["DejaVu Sans","Droid Sans Fallback"]}):
        fig,ax = plt.subplots(figsize=(10.5,4.8),constrained_layout=True)
        x = np.arange(len(rows))
        for key,name,colour in (
            ("per_current_head_best_previous_mean_ari","后层每头选择最佳前层头，再平均ARI","#226bb0"),
            ("mean_raw_value_then_cluster_ari","先平均16头V，再聚类计算ARI","#bd4936"),
        ):
            values = [row[key]for row in rows]
            ax.plot(x,values,"o-",color=colour,label=name,linewidth=2)
            for xx,yy in zip(x,values):
                other_key = "mean_raw_value_then_cluster_ari" if key=="per_current_head_best_previous_mean_ari" else "per_current_head_best_previous_mean_ari"
                offset = 10 if yy>=rows[int(xx)][other_key] else -18
                ax.annotate(f"{yy:.3f}",(xx,yy),xytext=(0,offset),textcoords="offset points",ha="center",fontsize=9,color=colour)
        ax.set_xticks(x,[f'{r["previous_layer"]+1}→{r["current_layer"]+1}'for r in rows])
        ax.set_xlabel("相邻全局层（自然层编号）");ax.set_ylabel("ARI")
        ax.set_ylim(0,1);ax.grid(axis="y",alpha=.25)
        ax.set_title(f"{frames}帧输入，{ratio:.0%}代表点：两种相邻层ARI")
        ax.legend(loc="lower right",fontsize=10)
        fig.savefig(destination,dpi=180);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=Path("outputs/value_medoids/chess_n10_s5"))
    parser.add_argument("--ratio",type=float,choices=(.01,.05,.10),default=.10)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--max-swaps",type=int,default=3000)
    parser.add_argument("--num-layers",type=int,default=3)
    parser.add_argument("--resume",action="store_true",help="Reuse converged per-layer fits after validation")
    parser.add_argument("--cluster-only",action="store_true",help="Only create per-layer fits, without a shared report")
    parser.add_argument("--fit-layers",type=int,nargs="+",help="Optional disjoint layer shard; requires --cluster-only")
    args = parser.parse_args()
    try:
        layer_pairs = adjacent_layer_pairs(args.num_layers)
    except ValueError as error:
        parser.error(str(error))
    fit_layers = sorted(set(args.fit_layers)) if args.fit_layers is not None else list(range(args.num_layers))
    if any(not 0<=layer<args.num_layers for layer in fit_layers):
        parser.error("Fit layers must belong to the requested layer range")
    if args.fit_layers is not None and not args.cluster_only:
        parser.error("--fit-layers requires --cluster-only to avoid incomplete shared reports")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = json.loads((args.input_dir/"metadata.json").read_text())
    cache = np.load(args.input_dir/"values.npz")
    root = args.input_dir/f"head_mean_{round(100*args.ratio)}pct_ari"
    root.mkdir(exist_ok=True)
    n = metadata["values_per_head"]
    k = round(args.ratio*n)
    mean_values,partitions,layers_summary = {},[],[]
    for layer in fit_layers:
        start = time.perf_counter()
        raw = cache[f"layer_{layer}"]
        assert raw.shape==(16,n,64)
        values = average_heads(raw)
        mean_values[f"layer_{layer}"] = values
        directory = root/f"layer_{layer:02d}"
        resumed = args.resume and (directory/"clusters.npz").is_file() and (directory/"clustering.json").is_file()
        starts = None
        if resumed:
            saved = np.load(directory/"clusters.npz")
            result = {**json.loads((directory/"clustering.json").read_text()),
                      **{key:saved[key]for key in ("medoids","assignments","distances")}}
        else:
            distance = pairwise_distances(torch.from_numpy(values).to(args.device))
            with torch.inference_mode():
                starts = [pam(distance,k,method,20261015+100*layer,args.max_swaps)for method in ("build","sample")]
            result = min(starts,key=lambda run:run["objective"])
        if not result["converged"]:
            raise RuntimeError(f"Layer {layer}: selected PAM solution did not converge")
        medoids,labels,nearest = (result[key]for key in ("medoids","assignments","distances"))
        counts = np.bincount(labels,minlength=k)
        assert len(set(medoids))==k and (counts>0).all()
        np.testing.assert_array_equal(labels[medoids],np.arange(k))
        # Independent FP64 distance/assignment check for every averaged patch.
        x = values.astype(np.float64)
        sq = np.einsum("nd,nd->n",x,x)
        exact = np.sqrt(np.maximum(sq[:,None]+sq[medoids][None,:]-2*x@x[medoids].T,0))
        chosen = np.linalg.norm(x-x[medoids[labels]],axis=1)
        tolerance = 2e-4*max(1.,float(exact.max()))
        assert np.all(chosen<=exact.min(1)+tolerance)
        np.testing.assert_allclose(nearest,chosen,rtol=2e-4,atol=tolerance)
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory/"clusters.npz",medoids=medoids,assignments=labels,distances=nearest,cluster_sizes=counts)
        stats = {key:value for key,value in result.items()if key not in ("medoids","assignments","distances")}
        if starts is not None:
            stats["starts"] = [{key:value for key,value in run.items()if key not in ("medoids","assignments","distances")}for run in starts]
        (directory/"clustering.json").write_text(json.dumps(stats,indent=2)+"\n")
        row = {"layer":layer,"heads_averaged":16,"patches":n,"medoids":k,"converged":True,
               "swaps":result["swaps"],"objective":result["objective"],"mean_distance":float(nearest.mean()),
               "cluster_size_min":int(counts.min()),"cluster_size_max":int(counts.max()),
               "resumed":resumed,"elapsed_seconds":time.perf_counter()-start}
        layers_summary.append(row)
        partitions.append(labels)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    if args.cluster_only:
        print(f"Completed mean-V layer shard {fit_layers}; shared reports were not modified",flush=True)
        return
    np.savez_compressed(root/"mean_values.npz",**mean_values)
    pair_matrices = np.load(args.input_dir/f"all_head_ari_{round(100*args.ratio)}pct/ari_matrices.npz")
    rows = []
    for previous,current in layer_pairs:
        a,b = partitions[previous:current+1]
        ari = float(all_head_ari(a[None],b[None])[0,0])
        assert abs(ari-adjusted_rand_score(a,b))<1e-12
        matrix = pair_matrices[f"L{previous:02d}_L{current:02d}"]
        row = {"previous_layer":previous,"current_layer":current,
               "mean_raw_value_then_cluster_ari":ari,
               "per_current_head_best_previous_mean_ari":float(matrix.max(axis=0).mean()),
               "all_head_pairs_mean_ari_reference":float(matrix.mean())}
        rows.append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    with (root/"adjacent_layers.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot_layer_ari(rows,root/"adjacent_layers_ari.png",len(metadata["frame_indices"]),args.ratio)
    (root/"summary.json").write_text(json.dumps({"input_metadata":metadata,"layers":list(range(args.num_layers)),"retention":args.ratio,"actual_retention":k/n,
        "heads_averaged":16,"clusters_per_layer":k,"patches_per_layer":n,
        "aggregation":"coordinate-wise arithmetic mean over 16 raw V heads, FP32, no L2 normalization or basis alignment",
        "operation_order":"average V first -> layerwise independent k-medoids -> ARI on all patch memberships",
        "not_equivalent_to":"averaging pairwise ARIs, averaging cluster IDs, averaging attention probabilities, or averaging projected attention outputs",
        "basis_caveat":"Independently learned head feature bases are not aligned; coordinate-wise mean is basis dependent and may cancel features",
        "matching_reference":"Each current-layer head independently selects the previous-layer head maximizing ARI; previous heads may be reused",
        "optimizer":"two PAM starts; exhaustive best single swaps; converged single-swap local optimum, not certified global optimum",
        "validation":"all averaged values and nearest assignments checked in FP64; all adjacent-layer ARIs checked against sklearn",
        "layers_summary":layers_summary,"adjacent_layers":rows},ensure_ascii=False,indent=2)+"\n")
    table = ''.join(f'<tr><td>第{r["previous_layer"]+1}→{r["current_layer"]+1}层</td><td>{r["per_current_head_best_previous_mean_ari"]:.6f}</td>'
                    f'<td>{r["mean_raw_value_then_cluster_ari"]:.6f}</td><td>{r["all_head_pairs_mean_ari_reference"]:.6f}</td></tr>'for r in rows)
    (root/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>后层择前层匹配与头平均V的层间ARI</title>'
        '<style>body{max-width:1150px;margin:30px auto;font-family:system-ui;line-height:1.8}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        f'<h1>{len(metadata["frame_indices"])}帧输入、前{args.num_layers}层、{args.ratio:.0%}代表点：两种跨层ARI</h1>'
        '<p>方法一：每个后层头在前层16个头中独立选最大ARI，再对16个后层头平均，允许重用前层头。'
        '方法二：每层先对16个头的原始64维V逐坐标取平均，再独立聚类，比较两层全部patch的归属。'
        '两者不同，方法二也不同于全部256个ARI的平均。</p>'
        '<table><tr><th>相邻层</th><th>后层每头最佳前层匹配：平均ARI</th><th>先平均V再聚类：层间ARI</th><th>全部256头对均值（参考）</th></tr>'+table+'</table>'
        '<img style="max-width:100%" src="adjacent_layers_ari.png">'
        '<p>注意：头特征基底未对齐，原始V平均具有基底依赖性，不能直接视作多头共识聚类。</p>'
        '<p><a href="adjacent_layers.csv">统计CSV</a> · <a href="summary.json">定义与完整元数据</a> · '
        f'<a href="../all_head_ari_{round(100*args.ratio)}pct/best_previous_heads.csv">逐后层头的最佳前层匹配</a></p></html>')


if __name__=="__main__":
    main()
