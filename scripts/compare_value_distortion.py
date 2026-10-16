#!/usr/bin/env python3
"""Global vs frozen+recenter vs two anchored neighbor-local Lloyd iterations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

from scripts.report_value_neighborhood_recall import open_values, load_fit, verify_distances, write_csv
from scripts.value_local_distortion import distortion, recenter_partition, source_neighborhoods, local_refine


def gap(value, reference):
    if reference <= 0:
        return 0. if value == 0 else float("inf")
    return 100*(value-reference)/reference


def save_partition(path, values, labels, medoids):
    objective, distances = distortion(values, labels, medoids)
    np.savez_compressed(path, assignments=labels, medoids=medoids, distances=distances,
                        cluster_sizes=np.bincount(labels, minlength=len(medoids)))
    return objective/len(values)


def compare_case(source_values, target_values, source_fit, target_fit, directory):
    directory.mkdir(parents=True, exist_ok=True)
    n = len(target_values)
    source_labels, source_medoids = source_fit
    target_labels, target_medoids = target_fit
    reference = distortion(target_values, target_labels, target_medoids)[0]/n
    frozen_medoids = recenter_partition(target_values, source_labels, source_medoids)
    frozen = save_partition(directory / "frozen.npz", target_values, source_labels, frozen_medoids)
    row = {"global_distortion": reference, "frozen_distortion": frozen,
           "frozen_gap_pct": gap(frozen, reference), "anchors": len(source_medoids),
           "inherited_medoid_distortion": distortion(target_values, source_labels, source_medoids)[0]/n}
    all_neighbors = source_neighborhoods(source_values, source_medoids, 8)
    for q in (4, 8):
        neighbors = all_neighbors[:, :q+1]
        fit = local_refine(target_values, source_labels, source_medoids, neighbors,
                           max_iterations=2, anchors=source_medoids,
                           save_iterations=(1, 2), stop_when_converged=False)
        np.save(directory / f"neighbors_q{q}.npy", neighbors)
        for iteration in (1, 2):
            snapshot = fit["snapshots"][iteration]
            labels, medoids = snapshot["assignments"], snapshot["medoids"]
            np.testing.assert_array_equal(labels[source_medoids], np.arange(len(medoids)))
            assert np.all(np.any(neighbors[source_labels] == labels[:, None], axis=1))
            assert (np.bincount(labels, minlength=len(medoids)) > 0).all()
            assert np.all(labels[medoids] == np.arange(len(medoids)))
            value = save_partition(directory / f"local_q{q}_iter{iteration}.npz", target_values, labels, medoids)
            prefix = f"local_q{q}_iter{iteration}"
            row[f"{prefix}_distortion"] = value
            row[f"{prefix}_gap_pct"] = gap(value, reference)
            row[f"{prefix}_migration_fraction"] = float(np.mean(labels != source_labels))
        assert row[f"local_q{q}_iter2_distortion"] <= row[f"local_q{q}_iter1_distortion"]+1e-10*max(1., reference)
        (directory / f"local_q{q}_history.json").write_text(json.dumps({
            "initial_medoids": "Inherited previous-layer medoid token indices, evaluated in current V",
            "fixed_anchor_tokens": source_medoids.tolist(), "fixed_source_neighborhoods": True,
            "history": fit["history"], "fixed_point_after_two_iterations": fit["converged"]}, indent=2) + "\n")
    return row


def summarize(rows):
    output = []
    for source in sorted({r["source_layer"] for r in rows}):
        for mode in ("matched_head_mean", "mean_value"):
            group = [r for r in rows if r["source_layer"] == source and r["mode"] == mode]
            if not group:
                continue
            row = {"source_layer": source, "target_layer": source+1, "mode": mode, "cases": len(group)}
            for key in group[0]:
                if key.endswith("_distortion") or key.endswith("_migration_fraction"):
                    row[key] = float(np.mean([r[key] for r in group]))
            row["frozen_gap_pct"] = gap(row["frozen_distortion"], row["global_distortion"])
            for q in (4, 8):
                for iteration in (1, 2):
                    prefix = f"local_q{q}_iter{iteration}"
                    row[f"{prefix}_gap_pct"] = gap(row[f"{prefix}_distortion"], row["global_distortion"])
                    row[f"{prefix}_mean_individual_gap_pct"] = float(np.mean([r[f"{prefix}_gap_pct"] for r in group]))
            output.append(row)
    return output


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis, mode, title in zip(axes, ("matched_head_mean", "mean_value"),
                                 ("ARI-best matched heads (ratio of mean distortions)", "Mean raw V, then cluster")):
        selected = [r for r in summary if r["mode"] == mode]
        for key, label, style in (("frozen_gap_pct", "Frozen + new medoids", "o-"),
                                   ("local_q4_iter1_gap_pct", "q=4, 1 iteration", "o--"),
                                   ("local_q4_iter2_gap_pct", "q=4, 2 iterations", "o-"),
                                   ("local_q8_iter1_gap_pct", "q=8, 1 iteration", "s--"),
                                   ("local_q8_iter2_gap_pct", "q=8, 2 iterations", "s-")):
            axis.plot([r["source_layer"]+1 for r in selected], [r[key] for r in selected], style, label=label, markersize=4)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_xticks([r["source_layer"]+1 for r in selected], [f"{r['source_layer']+1} to {r['target_layer']+1}" for r in selected], rotation=35)
        axis.set_ylabel("Distortion increase vs global (%)")
        axis.set_xlabel("Adjacent global layer")
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_report(root, metadata, ratio, rows, summary, checks):
    write_csv(root / "all_cases.csv", rows)
    write_csv(root / "per_head.csv", [r for r in rows if r["mode"] == "matched_head_mean"])
    write_csv(root / "layer_summary.csv", summary)
    write_csv(root / "global_fit_checks.csv", checks)
    (root / "summary.json").write_text(json.dumps({
        "input": metadata, "retention": ratio, "k": round(metadata["values_per_head"]*ratio),
        "distortion": "Mean current-layer raw V Euclidean distance, not squared distance",
        "frozen": "Source memberships fixed; exact current-layer medoid within every original cluster",
        "local": "Exactly two synchronous assignment then exact medoid-update iterations; initial medoid token indices inherited from source; source medoid anchors stay in their original clusters",
        "neighborhoods": "Fixed directed q nearest source medoids plus self; each token permanently uses original source-label candidates",
        "head_matching": "Max target ARI per source head using this retention; many-to-one allowed; matching and evaluation on same input",
        "source_partition": "Independent global fit of the previous layer for every adjacent-layer comparison; no multi-layer local rollout",
        "head_summary_gap": "Ratio of mean distortions, NOT mean of per-head percent gaps; both are in CSV",
        "global_optimum_guaranteed": False, "summary": summary,
        "validation": {"global_fits": len(checks), "max_assignment_regret_fp64": max(c["max_assignment_regret_fp64"] for c in checks)}
    }, ensure_ascii=False, indent=2) + "\n")
    lines = [f"# Global / Frozen / Neighbor-Restricted distortion：{metadata['num_frames']}帧，{ratio:.0%} retention", "",
             f"{metadata['scene']}/{metadata['sequence']}，前{len(metadata['global_layer_indices'])}个global层，N={metadata['values_per_head']}，K={round(metadata['values_per_head']*ratio)}。所有distortion为当前层原始V的平均欧氏距离。", "",
             "Frozen冻结成员但重新计算各簇精确medoid。Local从上一层medoid token位置初始化，按固定上一层邻居集合做同步reassignment→精确medoid update，共2轮。上一层medoid token始终锚定在原簇；另存第1轮。", "",
             "逐头模式为每个源头选ARI最大的下一层头，使用本次retention的全局划分重新匹配；允许多对一。表中先平均distortion，再计算相对global的增量百分比。平均V模式先平均16头原始向量，再独立进行三种聚类。", ""]
    lines += ["每组相邻层都使用上一层独立全局聚类作为同一个起点，评估单步迁移；不将Local输出递归作为下一对层的源划分。", ""]
    for mode, title in (("matched_head_mean", "逐头最佳匹配后平均"), ("mean_value", "先平均16头V")):
        lines += ["", f"## {title}", "", "| 相邻层 | Global D | Frozen D | Frozen Gap | Local q=4 D | q=4 Gap | Local q=8 D | q=8 Gap |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary:
            if row["mode"] != mode:
                continue
            lines.append(f"| {row['source_layer']+1}→{row['target_layer']+1} | {row['global_distortion']:.6f} | {row['frozen_distortion']:.6f} | {row['frozen_gap_pct']:+.2f}% | {row['local_q4_iter2_distortion']:.6f} | {row['local_q4_iter2_gap_pct']:+.2f}% | {row['local_q8_iter2_distortion']:.6f} | {row['local_q8_iter2_gap_pct']:+.2f}% |")
    lines += ["", "第1轮与第2轮的完整对照见layer_summary.csv和曲线；all_cases.csv含136个配对的全部结果；per_head.csv含128个逐头配对。每个case保存Frozen与两种q的两轮完整归属、medoid、距离和历史。", "",
              "Global为独立从头PAM的当前实现参考，不保证全局最优；若出现负Gap则如实报告，不截断。固定两轮并不保证Local已收敛，也不保证更大的q在有限迭代后一定更好。锚点约束只作用于Local。", "",
              "前后头的向量坐标不作跨头距离比较；邻居图在源头V空间，distortion在目标头当前V空间。头匹配与评估同输入，结果不代表独立数据泛化或任务精度/实际加速。", "",
              "![Distortion gaps](distortion_gaps.png)"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    plot(summary, root / "distortion_gaps.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=.10)
    args = parser.parse_args()
    root = args.input_dir
    metadata = json.loads((root / "metadata.json").read_text())
    layers = metadata["global_layer_indices"]
    n, k = metadata["values_per_head"], round(args.ratio*metadata["values_per_head"])
    destination = root / f"distortion_comparison_{round(100*args.ratio)}pct"
    destination.mkdir(exist_ok=True)
    fits, mean_fits, checks = {}, {}, []
    ratio_label = f"{args.ratio:.0%}"
    with open_values(root, layers) as cache:
        for layer in layers:
            values = cache[f"layer_{layer}"]
            fits[layer] = []
            for head in range(16):
                directory = root / f"keep_{ratio_label}/layer_{layer:02d}"
                labels, medoids, stats = load_fit(directory, f"head_{head:02d}", n, k)
                check = verify_distances(values[head], (labels, medoids), directory, f"head_{head:02d}")
                checks.append({"layer": layer, "head": head, "objective": stats["objective"],
                               "algorithm": stats.get("algorithm", "PAM exhaustive best single swap"), **check})
                fits[layer].append((labels, medoids))
            directory = root / f"head_mean_{round(100*args.ratio)}pct_ari/layer_{layer:02d}"
            labels, medoids, stats = load_fit(directory, "clusters", n, k)
            checks.append({"layer": layer, "head": -1, "objective": stats["objective"],
                           "algorithm": stats.get("algorithm", "PAM exhaustive best single swap"),
                           **verify_distances(values.mean(0, dtype=np.float32), (labels, medoids), directory, "clusters")})
            mean_fits[layer] = (labels, medoids)
            print(f"FP64 check: layer={layer+1}, K={k}", flush=True)
        rows, pairs = [], []
        for source, target in zip(layers[:-1], layers[1:]):
            previous_values, current_values = cache[f"layer_{source}"], cache[f"layer_{target}"]
            ari = np.array([[adjusted_rand_score(a[0], b[0]) for b in fits[target]] for a in fits[source]])
            np.savetxt(destination / f"L{source:02d}_L{target:02d}_ari.csv", ari, delimiter=",")
            for head in range(16):
                start = time.perf_counter()
                matched = int(ari[head].argmax())
                description = {"source_layer": source, "target_layer": target, "source_head": head,
                               "target_head": matched, "ari": float(ari[head, matched])}
                pairs.append(description)
                result = compare_case(previous_values[head], current_values[matched], fits[source][head], fits[target][matched],
                                      destination / f"per_head/L{source:02d}_H{head:02d}_to_L{target:02d}_H{matched:02d}")
                rows.append({"mode": "matched_head_mean", **description, **result})
                print(f"Local compare L{source+1} H{head:02d} -> L{target+1} H{matched:02d}: {time.perf_counter()-start:.2f}s", flush=True)
            result = compare_case(previous_values.mean(0, dtype=np.float32), current_values.mean(0, dtype=np.float32),
                                  mean_fits[source], mean_fits[target], destination / f"mean_value/L{source:02d}_to_L{target:02d}")
            rows.append({"mode": "mean_value", "source_layer": source, "target_layer": target, "source_head": -1,
                         "target_head": -1, "ari": float(adjusted_rand_score(mean_fits[source][0], mean_fits[target][0])), **result})
            write_csv(destination / "head_matching.csv", pairs)
            write_csv(destination / "all_cases.partial.csv", rows)
        write_report(destination, metadata, args.ratio, rows, summarize(rows), checks)
    print(f"COMPLETE: {destination/'report.md'}", flush=True)
    # The last finished long-input sequence also produces the shared summary.
    if metadata["num_frames"] == 100:
        scenes = ("chess", "fire", "heads", "office", "redkitchen")
        parent = root.parent
        if all((parent / f"{scene}_seq01_n100_s10/distortion_comparison_10pct/report.md").is_file() for scene in scenes):
            from scripts.batch_value_distortion import aggregate_long
            aggregate_long(parent)


if __name__ == "__main__":
    main()
