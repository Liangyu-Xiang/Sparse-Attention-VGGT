#!/usr/bin/env python3
"""Cross-layer candidate-region recall, with fixed ARI-best head matching.

Recall tests the target representative's TOKEN INDEX in source-cluster unions.
It does not match cluster labels or claim that q+1 source clusters contain only
q+1 candidate values. Mean-head analysis averages raw V BEFORE clustering.
"""
from __future__ import annotations

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score


@contextmanager
def open_values(root, layers):
    if (root / "values").is_dir():
        yield {f"layer_{layer}": np.load(root / f"values/layer_{layer}.npy", mmap_mode="r") for layer in layers}
    else:
        with np.load(root / "values.npz") as cache:
            yield cache


def neighborhood_recall(values, source_labels, source_medoids,
                        target_labels, target_medoids, qs):
    values = np.asarray(values, dtype=np.float64)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    source_medoids = np.asarray(source_medoids, dtype=np.int64)
    target_labels = np.asarray(target_labels, dtype=np.int64)
    target_medoids = np.asarray(target_medoids, dtype=np.int64)
    n, k = len(source_labels), len(source_medoids)
    if len(values) != n or len(target_labels) != n:
        raise ValueError("Both partitions must describe the same ordered tokens")
    if any(q < 0 or q >= k for q in qs):
        raise ValueError("q must be between 0 and K-1")
    distances = cdist(values[source_medoids], values[source_medoids])
    # Force self first, including coincident medoids. Other ties use label order.
    np.fill_diagonal(distances, -np.inf)
    ordered = np.argsort(distances, axis=1, kind="stable")
    ranks = np.empty_like(ordered)
    ranks[np.arange(k)[:, None], ordered] = np.arange(k)
    representative_source_labels = source_labels[target_medoids[target_labels]]
    token_ranks = ranks[source_labels, representative_source_labels]
    sizes = np.bincount(source_labels, minlength=k)
    candidate_sizes = np.cumsum(sizes[ordered], axis=1)
    # Chance recall after permuting target representatives over token locations,
    # preserving their multiplicities and source-cluster sizes.
    representative_mass = np.bincount(representative_source_labels, minlength=k)
    mass_in_region = np.cumsum(representative_mass[ordered], axis=1)
    rows = []
    for q in qs:
        rows.append({
            "q": int(q), "recall": float(np.mean(token_ranks <= q)),
            "hits": int(np.count_nonzero(token_ranks <= q)), "tokens": n,
            "candidate_token_fraction": float(sizes @ candidate_sizes[:, q] / n**2),
            "independent_assignment_recall": float(sizes @ mass_in_region[:, q] / n**2),
        })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_fit(directory, name, n, k):
    arrays = np.load(directory / f"{name}.npz")
    stats = json.loads((directory / f"{name if name != 'clusters' else 'clustering'}.json").read_text())
    labels, medoids = arrays["assignments"], arrays["medoids"]
    assert stats["converged"], (directory, name)
    assert stats["best_final_swap_delta"] >= -stats["tolerance"]
    assert labels.shape == (n,) and medoids.shape == (k,)
    assert len(np.unique(medoids)) == k
    np.testing.assert_array_equal(labels[medoids], np.arange(k))
    np.testing.assert_array_equal(np.bincount(labels, minlength=k), arrays["cluster_sizes"])
    assert np.isclose(arrays["distances"].sum(dtype=np.float64), stats["objective"])
    return labels, medoids, stats


def verify_distances(values, fit, directory, name, block_size=1024):
    labels, medoids = fit
    # SciPy evaluates direct FP64 coordinate differences, independently of
    # the FP32 Gram-matrix calculation used by PAM.
    targets = np.asarray(values[medoids], dtype=np.float64)
    chosen = np.empty(len(labels), dtype=np.float64)
    maximum_regret, maximum_distance, disagreements = 0., 0., 0
    for start in range(0, len(values), block_size):
        stop = min(len(values), start+block_size)
        exact = cdist(np.asarray(values[start:stop], dtype=np.float64), targets)
        chosen[start:stop] = exact[np.arange(stop-start), labels[start:stop]]
        maximum_regret = max(maximum_regret, float((chosen[start:stop]-exact.min(axis=1)).max()))
        maximum_distance = max(maximum_distance, float(exact.max()))
        disagreements += int(np.count_nonzero(labels[start:stop] != exact.argmin(1)))
    tolerance = 2e-4 * max(1., maximum_distance)
    assert maximum_regret <= tolerance
    with np.load(directory / f"{name}.npz") as saved:
        np.testing.assert_allclose(saved["distances"], chosen, rtol=2e-4, atol=tolerance)
    return {"max_assignment_regret_fp64": maximum_regret,
            "argmin_disagreements_fp64": disagreements,
            "distance_check_tolerance": tolerance}


def make_plot(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    modes = (("matched_head_mean", "ARI-best head pairs (mean over 16 heads)"),
             ("mean_value", "Average 16 raw V heads, then cluster"))
    for axis, (mode, title) in zip(axes, modes):
        for source in sorted({row["source_layer"] for row in rows}):
            selected = [r for r in rows if r["mode"] == mode and r["source_layer"] == source]
            axis.plot([r["q"] + 1 for r in selected], [100*r["recall"] for r in selected],
                      "o-", markersize=3, label=f"{source+1} to {source+2}")
        axis.set_xscale("log", base=2)
        axis.set_ylim(0, 101)
        axis.set_xlabel("Source clusters searched (q + 1)")
        axis.set_ylabel("Neighborhood Recall (%)")
        axis.set_title(title)
        axis.grid(alpha=.25)
        axis.legend(title="Global layer", fontsize=8)
    fig.savefig(destination, dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--qs", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32, 64, 103])
    parser.add_argument("--matched-only", action="store_true", help="Evaluate Recall only for fixed ARI-best head pairs")
    args = parser.parse_args()
    root = args.input_dir
    metadata = json.loads((root / "metadata.json").read_text())
    n, k = metadata["values_per_head"], round(.01*metadata["values_per_head"])
    layers = metadata["global_layer_indices"]
    qs = sorted(set(args.qs))
    output = root / "neighborhood_recall_1pct"
    output.mkdir(exist_ok=True)
    fits, mean_fits, fit_stats = {}, {}, []
    for layer in layers:
        fits[layer] = []
        for head in range(16):
            labels, medoids, stats = load_fit(root / f"keep_1%/layer_{layer:02d}", f"head_{head:02d}", n, k)
            fits[layer].append((labels, medoids))
            fit_stats.append({"mode": "per_head", "layer": layer, "head": head,
                              "swaps": stats["swaps"], "objective": stats["objective"],
                              "converged": stats["converged"], "seconds": stats["elapsed_seconds"]})
        labels, medoids, stats = load_fit(root / f"head_mean_1pct_ari/layer_{layer:02d}", "clusters", n, k)
        mean_fits[layer] = (labels, medoids)
        fit_stats.append({"mode": "mean_value", "layer": layer, "head": -1,
                          "swaps": stats["swaps"], "objective": stats["objective"],
                          "converged": stats["converged"], "seconds": stats.get("elapsed_seconds", 0)})
    all_rows, matching, summary = [], [], []
    with open_values(root, layers) as cache:
        for layer in layers:
            raw = cache[f"layer_{layer}"]
            for head in range(16):
                check = verify_distances(raw[head], fits[layer][head],
                                         root / f"keep_1%/layer_{layer:02d}", f"head_{head:02d}")
                next(r for r in fit_stats if r["mode"] == "per_head" and r["layer"] == layer and r["head"] == head).update(check)
            check = verify_distances(raw.mean(0, dtype=np.float32), mean_fits[layer],
                                     root / f"head_mean_1pct_ari/layer_{layer:02d}", "clusters")
            next(r for r in fit_stats if r["mode"] == "mean_value" and r["layer"] == layer).update(check)
            print(f"FP64 assignment verification complete: layer {layer+1}", flush=True)
        for source, target in zip(layers[:-1], layers[1:]):
            values = cache[f"layer_{source}"]
            ari = np.array([[adjusted_rand_score(a[0], b[0]) for b in fits[target]] for a in fits[source]])
            for h, (source_labels, source_medoids) in enumerate(fits[source]):
                best = int(ari[h].argmax())
                matching.append({"source_layer": source, "target_layer": target,
                                 "source_head": h, "target_head": best, "ari": float(ari[h, best])})
                for j, (target_labels, target_medoids) in enumerate(fits[target]):
                    if args.matched_only and j != best:
                        continue
                    rows = neighborhood_recall(values[h], source_labels, source_medoids,
                                               target_labels, target_medoids, qs)
                    for row in rows:
                        all_rows.append({"source_layer": source, "target_layer": target,
                                         "source_head": h, "target_head": j,
                                         "ari": float(ari[h, j]), "ari_best_match": j == best, **row})
            for q in qs:
                for mode in (("matched_head_mean",) if args.matched_only else ("matched_head_mean", "same_head_mean", "all_head_pairs_mean")):
                    rows = [r for r in all_rows if r["source_layer"] == source and r["q"] == q
                            and (mode == "all_head_pairs_mean" or
                                 (r["ari_best_match"] if mode == "matched_head_mean" else r["source_head"] == r["target_head"]))]
                    summary.append({"source_layer": source, "target_layer": target, "mode": mode,
                                    "q": q, "recall": float(np.mean([r["recall"] for r in rows])),
                                    "candidate_token_fraction": float(np.mean([r["candidate_token_fraction"] for r in rows])),
                                    "independent_assignment_recall": float(np.mean([r["independent_assignment_recall"] for r in rows])),
                                    "ari": float(np.mean([r["ari"] for r in rows]))})
            a, b = mean_fits[source], mean_fits[target]
            mean_ari = adjusted_rand_score(a[0], b[0])
            for row in neighborhood_recall(values.mean(0, dtype=np.float32), *a, *b, qs):
                summary.append({"source_layer": source, "target_layer": target, "mode": "mean_value",
                                **{key: row[key] for key in ("q", "recall", "candidate_token_fraction", "independent_assignment_recall")},
                                "ari": float(mean_ari)})
            print(f"Recall complete: global layer {source+1} -> {target+1}", flush=True)
    matched_rows = [r for r in all_rows if r["ari_best_match"]]
    if not args.matched_only:
        write_csv(output / "all_head_pairs.csv", all_rows)
    write_csv(output / "matched_heads.csv", matched_rows)
    write_csv(output / "head_matching.csv", matching)
    write_csv(output / "layer_summary.csv", summary)
    write_csv(output / "clustering_checks.csv", fit_stats)
    report = {"input": metadata, "k": k, "qs": qs,
              "validation": {"fits": len(fit_stats), "all_selected_fits_converged": True,
                             "fp64_argmin_disagreements": sum(r["argmin_disagreements_fp64"] for r in fit_stats),
                             "max_assignment_regret_fp64": max(r["max_assignment_regret_fp64"] for r in fit_stats)},
              "head_matching": "For each source head, target head maximizing full-partition ARI; many-to-one allowed; fixed across q; same input used for matching and evaluation",
              "neighbors": "q nearest other source medoids in raw source V Euclidean space, plus source cluster itself; directed kNN",
              "recall": "Fraction of tokens whose target-layer medoid token belonged to the source token's cluster or one of its q source-space neighboring clusters",
              "mean_value": "Arithmetic mean of 16 raw 64D heads at each token BEFORE independent per-layer PAM clustering",
              "optimisation": metadata.get("algorithm", "Best of BUILD and seeded sampling; exhaustive best-swap PAM"),
              "global_optimum_guaranteed": False,
              "candidate_token_fraction": "Mean source cluster-union size over tokens, divided by N; not the fraction of medoids compared",
              "independent_assignment_recall": "Expected recall after independently permuting the target representative assignment across token positions, preserving representative multiplicities",
              "summary": summary}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    make_plot(summary, output / "recall_curves.png")
    lines = [f"# VGGT 前{len(layers)}层 Neighborhood Recall，{metadata['num_frames']}帧、1%聚类", "",
             "逐头：每个源头选择下一层 ARI 最大的目标头（允许多对一），固定此配对后计算所有 q。逐层：先平均16头原始V，再独立聚类。",
             "", f"输入 {metadata['scene']}/{metadata['sequence']}，第{metadata['frame_indices'][0]}至{metadata['frame_indices'][-1]}帧，共{metadata['num_frames']}帧；每头{n}个patch，{k}类；排除special tokens。PAM多初始化选目标较低解；为单交换局部最优，不保证全局最优。",
             "", "邻居由上一层medoid的原始V欧氏距离排序，q不含自身簇。命中检查下一层代表token的上一层归属，不对齐簇标签。这里候选是簇内所有token，并非仅q+1个旧medoid。",
             "", "| 相邻global层 | 模式 | " + " | ".join(f"R@{q}" for q in qs) + " |", "|---|---|" + "---:|"*len(qs)]
    for source in layers[:-1]:
        for mode, label in (("matched_head_mean", "逐头最优匹配后平均"), ("mean_value", "先平均V再聚类")):
            by_q = {r["q"]: r for r in summary if r["source_layer"] == source and r["mode"] == mode}
            lines.append(f"| {source+1}→{source+2} | {label} | " + " | ".join(f"{100*by_q[q]['recall']:.2f}%" for q in qs) + " |")
    lines += ["", "选定头对见 head_matching.csv；逐头各q见 matched_heads.csv；汇总见 layer_summary.csv。",
              "", f"匹配与评估使用同一输入，仅说明本序列上的关系；不同层头坐标并无天然对齐，头平均为指定的算术平均操作。q={k-1}覆盖所有簇，Recall必为100%，是实现检查。",
              "", "![Recall曲线](recall_curves.png)"]
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
