"""Chinese matched-scene report; never treat missing Value-PAM results as zero."""
import argparse
import csv
import fcntl
import json
from pathlib import Path

RECON_KEYS = ("acc", "comp", "overall", "precision", "recall", "fscore")


def complete_reconstruction_means(rows, scenes):
    means = {}
    for method in ("dense", "saf3r", "value_pam"):
        selected = [r for r in rows if r["method"] == method]
        if len(selected) == len(scenes) and {r["scene"] for r in selected} == set(scenes):
            means[method] = {key: sum(r[key] for r in selected)/len(selected) for key in RECON_KEYS}
    return means


def matched_input(expected, actual):
    if len(expected["paths"]) != len(actual["paths"]):
        return False
    return (all(Path(a).resolve() == Path(b).resolve()
                for a, b in zip(expected["paths"], actual["paths"]))
            and expected["shape"] == actual["shape"])


def complete_scene_means(rows, scenes):
    """A method needs every requested scene; no partial-scene aggregate."""
    result = {}
    for method in ("dense", "saf3r", "value_pam"):
        selected = [row for row in rows if row["method"] == method]
        if {r["scene"] for r in selected} != set(scenes) or len(selected) != len(scenes):
            continue
        result[method] = {key: sum(r[key] for r in selected)/len(selected)
                          for key in ("auc03", "auc05", "auc15", "auc30",
                                      "latency_seconds", "peak_allocated_gib")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/compare_value_sparse_50"))
    parser.add_argument("--value-root", type=Path, default=Path("outputs/value_medoid_sparse_50"))
    args = parser.parse_args()
    report_lock = (args.root/".report.lock").open("a")
    fcntl.flock(report_lock,fcntl.LOCK_EX)
    manifest = json.loads((args.root / "inputs.json").read_text())
    names = {"dense": "原始VGGT", "saf3r": "SAF3R", "value_pam": "Value-PAM（前9层）"}
    rows, pending = [], []
    for method, root in (("dense", args.root/"dense"), ("saf3r", args.root/"saf3r"),
                         ("value_pam", args.value_root)):
        for scene, entry in manifest["scenes"].items():
            directory = root / scene
            result_file, input_file = directory / "result.json", directory / "input.json"
            if not result_file.exists():
                pending.append({"method": method, "scene": scene})
                continue
            actual = json.loads(input_file.read_text())
            expected = {"paths": entry["paths"], "shape": [entry["frames"], 3, 392, 518]}
            if not matched_input(expected, actual):
                raise ValueError(f"{method}/{scene}: input mismatch, refusing comparison")
            data = json.loads(result_file.read_text())
            assert data["frames"] == manifest["frames"] and data["scene"] == scene
            rows.append({"method": method, "scene": scene, **data["pose"],
                         "latency_seconds": data["latency_seconds"],
                         "peak_allocated_gib": data["peak_allocated_gib"]})
    means = complete_scene_means(rows, list(manifest["scenes"]))
    reconstruction, reconstruction_pending = [], []
    for method in names:
        for scene,entry in manifest["scenes"].items():
            directory = args.root/"reconstruction"/method/scene
            result = directory/"result.json"
            if not result.is_file():
                reconstruction_pending.append({"method":method,"scene":scene})
                continue
            geometry = json.loads(result.read_text())
            capture = json.loads((directory/"capture.json").read_text())
            expected = {"paths":entry["paths"],"shape":[entry["frames"],3,392,518]}
            if not matched_input(expected,capture) or geometry["frames"] != manifest["frames"]:
                raise ValueError("Reconstruction comparison input mismatch")
            if geometry["threshold_m"] != .05 or geometry["sampled_points"] != 1_000_000:
                raise ValueError("Reconstruction protocol mismatch")
            reconstruction.append({"method":method,"scene":scene,**geometry["recon_unposed"]})
    reconstruction_means = complete_reconstruction_means(reconstruction,list(manifest["scenes"]))
    with (args.root/"comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "scene", "auc03", "auc05", "auc15", "auc30", "latency_seconds", "peak_allocated_gib"))
        writer.writeheader()
        writer.writerows(rows)
    with (args.root/"reconstruction.csv").open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=("method","scene",*RECON_KEYS))
        writer.writeheader()
        writer.writerows(reconstruction)
    (args.root/"comparison.json").write_text(json.dumps({"manifest": manifest,
        "rows": rows, "means_all_requested_scenes": means, "pending": pending,
        "reconstruction_rows":reconstruction,"reconstruction_means":reconstruction_means,
        "reconstruction_pending":reconstruction_pending}, indent=2) + "\n")
    lines = [f"# 7Scenes：{manifest['frames']}帧、相同输入的三方法比较", "",
             "场景：" + "、".join(manifest["scenes"]) + "。相同checkpoint、VGGT crop预处理518×392、uniform抽帧；BF16 autocast、FP32权重、关闭TF32。",
             "", "位姿按仓库DA3-Bench的全部帧对相对旋转/平移角误差计算AUC，下表换算为百分数，越高越好。不是绝对位姿误差或深度/重建指标。", "",
             "| 方法 | 场景 | AUC@3° % | AUC@5° % | AUC@15° % | AUC@30° % | 单次前向秒 | 峰值分配GiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    def row_text(name, scene, row):
        return f"| {name} | {scene} | " + " | ".join(f"{row[k]*100:.2f}" for k in ("auc03", "auc05", "auc15", "auc30")) + f" | {row['latency_seconds']:.3f} | {row['peak_allocated_gib']:.3f} |"
    for row in rows:
        lines.append(row_text(names[row["method"]], row["scene"], row))
    lines += ["", "## 相同三个场景的宏平均", "",
              "| 方法 | 场景 | AUC@3° % | AUC@5° % | AUC@15° % | AUC@30° % | 单次前向秒 | 峰值分配GiB均值 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for method, row in means.items():
        lines.append(row_text(names[method], "三场景平均", row))
    if "dense" in means and "saf3r" in means:
        lines += ["", f"按本次单次前向均值，SAF3R相对原始VGGT耗时比为{means['dense']['latency_seconds']/means['saf3r']['latency_seconds']:.3f}×（包含冷启动/JIT，不能当作稳态kernel加速比）。"]
    lines += ["", "## 点云重建：recon_unposed", "",
              "使用预测位姿及预测深度，经仓库Umeyama/RANSAC尺度与刚体对齐，使用GT深度的有效像素mask；不是GT位姿融合。TSDF体素和评测下采样均为7.8125 mm，SDF截断4 cm，预测/GT网格各采样100万点。三方法共享每场景同一份GT点云，采样固定随机种子42。",
              "", "Accuracy为预测→GT的平均最近邻距离，Completeness为GT→预测；此处CD=(Accuracy+Completeness)/2，为非平方距离的双向均值。三者下表单位cm，越低越好。Precision/Recall/F-score使用5 cm阈值，换算为百分数，越高越好。", "",
              "| 方法 | 场景 | Accuracy↓ cm | Completeness↓ cm | CD↓ cm | Precision↑ % | Recall↑ % | F-score↑ % |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    def recon_row(method,scene,row):
        return f"| {names[method]} | {scene} | " + " | ".join(f"{row[key]*100:.3f}" for key in RECON_KEYS) + " |"
    for row in reconstruction:
        lines.append(recon_row(row["method"],row["scene"],row))
    if not reconstruction:
        lines.append("| 三方法 | 三场景 | 待计算 | 待计算 | 待计算 | 待计算 | 待计算 | 待计算 |")
    if reconstruction_means:
        lines += ["", "重建指标三场景宏平均：", "",
                  "| 方法 | 场景 | Accuracy↓ cm | Completeness↓ cm | CD↓ cm | Precision↑ % | Recall↑ % | F-score↑ % |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for method,row in reconstruction_means.items():
            lines.append(recon_row(method,"三场景平均",row))
    lines += ["", "原推理未持久化深度，因此重建需补做深度前向；Value-PAM只重放已经完成的9层medoid集合，不重新聚类，并核对输出位姿与原实验一致。重建捕获/CPU融合耗时独立统计，不替换上表在线方法耗时。无空闲GPU时排队，不抢占现有PAM进程。空网格/采样失败直接报告失败，禁止随机点云兜底。"]
    lines += ["", "## 统计口径与限制", "",
              "- 每场景单次在线前向，无额外warmup；CUDA前后同步。排除模型加载、图像读取/预处理、传输和位姿打分，包含前向内选点及首次JIT。不同GPU均为同型号RTX 4090；并行运行仍可能存在共享主机资源干扰。",
              "- 峰值是PyTorch max_memory_allocated，含常驻权重，不是nvidia-smi整卡占用；表中均值不是三个场景并发显存。",
              "- SAF3R使用原仓库eth3d-train-fltr-calib_cmpmse profile和lazy QK Top-K=4/frame，在全部24个global层应用原配置；不人为改成前9层或25%保留率。",
              "- 当前环境SAF3R使用已有blockwise PyTorch per-frame QK Top-K兼容路径，其attention仍使用仓库Triton实现；这里测得的是该兼容实现性能，不是论文原始融合选点kernel的速度。",
              "- Value-PAM只替换前9个global层，25%保留率；在线PAM、逐层完整规模显存冒烟检查及簇结果写盘均在现有原型前向计时中，不能将该耗时当作纯稀疏attention耗时。",
              "- 未完成的场景不记零、不参与三场景均值；不混用旧128帧结果。"]
    if pending:
        lines += ["", "尚缺结果：" + "、".join(f"{names[p['method']]}/{p['scene']}" for p in pending) + "。"]
    if reconstruction_pending:
        lines += ["", "尚缺重建结果：" + "、".join(f"{names[p['method']]}/{p['scene']}" for p in reconstruction_pending) + "。"]
    (args.root/"report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed_rows": len(rows), "mean_methods": list(means), "pending": pending}, ensure_ascii=False))


if __name__ == "__main__":
    main()
