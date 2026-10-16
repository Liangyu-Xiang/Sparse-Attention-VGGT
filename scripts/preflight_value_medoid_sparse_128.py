"""Read-only dataset/GPU preflight; never allocates PAM's quadratic matrix."""

import argparse
import csv
import datetime
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saf3r.utils.data_utils import load_and_process_images_vggt


def distance_storage(frames, patch_height, patch_width, retention=0.25):
    n = frames * patch_height * patch_width
    return {
        "patch_tokens": n,
        "retained_patch_keys": round(n * retention),
        "special_tokens": frames * 5,
        "distance_elements": n * n,
        "distance_bytes_fp32": n * n * 4,
        "distance_gib_fp32": n * n * 4 / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "workspace/benchmark_dataset/7scenes/7Scenes")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/value_medoid_sparse_128/preflight")
    args = parser.parse_args()
    raw_gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    gpus = [{"index": int(i), "name": name.strip(),
             "total_mib": int(total), "free_mib": int(free)}
            for i, name, total, free in csv.reader(io.StringIO(raw_gpu))]
    scenes = []
    for scene in ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]:
        directory = args.data_root / scene / ("seq-02" if scene == "stairs" else "seq-01")
        candidates = []
        for index in range(500 if scene == "stairs" else 1000):
            image = directory / f"frame-{index:06d}.color.png"
            pose = directory / f"frame-{index:06d}.pose.txt"
            if image.is_file() and pose.is_file():
                candidates.append(image)
        if len(candidates) < 128:
            raise RuntimeError(f"{scene}: fewer than 128 eligible frames")
        selected = [candidates[i] for i in np.linspace(0, len(candidates)-1, 128, dtype=np.int64)]
        representatives = {}
        for image in selected:
            with Image.open(image) as pil:
                representatives.setdefault(pil.size, image)
        # Apply the real preprocessing to each unique input size. The loader
        # pads mixed shapes to the largest height/width, without changing N.
        shapes = [tuple(load_and_process_images_vggt([str(p)]).shape[-2:])
                  for p in representatives.values()]
        height, width = max(s[0] for s in shapes), max(s[1] for s in shapes)
        assert height % 14 == width % 14 == 0
        estimate = distance_storage(128, height // 14, width // 14)
        scenes.append({"scene": scene, "frames": 128,
                       "original_sizes_wh": list(representatives),
                       "processed_size_hw": [height, width],
                       "selected_images": list(map(str, selected)), **estimate})
    exceeds = all(s["distance_bytes_fp32"] > max(g["total_mib"] for g in gpus) * 2**20
                  for s in scenes)
    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "stopped_memory_preflight" if exceeds else "requires_further_peak_memory_check",
        "configuration": {"global_layers_1based": list(range(1, 10)),
                          "retention": 0.25, "selector": "existing full FP32 PAM",
                          "value_representation": "mean of 16 raw V heads before clustering",
                          "key_set_shared_across_heads": True,
                          "special_tokens": "always dense", "all_queries_retained": True},
        "gpus": gpus, "scenes": scenes,
        "model_forward_executed": False, "sparse_attention_executed": False,
        "actual_oom_triggered": False, "evaluation_metrics": None,
        "memory_note": "Only one shared FP32 NxN distance matrix; excludes model and PAM scratch. No 16-head multiplier. Existing PAM does not shard this matrix across GPUs.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 128帧代表性Value稀疏注意力：显存预检", "",
             "按照用户要求：不改PAM算法，先平均16个头的V，共享25%的patch Key集合；预计超显存则停止。", "",
             "预期在前9个全局层应用该选择；保留全部Query，special tokens保持稠密。本次仅做预检。", "",
             "| 场景 | 帧数 | 输入高×宽 | patch数N | 保留Key数 | 单个FP32距离矩阵GiB |",
             "|---|---:|---|---:|---:|---:|"]
    for s in scenes:
        h, w = s["processed_size_hw"]
        lines.append(f"| {s['scene']} | 128 | {h}×{w} | {s['patch_tokens']} | {s['retained_patch_keys']} | {s['distance_gib_fp32']:.3f} |")
    lines += ["", f"所有GPU中最大单卡总显存：{max(g['total_mib'] for g in gpus)/1024:.3f} GiB。",
              "", "结论：显存预检未通过，已主动中止。" if exceeds else "结论：仍需进一步检查峰值显存。",
              "", "上述距离矩阵大小仅为下界，未计模型及PAM临时矩阵。头平均只将特征降为一套表示，不减少N；25%决定medoid数量，不减少全体Value的两两距离矩阵。",
              "", "现有PAM没有跨GPU切分距离矩阵，多张空闲GPU的显存不会自动合并。未改用近似算法、降低帧数或缩小输入。",
              "", "没有加载模型、执行稀疏推理或触发真实OOM，因此没有精度及耗时结果。完整抽帧清单、GPU容量和字节数见report.json。"]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.md"),
                      "patch_tokens": scenes[0]["patch_tokens"],
                      "distance_gib": scenes[0]["distance_gib_fp32"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
