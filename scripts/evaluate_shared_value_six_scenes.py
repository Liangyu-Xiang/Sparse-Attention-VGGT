"""Run paired six-scene quality/efficiency evaluation, all methods on one exclusive GPU.

From repository root on the GPU machine:
    python scripts/evaluate_shared_value_six_scenes.py --gpu 6
    python scripts/evaluate_shared_value_six_scenes.py --gpu 6 --report-only

Exactly three scenes per dataset. Each method samples the same frames, the same
checkpoint, and the same quality protocol. Run order per scene is Dense, SAF3R,
sampled256 (original state loss), shared_value (new state loss).
GPU leasing prevents overlapping CUDA programs during timing.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = {
    "7scenes": ("heads", "chess", "stairs"),
    "eth3d": ("courtyard", "kicker", "playground"),
}
METHODS = {
    "dense": ("dense", None),
    "saf3r": ("saf3r", None),
    "sampled256": ("zdrop_history_states", "configs/evaluation/reuse_sampled_history_256.json"),
    "shared_value": ("zdrop_history_states", "configs/evaluation/reuse_shared_value_loss_256.json"),
}
METRICS = ("auc03", "auc30", "acc", "comp", "cd", "f1")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def aggregate(root):
    final = {}
    for dataset, scenes in SCENES.items():
        summary = {}
        for method in METHODS:
            items = []
            for scene in scenes:
                path = root / dataset / scene / method / "result.json"
                if not path.exists():
                    raise FileNotFoundError(f"Missing paired result: {path}")
                item = json.loads(path.read_text())
                if not item["quality"]["complete"]:
                    raise RuntimeError(f"Incomplete quality: {path}: {item['quality']['unavailable']}")
                items.append(item)
            for scene, models in zip(scenes, [
                [json.loads((root / dataset / scene / m / "result.json").read_text()) for m in METHODS]
                for scene in scenes
            ]):
                ref = models[0]
                for model in models[1:]:
                    if model["input"] != ref["input"] or model["quality_protocol"] != ref["quality_protocol"]:
                        raise ValueError(f"Different frames/checkpoint/protocol: {dataset}/{scene}")
                    if model["hardware"] != ref["hardware"]:
                        raise ValueError(f"Different device/software: {dataset}/{scene}")
                    if any(model["timing"][k] != ref["timing"][k]
                           for k in ("warmup", "repeats", "capture")):
                        raise ValueError(f"Unpaired timing protocol: {dataset}/{scene}")
            summary[method] = dict(
                scenes=len(items),
                names=list(scenes),
                frames=[item["frames"] for item in items],
                metrics={metric: sum(item["quality"]["values"][metric] for item in items) / len(items)
                         for metric in METRICS},
                mean_forward_seconds=sum(item["timing"]["median_seconds"] for item in items) / len(items),
                per_scene_forward_seconds=[item["timing"]["median_seconds"] for item in items],
                peak_allocated_gib=max(
                    max(p["peak_allocated_gib"] for p in item["timing"]["samples"])
                    for item in items
                ),
            )
        baseline = summary["dense"]["mean_forward_seconds"]
        for method in METHODS:
            summary[method]["speedup_vs_paired_dense"] = (
                baseline / summary[method]["mean_forward_seconds"])
        final[dataset] = summary
    dump(root / "summary.json", final)
    lines = [
        "# Paired six-scene evaluation: shared Value state loss",
        "",
        "Three scene macro means, matched frames/checkpoint/protocol.",
        "Timings: one warmup + three forward repeats per scene, exclusive physical GPU.",
        "Mean timing = arithmetic mean of the three per-scene warmed medians.",
        "Speedup uses the same three scenes' mean Dense runtime.",
        "",
        "| Dataset | Method | AUC@3% ↑ | AUC@30% ↑ | ACC m ↓ | Comp m ↓ | CD m ↓ | F1% ↑ | mean forward s ↓ | vs Dense |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset in SCENES:
        for method in METHODS:
            result = final[dataset][method]
            m = result["metrics"]
            lines.append(
                "| " + dataset + " | " + method + " | "
                + " | ".join(f"{m[k] * (100 if k in ('auc03', 'auc30','f1') else 1):.4f}"
                             for k in METRICS)
                + f" | {result['mean_forward_seconds']:.4f} | "
                + f"{result['speedup_vs_paired_dense']:.3f}x |"
            )
    lines += ["", "Note: sampled256 keeps prior compensation; shared_value removes it. "
              "Differences reflect the entire changed state definition, not a pure kernel-only speed comparison.",
              ""]
    (root / "report.md").write_text("\n".join(lines))
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=6)
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/shared_value_loss_six_20261104")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.report_only:
        aggregate(args.output_root)
        print("Report:", args.output_root / "report.md")
        return
    # Run all methods in the same scene, then move to the next scene.
    for dataset, scenes in SCENES.items():
        for scene in scenes:
            dense_path = args.output_root / dataset / scene / "dense" / "forward.json"
            for name, (method, config) in METHODS.items():
                output = args.output_root / dataset / scene / name
                cmd = [
                    sys.executable, "scripts/evaluate_zdrop_history_states.py",
                    "--dataset", dataset, "--scene", scene,
                    "--frames", "128", "--method", method,
                    "--output-dir", str(output),
                    "--warmup", "1", "--repeats", "3",
                ]
                if config:
                    cmd += ["--config", str(ROOT / config)]
                if name != "dense":
                    cmd += ["--baseline", str(dense_path)]
                monitor = args.output_root / "monitors" / f"{dataset}_{scene}_{name}.json"
                log = args.output_root / "logs" / f"{dataset}_{scene}_{name}.log"
                call = [
                    sys.executable, "scripts/run_gpu_lease.py",
                    "--strict-contexts", "--gpu", str(args.gpu),
                    "--monitor", str(monitor), "--log", str(log),
                    "--", *cmd,
                ]
                if (output / "result.json").exists():
                    print("EXISTS (not overwritten):", output, flush=True)
                    continue
                if output.exists():
                    raise RuntimeError(f"Existing incomplete output, resolve manually: {output}")
                print("RUN:", dataset, scene, name, " ".join(call), flush=True)
                if args.dry_run:
                    continue
                subprocess.run(call, check=True, cwd=ROOT)
    if not args.dry_run:
        aggregate(args.output_root)
        print("Complete:", args.output_root / "report.md")


if __name__ == "__main__":
    main()
