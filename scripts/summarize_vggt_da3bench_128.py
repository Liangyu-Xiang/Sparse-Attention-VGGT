#!/usr/bin/env python3
"""Combine the three dense-VGGT DA3-Bench shards into dataset-level metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SHARD_SCENE_COUNTS = {
    "gpu0": {"7scenes": 3, "eth3d": 3},
    "gpu1": {"7scenes": 2, "eth3d": 4},
    "gpu2": {"7scenes": 2, "eth3d": 4},
}

# SAF3R Tables 2 and 3, dense VGGT row.  Pose is reported in percent.
PAPER_REFERENCE = {
    "7scenes": {"AUC@30": 84.64, "AUC@3": 23.66, "Acc": 0.155, "Comp": 0.125, "CD": 0.140, "F1": 0.480},
    "eth3d": {"AUC@30": 81.68, "AUC@3": 29.42, "Acc": 0.306, "Comp": 0.831, "CD": 0.568, "F1": 0.634},
}


def read_dataset_averages(log_path: Path) -> dict[str, dict[str, float]]:
    lines = log_path.read_text().splitlines()
    parsed: dict[str, dict[str, float]] = {}
    for index, line in enumerate(lines):
        if line != "Dataset AVG" or index + 2 >= len(lines):
            continue
        names = lines[index + 1].split("\t")
        values = lines[index + 2].split("\t")
        if len(names) != len(values):
            continue

        # The preceding per-scene table states which dataset this summary
        # belongs to.  The full log contains one 7Scenes and one ETH3D block.
        dataset = None
        for previous in reversed(lines[:index]):
            if previous.startswith("Per-scene metrics of dataset ["):
                dataset = previous.removeprefix("Per-scene metrics of dataset [").removesuffix("]")
                break
        if dataset is None:
            continue

        parsed[dataset] = {
            name: float(value)
            for name, value in zip(names, values)
            if value != "-"
        }
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/reproduce_vggt_da3bench_128"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    totals: dict[str, dict[str, float]] = {"7scenes": {}, "eth3d": {}}
    weights: dict[str, int] = {"7scenes": 0, "eth3d": 0}
    missing = []
    for shard, counts in SHARD_SCENE_COUNTS.items():
        log_path = args.log_dir / f"{shard}.log"
        if not log_path.exists():
            missing.append(str(log_path))
            continue
        shard_metrics = read_dataset_averages(log_path)
        for dataset, scene_count in counts.items():
            if dataset not in shard_metrics:
                missing.append(f"{log_path} ({dataset} summary)")
                continue
            weights[dataset] += scene_count
            for name, value in shard_metrics[dataset].items():
                totals[dataset][name] = totals[dataset].get(name, 0.0) + scene_count * value

    if missing:
        raise SystemExit("Cannot summarize incomplete evaluation logs: " + ", ".join(missing))

    results = {
        dataset: {
            # The evaluator emits AUC in [0, 1], whereas SAF3R Table 2
            # reports it as a percentage.
            name: (100.0 if name.startswith("AUC@") else 1.0) * value / weights[dataset]
            for name, value in metrics.items()
        }
        for dataset, metrics in totals.items()
    }
    payload = {"observed": results, "paper_reference": PAPER_REFERENCE}
    output_path = args.output or args.log_dir / "summary.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for dataset in ("7scenes", "eth3d"):
        observed = results[dataset]
        reference = PAPER_REFERENCE[dataset]
        print(dataset)
        for name in ("AUC@30", "AUC@3", "Acc", "Comp", "CD", "F1"):
            print(f"  {name:<7} observed={observed[name]:.4f}  paper={reference[name]:.4f}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
