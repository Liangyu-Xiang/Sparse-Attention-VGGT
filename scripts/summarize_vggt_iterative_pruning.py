#!/usr/bin/env python3
"""Summarize cross-scene output fidelity of iterative attention pruning."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--static-reference",
        type=Path,
        help="Optional cross-scene static-2x2 summary for same-protocol fidelity comparison.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("layer", "head", "patch_keys_per_query", "removed_keys_per_query"):
            row[key] = int(row[key])
        for key in ("patch_key_retention", "nmse_to_dense"):
            row[key] = float(row[key])
        if row["mean_deleted_key_cost"]:
            row["mean_deleted_key_cost"] = float(row["mean_deleted_key_cost"])
    return rows


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.input_root / "iterative_pruning_cross_scene_summary.json"
    output_markdown = args.output_markdown or args.input_root / "iterative_pruning_cross_scene_summary.md"
    scene_summaries = sorted(args.input_root.glob("*/layer_00_iterative_pruning_summary.json"))
    if not scene_summaries:
        raise FileNotFoundError(f"No iterative-pruning summaries found under {args.input_root}")

    scene_metadata = []
    final_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    stage_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    reference: dict[str, Any] | None = None
    for layer_zero_summary in scene_summaries:
        scene_dir = layer_zero_summary.parent
        layer_zero = json.loads(layer_zero_summary.read_text())
        if reference is None:
            reference = layer_zero
        else:
            for key in ("frame_indices", "patch_grid", "prune_stage_retentions"):
                if layer_zero[key] != reference[key]:
                    raise ValueError(f"{key} differs between scene profiles")
        scene_metadata.append({"scene": layer_zero["scene"], "sequence": layer_zero["sequence"]})
        for layer in range(3):
            summary_path = scene_dir / f"layer_{layer:02d}_iterative_pruning_summary.json"
            metrics_path = scene_dir / f"layer_{layer:02d}_iterative_pruning_metrics.csv"
            if not summary_path.is_file() or not metrics_path.is_file():
                raise FileNotFoundError(f"Incomplete iterative-pruning results in {scene_dir}")
            summary = json.loads(summary_path.read_text())
            for head in summary["per_head_final"]:
                final_scores[(layer, head["head"])].append(head["final_nmse_to_dense"])
            for row in read_csv(metrics_path):
                stage_scores[(layer, row["patch_keys_per_query"])].append(row["nmse_to_dense"])

    assert reference is not None
    layers = []
    for layer in range(3):
        per_head = [
            {
                "head": head,
                "cross_scene_mean_final_nmse_to_dense": float(np.mean(final_scores[(layer, head)])),
                "cross_scene_median_final_nmse_to_dense": float(np.median(final_scores[(layer, head)])),
            }
            for head in range(16)
        ]
        stage_curve = [
            {
                "patch_keys_per_query": count,
                "patch_key_retention": count / (reference["patch_grid"][0] * reference["patch_grid"][1] * len(reference["frame_indices"])),
                "cross_scene_mean_nmse_to_dense": float(np.mean(values)),
            }
            for (score_layer, count), values in sorted(stage_scores.items(), reverse=True)
            if score_layer == layer
        ]
        layers.append(
            {
                "layer": layer,
                "cross_scene_mean_final_nmse_to_dense": float(np.mean([item["cross_scene_mean_final_nmse_to_dense"] for item in per_head])),
                "per_head": per_head,
                "mean_nmse_curve": stage_curve,
            }
        )

    result = {
        "method": "batched_iterative_output_preserving_pruning",
        "method_note": (
            "Each stage recomputes the current attention/output and deletes a batch of keys by "
            "the exact single-key deletion cost. Results are greedy-local, not exhaustive-global."
        ),
        "calibration_sequences": scene_metadata,
        "frame_indices": reference["frame_indices"],
        "patch_grid": reference["patch_grid"],
        "prune_stage_retentions": reference["prune_stage_retentions"],
        "layers": layers,
    }
    if args.static_reference:
        static_summary = json.loads(args.static_reference.read_text())
        result["static_2x2_reference"] = {
            "source": str(args.static_reference),
            "layer_mean_nmse": {
                str(layer["layer"]): layer["headwise_static_2x2"]["cross_scene_mean_nmse"]
                for layer in static_summary["layers"]
            },
        }
    output_json.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Iterative output-preserving pruning: cross-scene fidelity",
        "",
        "Per-query patch keys are greedily removed in batches at 100% → 75% → 50% → 37.5% → 25.68%. "
        "All special keys remain dense; each reported NMSE compares the sparse output with the "
        "exact dense output at the same layer and head.",
        "",
        "| Layer | Iterative-pruning mean NMSE at 25.68% |",
        "| ---: | ---: |",
    ]
    static_reference = result.get("static_2x2_reference", {}).get("layer_mean_nmse", {})
    if static_reference:
        lines[-2] = "| Layer | Iterative-pruning mean NMSE at 25.68% | Static 2×2 reference NMSE |"
        lines[-1] = "| ---: | ---: | ---: |"
    for layer in layers:
        line = f"| {layer['layer']} | {layer['cross_scene_mean_final_nmse_to_dense']:.6f} |"
        if static_reference:
            line += f" {static_reference[str(layer['layer'])]:.6f} |"
        lines.append(line)
    lines.extend(["", "## Mean pruning curves", ""])
    for layer in layers:
        curve = "; ".join(
            f"{entry['patch_key_retention']:.2%}: {entry['cross_scene_mean_nmse_to_dense']:.6f}"
            for entry in layer["mean_nmse_curve"]
        )
        lines.append(f"- Layer {layer['layer']}: {curve}")
    lines.append("")
    output_markdown.write_text("\n".join(lines))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_markdown}")


if __name__ == "__main__":
    main()
