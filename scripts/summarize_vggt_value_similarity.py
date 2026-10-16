#!/usr/bin/env python3
"""Summarize per-head patch-value cosine similarity across VGGT layers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        args.input_dir.glob("layer_*_value_similarity_summary.json"),
        key=lambda path: int(path.name.split("_")[1]),
    )
    if not paths:
        raise FileNotFoundError(f"No value-similarity summaries in {args.input_dir}")
    summaries = [json.loads(path.read_text()) for path in paths]
    reference = summaries[0]
    for summary in summaries[1:]:
        for key in ("scene", "sequence", "frame_indices", "patch_grid", "method", "definition"):
            if summary[key] != reference[key]:
                raise ValueError(f"{key} differs between layer summaries")
    layers = []
    for summary in summaries:
        per_head = summary["per_head"]
        layers.append(
            {
                "layer": summary["layer"],
                "mean_head_pairwise_cosine": float(
                    np.mean([row["mean_pairwise_cosine"] for row in per_head])
                ),
                "mean_head_pairwise_cosine_std": float(
                    np.mean([row["pairwise_cosine_std"] for row in per_head])
                ),
                "mean_head_value_l2_norm": float(
                    np.mean([row["mean_value_l2_norm"] for row in per_head])
                ),
                "per_head": per_head,
            }
        )
    result = {
        "method": reference["method"],
        "definition": reference["definition"],
        "scene": reference["scene"],
        "sequence": reference["sequence"],
        "frame_indices": reference["frame_indices"],
        "patch_grid": reference["patch_grid"],
        "layers": layers,
    }
    output_json = args.output_json or args.input_dir / "value_similarity_summary.json"
    output_csv = args.output_csv or args.input_dir / "value_similarity_per_head.csv"
    output_markdown = args.output_markdown or args.input_dir / "value_similarity_summary.md"
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    per_head_rows = [row for layer in layers for row in layer["per_head"]]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_head_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(per_head_rows)
    lines = [
        "# Patch-value pairwise cosine similarity",
        "",
        "Each statistic uses all ordered off-diagonal pairs of L2-normalized patch-key V vectors. "
        "Higher mean cosine means more mutually substitutable directions; special-token values are excluded.",
        "",
        "| Layer | Mean pairwise cosine | Mean pairwise cosine std | Mean V L2 norm |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for layer in layers:
        lines.append(
            f"| {layer['layer']} | {layer['mean_head_pairwise_cosine']:.6f} | "
            f"{layer['mean_head_pairwise_cosine_std']:.6f} | "
            f"{layer['mean_head_value_l2_norm']:.6f} |"
        )
    lines.append("")
    output_markdown.write_text("\n".join(lines))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_markdown}")


if __name__ == "__main__":
    main()
