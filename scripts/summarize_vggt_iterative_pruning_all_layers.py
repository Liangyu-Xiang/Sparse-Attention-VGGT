#!/usr/bin/env python3
"""Summarize one-sequence iterative-pruning diagnostics across all global layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.input_dir / "all_layers_step5_summary.json"
    output_markdown = args.output_markdown or args.input_dir / "all_layers_step5_summary.md"
    paths = sorted(
        args.input_dir.glob("layer_*_iterative_pruning_summary.json"),
        key=lambda path: int(path.name.split("_")[1]),
    )
    if not paths:
        raise FileNotFoundError(f"No iterative-pruning layer summaries in {args.input_dir}")
    summaries = [json.loads(path.read_text()) for path in paths]
    reference = summaries[0]
    for summary in summaries[1:]:
        for key in ("scene", "sequence", "frame_indices", "patch_grid", "prune_schedule"):
            if summary[key] != reference[key]:
                raise ValueError(f"{key} differs between layer summaries")
    layers = []
    for summary in summaries:
        per_head = [entry["final_nmse_to_dense"] for entry in summary["per_head_final"]]
        layers.append(
            {
                "layer": summary["layer"],
                "mean_final_nmse_to_dense": summary["mean_final_nmse_to_dense"],
                "median_head_final_nmse_to_dense": float(np.median(per_head)),
                "max_head_final_nmse_to_dense": float(np.max(per_head)),
            }
        )
    result = {
        "method": reference["method"],
        "method_note": reference["method_note"],
        "scene": reference["scene"],
        "sequence": reference["sequence"],
        "frame_indices": reference["frame_indices"],
        "patch_grid": reference["patch_grid"],
        "prune_schedule": reference["prune_schedule"],
        "prune_stage_retentions": reference["prune_stage_retentions"],
        "layers": layers,
    }
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# All-layer iterative-pruning diagnostic",
        "",
        f"Input: `{reference['scene']}/{reference['sequence']}`, frames "
        f"{', '.join(map(str, reference['frame_indices']))}.",
        "",
        "Each iteration removes at most 5% of the currently retained patch keys, "
        "until 25.68% remain.  Special keys remain dense.",
        "",
        "| Global layer | Mean final NMSE | Median head NMSE | Worst head NMSE |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for layer in layers:
        lines.append(
            f"| {layer['layer']} | {layer['mean_final_nmse_to_dense']:.6f} | "
            f"{layer['median_head_final_nmse_to_dense']:.6f} | "
            f"{layer['max_head_final_nmse_to_dense']:.6f} |"
        )
    lines.append("")
    output_markdown.write_text("\n".join(lines))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_markdown}")


if __name__ == "__main__":
    main()
