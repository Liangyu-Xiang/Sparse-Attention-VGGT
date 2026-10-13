#!/usr/bin/env python3
"""Compare per-head iterative-pruning fidelity with uniform 2x2 sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterative-dir", type=Path, required=True)
    parser.add_argument("--uniform-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def layer_paths(root: Path, suffix: str) -> dict[int, Path]:
    result = {}
    for path in root.glob(f"layer_*_{suffix}"):
        result[int(path.name.split("_")[1])] = path
    return result


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.iterative_dir / "iterative_vs_uniform_2x2.json"
    output_markdown = args.output_markdown or args.iterative_dir / "iterative_vs_uniform_2x2.md"
    iterative_paths = layer_paths(args.iterative_dir, "iterative_pruning_summary.json")
    uniform_paths = layer_paths(args.uniform_dir, "uniform_2x2_summary.json")
    layers_to_compare = sorted(set(iterative_paths) & set(uniform_paths))
    if not layers_to_compare:
        raise FileNotFoundError("No layer summaries present in both input directories")

    layer_rows: list[dict[str, Any]] = []
    better_rank_modulo: dict[str, list[int]] = {}
    better_best_uniform: dict[str, list[int]] = {}
    all_heads: list[dict[str, Any]] = []
    for layer in layers_to_compare:
        iterative = json.loads(iterative_paths[layer].read_text())
        uniform = json.loads(uniform_paths[layer].read_text())
        for key in ("scene", "sequence", "frame_indices", "patch_grid"):
            if iterative[key] != uniform[key]:
                raise ValueError(f"Layer {layer}: {key} differs between methods")
        iterative_by_head = {entry["head"]: entry for entry in iterative["per_head_final"]}
        uniform_by_head = {entry["head"]: entry for entry in uniform["per_head"]}
        if set(iterative_by_head) != set(uniform_by_head):
            raise ValueError(f"Layer {layer}: head coverage differs between methods")
        heads = []
        for head in sorted(iterative_by_head):
            iterative_nmse = iterative_by_head[head]["final_nmse_to_dense"]
            rank_modulo_nmse = uniform_by_head[head]["head_rank_modulo_four_nmse_to_dense"]
            best_uniform_nmse = uniform_by_head[head]["best_uniform_nmse_to_dense"]
            entry = {
                "layer": layer,
                "head": head,
                "iterative_pruning_nmse": iterative_nmse,
                "head_rank_modulo_four_uniform_nmse": rank_modulo_nmse,
                "best_uniform_2x2_nmse": best_uniform_nmse,
                "iterative_minus_rank_modulo_uniform": iterative_nmse - rank_modulo_nmse,
                "iterative_minus_best_uniform": iterative_nmse - best_uniform_nmse,
                "iterative_beats_rank_modulo_uniform": iterative_nmse < rank_modulo_nmse,
                "iterative_beats_best_uniform": iterative_nmse < best_uniform_nmse,
            }
            heads.append(entry)
            all_heads.append(entry)
        rank_better = [entry["head"] for entry in heads if entry["iterative_beats_rank_modulo_uniform"]]
        best_better = [entry["head"] for entry in heads if entry["iterative_beats_best_uniform"]]
        better_rank_modulo[str(layer)] = rank_better
        better_best_uniform[str(layer)] = best_better
        layer_rows.append(
            {
                "layer": layer,
                "mean_iterative_pruning_nmse": float(np.mean([entry["iterative_pruning_nmse"] for entry in heads])),
                "mean_rank_modulo_uniform_nmse": float(np.mean([entry["head_rank_modulo_four_uniform_nmse"] for entry in heads])),
                "mean_best_uniform_nmse": float(np.mean([entry["best_uniform_2x2_nmse"] for entry in heads])),
                "heads_iterative_beats_rank_modulo_uniform": rank_better,
                "heads_iterative_beats_best_uniform": best_better,
            }
        )

    result = {
        "input": {
            "scene": json.loads(iterative_paths[layers_to_compare[0]].read_text())["scene"],
            "sequence": json.loads(iterative_paths[layers_to_compare[0]].read_text())["sequence"],
            "layers": layers_to_compare,
        },
        "comparison": {
            "iterative": "5%-of-current output-preserving greedy pruning",
            "uniform_rank_modulo": "uniform 2x2 corner = head_id modulo 4",
            "best_uniform": "lowest-NMSE corner among four static uniform 2x2 phases",
        },
        "layers": layer_rows,
        "heads_iterative_beats_rank_modulo_uniform": better_rank_modulo,
        "heads_iterative_beats_best_uniform": better_best_uniform,
        "all_heads": all_heads,
    }
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Five-percent iterative pruning vs. uniform 2x2 sampling",
        "",
        "A smaller NMSE is better. `rank-mod-4` is the previous executable uniform baseline; "
        "`best-uniform` is the best of its four static phases per head and is a stricter comparator.",
        "",
        "| Layer | Iterative mean | Rank-mod-4 mean | Best-uniform mean | Iterative beats rank-mod-4 | Iterative beats best-uniform |",
        "| ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for layer in layer_rows:
        rank_heads = ",".join(map(str, layer["heads_iterative_beats_rank_modulo_uniform"])) or "none"
        best_heads = ",".join(map(str, layer["heads_iterative_beats_best_uniform"])) or "none"
        lines.append(
            f"| {layer['layer']} | {layer['mean_iterative_pruning_nmse']:.6f} | "
            f"{layer['mean_rank_modulo_uniform_nmse']:.6f} | {layer['mean_best_uniform_nmse']:.6f} | "
            f"{rank_heads} | {best_heads} |"
        )
    lines.append("")
    output_markdown.write_text("\n".join(lines))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_markdown}")


if __name__ == "__main__":
    main()
