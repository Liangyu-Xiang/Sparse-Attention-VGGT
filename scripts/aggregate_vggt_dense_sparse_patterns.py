#!/usr/bin/env python3
"""Aggregate per-scene dense sparse-pattern profiles into static 2x2 masks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


STATIC_PREFIX = "uniform_2x2_corner_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def corner(candidate: str) -> int:
    if not candidate.startswith(STATIC_PREFIX):
        raise ValueError(f"Expected static 2x2 candidate, got {candidate}")
    return int(candidate.removeprefix(STATIC_PREFIX))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise ValueError(f"No metrics in {path}")
    for row in result:
        row["layer"] = int(row["layer"])
        row["head"] = int(row["head"])
        row["nmse"] = float(row["nmse"])
        row["patch_keys_per_query"] = int(row["patch_keys_per_query"])
        row["patch_key_retention"] = float(row["patch_key_retention"])
        row["special_keys_always_dense"] = int(row["special_keys_always_dense"])
    return result


def average(rows: list[dict[str, Any]]) -> dict[tuple[int, int, str], float]:
    values: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        values[(row["layer"], row["head"], row["candidate"])].append(row["nmse"])
    return {key: sum(items) / len(items) for key, items in values.items()}


def summarize_layer(
    layer: int,
    scores: dict[tuple[int, int, str], float],
    heads: int,
) -> dict[str, Any]:
    static_by_head: list[tuple[int, str, float]] = []
    best_by_head: list[tuple[int, str, float]] = []
    candidates = sorted({candidate for (row_layer, _head, candidate) in scores if row_layer == layer})
    for head in range(heads):
        choices = [(candidate, scores[(layer, head, candidate)]) for candidate in candidates]
        best_by_head.append((head, *min(choices, key=lambda item: item[1])))
        static_by_head.append(
            (head, *min((item for item in choices if item[0].startswith(STATIC_PREFIX)), key=lambda item: item[1]))
        )
    static_candidates = [candidate for candidate in candidates if candidate.startswith(STATIC_PREFIX)]
    one_mask_candidate = min(
        static_candidates,
        key=lambda candidate: sum(scores[(layer, head, candidate)] for head in range(heads)) / heads,
    )
    modulo_four_baseline = sum(
        scores[(layer, head, f"{STATIC_PREFIX}{head % 4}")]
        for head in range(heads)
    ) / heads
    groups: dict[int, list[int]] = defaultdict(list)
    per_head = []
    dense_only = []
    for (head, static_candidate, static_nmse), (_, dense_candidate, dense_nmse) in zip(static_by_head, best_by_head):
        static_corner = corner(static_candidate)
        groups[static_corner].append(head)
        if static_candidate != dense_candidate:
            dense_only.append(head)
        per_head.append(
            {
                "head": head,
                "corner": static_corner,
                "row_offset_in_2x2": static_corner // 2,
                "column_offset_in_2x2": static_corner % 2,
                "cross_scene_mean_nmse": static_nmse,
                "dense_calibration_best_candidate": dense_candidate,
                "dense_calibration_best_mean_nmse": dense_nmse,
                "static_minus_dense_calibration_mean_nmse": static_nmse - dense_nmse,
            }
        )
    return {
        "layer": layer,
        "global_single_static_2x2_mask": {
            "corner": corner(one_mask_candidate),
            "row_offset_in_2x2": corner(one_mask_candidate) // 2,
            "column_offset_in_2x2": corner(one_mask_candidate) % 2,
            "cross_scene_mean_nmse": sum(
                scores[(layer, head, one_mask_candidate)] for head in range(heads)
            ) / heads,
        },
        "head_rank_modulo_four_baseline_mean_nmse": modulo_four_baseline,
        "headwise_static_2x2": {
            "cross_scene_mean_nmse": sum(item[2] for item in static_by_head) / heads,
            "heads_by_corner": {str(key): value for key, value in sorted(groups.items())},
            "per_head": per_head,
        },
        "dense_oracle_comparison": {
            "cross_scene_mean_best_nmse": sum(item[2] for item in best_by_head) / heads,
            "heads_where_dense_dependent_candidate_wins": dense_only,
            "note": "Dense-result-dependent winners cannot be used directly for sparse inference.",
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-scene dense-calibrated static 2x2 sparse patterns",
        "",
        "The pattern was selected by averaging exact sparse-vs-dense output NMSE "
        f"over {len(summary['calibration_sequences'])} 7Scenes sequences.  Each sequence uses "
        "ten frames (0, 5, …, 45), retains one patch key per 2×2 cell (~25.7%), and keeps "
        "all special keys dense.",
        "",
        "Corner encoding: 0=even/even, 1=even/odd, 2=odd/even, 3=odd/odd patch row/column offsets.",
        "",
        "| Layer | Best one-mask corner | One-mask mean NMSE | Head-rank-mod-4 baseline | Best per-head corners | Per-head mean NMSE | Dense-only winner heads |",
        "| ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for layer in summary["layers"]:
        global_mask = layer["global_single_static_2x2_mask"]
        groups = "; ".join(
            f"c{key}: {','.join(map(str, value))}"
            for key, value in layer["headwise_static_2x2"]["heads_by_corner"].items()
        )
        dense_only = layer["dense_oracle_comparison"]["heads_where_dense_dependent_candidate_wins"]
        lines.append(
            f"| {layer['layer']} | {global_mask['corner']} | "
            f"{global_mask['cross_scene_mean_nmse']:.6f} | "
            f"{layer['head_rank_modulo_four_baseline_mean_nmse']:.6f} | {groups} | "
            f"{layer['headwise_static_2x2']['cross_scene_mean_nmse']:.6f} | "
            f"{', '.join(map(str, dense_only)) if dense_only else 'none'} |"
        )
    lines.extend(
        [
            "",
            "The matching JSON is the executable per-head configuration.  It selects only "
            "fixed 2×2 masks; dense-only oracle candidates are recorded solely as diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_json = args.output_json or args.input_root / "selected_cross_scene_static_2x2_patterns.json"
    output_markdown = args.output_markdown or args.input_root / "selected_cross_scene_static_2x2_patterns.md"
    metadata_files = sorted(args.input_root.glob("*/layer_00_optimal_patterns.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No scene profiles found under {args.input_root}")

    all_rows = []
    sequences = []
    prototype: dict[str, Any] | None = None
    for layer_zero_path in metadata_files:
        scene_dir = layer_zero_path.parent
        metadata = json.loads(layer_zero_path.read_text())
        if prototype is None:
            prototype = metadata
        else:
            for key in ("frame_indices", "patch_grid", "heads"):
                if metadata[key] != prototype[key]:
                    raise ValueError(f"{key} varies across calibration profiles")
        sequences.append({"scene": metadata["scene"], "sequence": metadata["sequence"]})
        for layer in range(3):
            csv_path = scene_dir / f"layer_{layer:02d}_pattern_metrics.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Incomplete profile for {scene_dir}: missing {csv_path.name}")
            all_rows.extend(read_csv(csv_path))

    assert prototype is not None
    scores = average(all_rows)
    summary = {
        "format": "vggt_cross_scene_static_2x2_sparse_pattern_v1",
        "purpose": "Executable 2x2 patch-key masks selected by cross-scene dense calibration.",
        "calibration_sequences": sequences,
        "frame_indices": prototype["frame_indices"],
        "patch_grid": prototype["patch_grid"],
        "patch_keys_per_query": all_rows[0]["patch_keys_per_query"],
        "patch_key_retention": all_rows[0]["patch_key_retention"],
        "special_keys_always_dense": all_rows[0]["special_keys_always_dense"],
        "layers": [summarize_layer(layer, scores, prototype["heads"]) for layer in range(3)],
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    output_markdown.write_text(markdown(summary))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_markdown}")


if __name__ == "__main__":
    main()
