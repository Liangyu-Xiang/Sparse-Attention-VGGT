#!/usr/bin/env python3
"""Convert dense sparse-pattern diagnostics into an executable 2x2-mask config.

The profiler evaluates both executable static masks and dense-result-dependent
oracle supports.  This tool makes that distinction explicit: the emitted
configuration contains the best *static* 2x2 corner per attention head, while
retaining oracle measurements only as a calibration diagnostic.
"""

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
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory produced by find_vggt_dense_sparse_patterns.py.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Defaults to selected_static_2x2_patterns.json in --input-dir.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Defaults to selected_static_2x2_patterns.md in --input-dir.",
    )
    return parser.parse_args()


def corner_description(corner: int) -> dict[str, int]:
    return {
        "corner": corner,
        "row_offset_in_2x2": corner // 2,
        "column_offset_in_2x2": corner % 2,
    }


def static_corner(candidate: str) -> int:
    if not candidate.startswith(STATIC_PREFIX):
        raise ValueError(f"Not a static 2x2 candidate: {candidate}")
    return int(candidate.removeprefix(STATIC_PREFIX))


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    for row in rows:
        row["layer"] = int(row["layer"])
        row["head"] = int(row["head"])
        row["nmse"] = float(row["nmse"])
        row["patch_keys_per_query"] = int(row["patch_keys_per_query"])
        row["patch_key_retention"] = float(row["patch_key_retention"])
        row["special_keys_always_dense"] = int(row["special_keys_always_dense"])
    return rows


def make_layer_summary(
    metadata: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_head: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_head[row["head"]].append(row)
    if set(by_head) != set(range(metadata["heads"])):
        raise ValueError(f"Incomplete head coverage for layer {metadata['layer']}")

    head_patterns = []
    groups: dict[int, list[int]] = defaultdict(list)
    static_nmses = []
    oracle_nmses = []
    oracle_only_heads = []
    for head in sorted(by_head):
        head_rows = by_head[head]
        dense_best = min(head_rows, key=lambda row: row["nmse"])
        static_best = min(
            (row for row in head_rows if row["candidate"].startswith(STATIC_PREFIX)),
            key=lambda row: row["nmse"],
        )
        corner = static_corner(static_best["candidate"])
        groups[corner].append(head)
        static_nmses.append(static_best["nmse"])
        oracle_nmses.append(dense_best["nmse"])
        if dense_best["candidate"] != static_best["candidate"]:
            oracle_only_heads.append(head)
        head_patterns.append(
            {
                "head": head,
                "static_2x2_mask": corner_description(corner),
                "static_nmse": static_best["nmse"],
                "dense_calibration_best_candidate": dense_best["candidate"],
                "dense_calibration_best_nmse": dense_best["nmse"],
                "static_minus_dense_calibration_nmse": static_best["nmse"] - dense_best["nmse"],
            }
        )

    static_candidate_means = {
        candidate: score
        for candidate, score in metadata["candidate_mean_nmse"].items()
        if candidate.startswith(STATIC_PREFIX)
    }
    global_candidate = min(static_candidate_means, key=static_candidate_means.get)
    layer = metadata["layer"]
    return {
        "layer": layer,
        "global_single_static_2x2_mask": {
            **corner_description(static_corner(global_candidate)),
            "mean_nmse": static_candidate_means[global_candidate],
        },
        "headwise_static_2x2": {
            "mean_nmse": sum(static_nmses) / len(static_nmses),
            "heads_by_corner": {
                str(corner): heads for corner, heads in sorted(groups.items())
            },
            "per_head": head_patterns,
        },
        "dense_oracle_comparison": {
            "mean_best_nmse": sum(oracle_nmses) / len(oracle_nmses),
            "heads_where_dense_dependent_candidate_wins": oracle_only_heads,
            "note": (
                "These candidates require dense attention/output and are not usable "
                "as an acceleration-time support selector."
            ),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Dense-calibrated static 2x2 sparse patterns",
        "",
        "## Protocol",
        "",
        f"- Input: `{summary['scene']}/{summary['sequence']}`, frames "
        f"{', '.join(map(str, summary['frame_indices']))}.",
        f"- Patch grid: {summary['patch_grid'][0]} × {summary['patch_grid'][1]}; "
        f"{summary['patch_keys_per_query']} patch keys retained per query "
        f"({summary['patch_key_retention']:.4%}).",
        f"- All {summary['special_keys_always_dense']} special keys remain dense. "
        "Only patch-query outputs are scored against their exact dense counterparts.",
        "- Corner `k` means row offset `k // 2` and column offset `k % 2` in "
        "each 2×2 patch cell (0=even/even, 1=even/odd, 2=odd/even, 3=odd/odd).",
        "",
        "## Selected executable pattern",
        "",
        "| Layer | Best one-mask corner | One-mask mean NMSE | Best per-head corners | Per-head mean NMSE | Dense-only winner heads |",
        "| ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for layer in summary["layers"]:
        global_mask = layer["global_single_static_2x2_mask"]
        groups = layer["headwise_static_2x2"]["heads_by_corner"]
        groups_text = "; ".join(
            f"c{corner}: {','.join(map(str, heads))}" for corner, heads in groups.items()
        )
        dense_only = layer["dense_oracle_comparison"]["heads_where_dense_dependent_candidate_wins"]
        dense_only_text = ", ".join(map(str, dense_only)) if dense_only else "none"
        lines.append(
            f"| {layer['layer']} | {global_mask['corner']} | {global_mask['mean_nmse']:.6f} | "
            f"{groups_text} | {layer['headwise_static_2x2']['mean_nmse']:.6f} | "
            f"{dense_only_text} |"
        )
    lines.extend(
        [
            "",
            "The JSON companion is the deployable per-head configuration.  It deliberately "
            "uses the best static 2×2 mask even where a dense-result-dependent oracle was "
            "slightly better in calibration.  This avoids hiding dense QK/output computation "
            "inside a purported sparse inference path.",
            "",
            "## Scope",
            "",
            "This is a first-stage calibration on one 10-frame Chess sequence, not a claim of "
            "cross-scene generalization.  The next validation should apply this fixed config to "
            "held-out scenes and measure end-to-end pose/reconstruction quality.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_json = args.output_json or input_dir / "selected_static_2x2_patterns.json"
    output_markdown = args.output_markdown or input_dir / "selected_static_2x2_patterns.md"
    metadata_paths = sorted(input_dir.glob("layer_*_optimal_patterns.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No layer_*_optimal_patterns.json files in {input_dir}")

    layer_summaries = []
    reference: dict[str, Any] | None = None
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        csv_path = input_dir / f"layer_{metadata['layer']:02d}_pattern_metrics.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Metrics file missing for {metadata_path}: {csv_path}")
        if reference is None:
            reference = metadata
        else:
            for key in ("scene", "sequence", "frame_indices", "patch_grid"):
                if metadata[key] != reference[key]:
                    raise ValueError(f"{key} disagrees between layer profiles")
        layer_summaries.append(make_layer_summary(metadata, read_rows(csv_path)))

    assert reference is not None
    first_row = read_rows(input_dir / f"layer_{reference['layer']:02d}_pattern_metrics.csv")[0]
    summary = {
        "format": "vggt_static_2x2_sparse_pattern_v1",
        "purpose": (
            "Executable, per-head 2x2 patch-key masks selected using dense attention calibration."
        ),
        "scene": reference["scene"],
        "sequence": reference["sequence"],
        "frame_indices": reference["frame_indices"],
        "patch_grid": reference["patch_grid"],
        "patch_keys_per_query": first_row["patch_keys_per_query"],
        "patch_key_retention": first_row["patch_key_retention"],
        "special_keys_always_dense": first_row["special_keys_always_dense"],
        "layers": layer_summaries,
    }
    output_json.write_text(json.dumps(summary, indent=2) + "\n")
    output_markdown.write_text(render_markdown(summary))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_markdown}")


if __name__ == "__main__":
    main()
