#!/usr/bin/env python3
"""Aggregate multi-scene sparse-support transfer diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MANUAL_CHAIN = [(10, 3), (11, 0), (12, 8), (13, 10), (14, 10), (15, 8), (16, 2)]
DIRECT_REUSE_PROBES = [
    ((10, 3), (12, 8)),
    ((10, 3), (14, 10)),
    ((12, 8), (16, 2)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def score(candidate: np.ndarray, self_error: np.ndarray, uniform: np.ndarray) -> dict[str, float]:
    ratio = candidate / (self_error + 1.0e-8)
    beats_uniform = candidate <= uniform
    within_self = ratio <= 1.10
    valid = uniform > self_error + 1.0e-8
    normalized = (candidate - self_error) / (uniform - self_error + 1.0e-8)
    valid_normalized = normalized[valid]
    result = {
        "mean_transfer_error": float(candidate.mean()),
        "median_transfer_error": float(np.median(candidate)),
        "median_transfer_to_self_error_ratio": float(np.median(ratio)),
        "fraction_within_10pct_of_self": float(within_self.mean()),
        "fraction_beats_uniform": float(beats_uniform.mean()),
        "fraction_both": float((within_self & beats_uniform).mean()),
        "fraction_valid_normalization": float(valid.mean()),
    }
    if valid_normalized.size:
        result.update({
            "mean_normalized_regret_valid_only": float(valid_normalized.mean()),
            "median_normalized_regret_valid_only": float(np.median(valid_normalized)),
            "fraction_normalized_regret_le_0.10_valid_only": float(
                (valid_normalized <= 0.10).mean()
            ),
            "fraction_normalized_regret_le_0.25_valid_only": float(
                (valid_normalized <= 0.25).mean()
            ),
            "fraction_normalized_regret_le_0.50_valid_only": float(
                (valid_normalized <= 0.50).mean()
            ),
        })
    return result


def main() -> None:
    args = parse_args()
    scene_dirs = sorted(path.parent for path in args.root.glob("*/transfer_metrics.npz"))
    if len(scene_dirs) < 2:
        raise RuntimeError(f"Expected multiple completed scenes below {args.root}")
    scenes = [path.name for path in scene_dirs]
    data = [np.load(path / "transfer_metrics.npz") for path in scene_dirs]
    deltas = sorted(
        int(name.removeprefix("transfer_error_d"))
        for name in data[0].files if name.startswith("transfer_error_d")
    )
    self_all = np.stack([item["self_error"] for item in data])
    uniform_all = np.stack([item["uniform_error"] for item in data])
    output: dict[str, object] = {
        "scenes": scenes,
        "retention": 0.25,
        "interpretation": {
            "oracle": "best source selected separately on each evaluated scene",
            "fixed_in_sample": "one source selected by mean error over all seven scenes",
            "leave_one_scene_out": "source selected on six scenes and evaluated on the held-out scene",
        },
        "deltas": {},
    }
    mapping_rows: list[dict[str, object]] = []

    for delta in deltas:
        transfer = np.stack([item[f"transfer_error_d{delta}"] for item in data])
        # [scene, layer-pair, source-head, target-head]
        target_self = self_all[:, delta:]
        target_uniform = uniform_all[:, delta:]
        oracle = transfer.min(axis=2)
        global_source = transfer.mean(axis=0).argmin(axis=1)
        fixed = np.take_along_axis(
            transfer,
            global_source[None, :, None, :],
            axis=2,
        ).squeeze(2)
        same_index = np.diagonal(transfer, axis1=2, axis2=3)

        loso = np.empty_like(oracle)
        for held_out in range(len(scenes)):
            train_indices = [i for i in range(len(scenes)) if i != held_out]
            source = transfer[train_indices].mean(axis=0).argmin(axis=1)
            loso[held_out] = np.take_along_axis(
                transfer[held_out], source[:, None, :], axis=1
            ).squeeze(1)

        per_scene_best_source = transfer.argmin(axis=2)
        mapping_agreement = (
            per_scene_best_source == global_source[None]
        ).mean()
        valid = target_self < target_uniform
        delta_result = {
            "fraction_targets_self_better_than_uniform": float(valid.mean()),
            "per_scene_oracle": score(oracle, target_self, target_uniform),
            "fixed_in_sample": score(fixed, target_self, target_uniform),
            "leave_one_scene_out": score(loso, target_self, target_uniform),
            "same_head_index": score(same_index, target_self, target_uniform),
            "scene_agreement_with_fixed_best_source": float(mapping_agreement),
        }
        output["deltas"][str(delta)] = delta_result

        mean_transfer = transfer.mean(axis=0)
        for pair_index, target_layer in enumerate(range(delta, 24)):
            source_layer = target_layer - delta
            for target_head in range(16):
                source_head = int(global_source[pair_index, target_head])
                mapping_rows.append({
                    "delta": delta,
                    "source_layer": source_layer,
                    "source_head": source_head,
                    "target_layer": target_layer,
                    "target_head": target_head,
                    "mean_transfer_error": float(
                        mean_transfer[pair_index, source_head, target_head]
                    ),
                    "mean_self_error": float(target_self[:, pair_index, target_head].mean()),
                    "mean_uniform_error": float(target_uniform[:, pair_index, target_head].mean()),
                    "scene_agreement": float(
                        (per_scene_best_source[:, pair_index, target_head] == source_head).mean()
                    ),
                })

    # Report the manually discussed adjacent-layer chain separately.
    d1 = np.stack([item["transfer_error_d1"] for item in data])
    chain_rows = []
    for (source_layer, source_head), (target_layer, target_head) in zip(
        MANUAL_CHAIN, MANUAL_CHAIN[1:]
    ):
        pair_index = source_layer
        candidate = d1[:, pair_index, source_head, target_head]
        target_self = self_all[:, target_layer, target_head]
        target_uniform = uniform_all[:, target_layer, target_head]
        chain_rows.append({
            "source": f"L{source_layer}H{source_head}",
            "target": f"L{target_layer}H{target_head}",
            **score(candidate, target_self, target_uniform),
        })
    output["manual_chain"] = chain_rows

    direct_rows = []
    for (source_layer, source_head), (target_layer, target_head) in DIRECT_REUSE_PROBES:
        delta = target_layer - source_layer
        transfer = np.stack([item[f"transfer_error_d{delta}"] for item in data])
        candidate = transfer[:, source_layer, source_head, target_head]
        target_self = self_all[:, target_layer, target_head]
        target_uniform = uniform_all[:, target_layer, target_head]
        direct_rows.append({
            "source": f"L{source_layer}H{source_head}",
            "target": f"L{target_layer}H{target_head}",
            "delta": delta,
            **score(candidate, target_self, target_uniform),
        })
    output["direct_reuse_probes"] = direct_rows

    (args.root / "aggregate_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    with (args.root / "fixed_best_mappings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(mapping_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(mapping_rows)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
