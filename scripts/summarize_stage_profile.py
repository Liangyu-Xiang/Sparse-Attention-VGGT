"""Summarize stage_profile.json into an exclusive per-stage timing report.

Run: python scripts/summarize_stage_profile.py outputs/.../stage_profile.json
No CUDA or PyTorch dependency is required.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


PARENTS = (
    "value_update", "pooling", "coarse_observation", "reuse_prior",
    "reuse_and_budget", "history_summary", "sub/compact_history_bank",
    "execution_and_evidence",
)


def summarize(data):
    totals = defaultdict(float)
    calls = defaultdict(int)
    sub_stages = defaultdict(float)
    layers = data["layers"]
    for row in layers:
        for name, val in row["stages"].items():
            totals[name] += val["ms"]
            calls[name] += val["calls"]
            if name.startswith("sub/") and name != "sub/compact_history_bank":
                sub_stages[name] += val["ms"]

    table = []
    accounted = sum(totals[p] for p in PARENTS)
    for name in PARENTS:
        if name == "reuse_and_budget":
            residual = totals[name] - sum(sub_stages.values())
            for subname, elapsed in sorted(sub_stages.items(), key=lambda kv: -kv[1]):
                table.append(dict(stage=subname, ms=elapsed, calls=calls[subname]))
            table.append(dict(stage="reuse_and_budget/other", ms=residual,
                              calls=calls[name]))
        else:
            table.append(dict(stage=name, ms=totals[name], calls=calls[name]))
    table = sorted(table, key=lambda x: -x["ms"])
    for item in table:
        item["percentage_of_accounted"] = 100 * item["ms"] / max(accounted, 1e-12)
    return table, accounted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    data = json.loads(args.profile.read_text())
    table, accounted = summarize(data)
    wall = data["profile_wall_ms"]
    print(f"Profile wall: {wall / 1000:.3f} s (instrumented, not baseline)")
    print(f"Measured top-level event spans: {accounted / 1000:.3f} s")
    print("Stages: nested sub/ durations are removed from reuse_and_budget/other")
    print(f"{'stage':40} {'ms':>12} {'% parent sum':>14} {'calls':>8}")
    for row in table:
        print(f"{row['stage']:40} {row['ms']:12.3f} {row['percentage_of_accounted']:13.2f}% {row['calls']:8}")
    print(f"Unattributed instrumented wall minus parent spans: {(wall-accounted)/1000:.3f} s")
    print("Warning: CUDA event spans include launch gaps; no stage numbers are clean uninstrumented speedups.")


if __name__ == "__main__":
    main()
