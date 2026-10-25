#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for the Stage-1 diagnostics documented in:
# docs/experiments/unified_next_round_handoff.md
#
# GPU placement is intentionally external:
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_unified_next_round_diagnostics.sh
#
# Override:
#   SCENES="chess fire redkitchen"
#   OUT_ROOT=outputs/unified_next_round
#   PYTHON=python

PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-outputs/unified_next_round}"
SCENES="${SCENES:-chess fire redkitchen}"

run_fresh () {
  local out="$1"; shift
  if [[ -e "$out" ]]; then
    echo "Refusing to overwrite existing diagnostic output: $out" >&2
    exit 2
  fi
  "$@"
}

mkdir -p "$OUT_ROOT"

run_fresh "$OUT_ROOT/history_scaling"   "$PYTHON" scripts/benchmark_history_reuse_scaling.py   --frames 128 --reuse-ratios 0 0.25 0.50 0.75 0.80   --warmup 5 --repeats 30   --output-dir "$OUT_ROOT/history_scaling"

for scene in $SCENES; do
  run_fresh "$OUT_ROOT/reuse_ceiling/$scene"     "$PYTHON" scripts/diagnose_history_reuse_ceiling.py     --scene "$scene" --num-frames 2 --first-frame 0 --frame-gap 20     --target-retention 0.20 --candidate-retentions 0.20 0.30 0.40 0.50 0.80     --layer-start 8 --layer-end 17     --output-dir "$OUT_ROOT/reuse_ceiling/$scene"

  run_fresh "$OUT_ROOT/history_pipeline/$scene"     "$PYTHON" scripts/diagnose_history_pipeline.py     --scene "$scene" --num-frames 2 --first-frame 0 --frame-gap 20     --target-retention 0.20 --layer-start 9 --layer-end 16     --output-dir "$OUT_ROOT/history_pipeline/$scene"

  run_fresh "$OUT_ROOT/compressibility/$scene"     "$PYTHON" scripts/diagnose_value_cluster_compressibility.py     --scene "$scene" --num-frames 2 --first-frame 0 --frame-gap 20     --layer-start 10 --layer-end 15 --top-clusters 64     --output-dir "$OUT_ROOT/compressibility/$scene"
done

inputs=()
for scene in $SCENES; do
  inputs+=("$OUT_ROOT/compressibility/$scene/clusters.csv")
done

"$PYTHON" scripts/summarize_value_compressibility.py   "${inputs[@]}"   --positive-fraction 0.20   --output "$OUT_ROOT/compressibility/summary.json"

echo
echo "Stage-1 diagnostics complete."
echo "Read: docs/experiments/unified_next_round_handoff.md"
echo "Next required artifact: $OUT_ROOT/analysis.md"
