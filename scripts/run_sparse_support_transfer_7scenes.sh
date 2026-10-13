#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
output_root="${1:-${root_dir}/outputs/sparse_support_transfer/7scenes_top25}"
scenes=(chess fire heads office pumpkin redkitchen stairs)
sequences=(seq-01 seq-01 seq-01 seq-01 seq-01 seq-01 seq-02)

mkdir -p "$output_root"
cd "$root_dir"
pids=()
for gpu in 0 1 2 3 4 5 6; do
  scene="${scenes[$gpu]}"
  sequence="${sequences[$gpu]}"
  scene_dir="$output_root/$scene"
  mkdir -p "$scene_dir"
  CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    "$python_bin" scripts/analyze_sparse_support_transfer.py \
      --scene "$scene" \
      --sequence "$sequence" \
      --first-frame 0 \
      --frame-gap 20 \
      --num-frames 2 \
      --retention 0.25 \
      --layer-deltas 1 2 4 8 \
      --query-chunk-size 64 \
      --output-dir "$scene_dir" \
      >"$scene_dir/run.log" 2>&1 &
  pids+=("$!")
  echo "gpu=$gpu pid=${pids[-1]} scene=$scene sequence=$sequence"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
