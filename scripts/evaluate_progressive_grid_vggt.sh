#!/usr/bin/env bash
set -euo pipefail

# Evaluate the progressive 2x2-grid sparse attention on all 7Scenes and ETH3D
# scenes.  Each process owns exactly one GPU and one disjoint scene shard.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
output_dir="${1:-${root_dir}/outputs/progressive_grid_uniform_25pct}"
# The four evaluator processes share 64 available CPU cores.  TraDE uses
# cKDTree's parallel query path; give each SAF3R shard a bounded quarter of
# those cores instead of letting every process spawn workers for all cores.
kdtree_workers="${SAF3R_KDTREE_WORKERS:-16}"

mkdir -p "$output_dir"
cd "$root_dir"

run_shard() {
  local gpu="$1"
  local config="$2"
  local log_file="$3"
  CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    SAF3R_KDTREE_WORKERS="$kdtree_workers" \
    "$python_bin" evaluation/evaluate_da3bench.py \
      --config_path ../configs/evaluation --config "$config" \
      >"$log_file" 2>&1 &
}

run_shard 0 progressive_grid_full_gpu0 "$output_dir/gpu0.log"
pid0=$!
run_shard 1 progressive_grid_full_gpu1 "$output_dir/gpu1.log"
pid1=$!
run_shard 2 progressive_grid_full_gpu2 "$output_dir/gpu2.log"
pid2=$!
run_shard 3 progressive_grid_full_gpu3 "$output_dir/gpu3.log"
pid3=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
wait "$pid3" || status=1
exit "$status"
