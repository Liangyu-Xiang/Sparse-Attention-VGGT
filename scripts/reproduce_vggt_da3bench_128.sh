#!/usr/bin/env bash
set -euo pipefail

# Reproduce the dense VGGT rows in SAF3R Tables 2 and 3 on DA3-Bench.
# The data and checkpoint must be available through the workspace-local links.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
output_dir="${1:-${root_dir}/outputs/reproduce_vggt_da3bench_128}"

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
    "$python_bin" evaluation/evaluate_da3bench.py \
      --config_path ../configs/evaluation --config "$config" \
      >"$log_file" 2>&1 &
}

run_shard 0 reproduce_vggt_da3bench_128_gpu0 "$output_dir/gpu0.log"
pid0=$!
run_shard 1 reproduce_vggt_da3bench_128_gpu1 "$output_dir/gpu1.log"
pid1=$!
run_shard 2 reproduce_vggt_da3bench_128_gpu2 "$output_dir/gpu2.log"
pid2=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
exit "$status"
