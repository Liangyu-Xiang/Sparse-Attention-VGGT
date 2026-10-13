#!/usr/bin/env bash
# Profile the first three global layers with iterative output-preserving pruning.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
output_root="${1:-${root_dir}/outputs/iterative_pruning/7scenes_n10_s5_layers_00_02}"
gpu_ids="${GPU_IDS:-0,1,2,3,4,5,6}"
IFS=',' read -r -a gpus <<<"$gpu_ids"
scenes=(chess fire heads office pumpkin redkitchen stairs)
sequences=(seq-01 seq-01 seq-01 seq-01 seq-01 seq-01 seq-02)

if (( ${#gpus[@]} == 0 )); then
  echo "GPU_IDS must provide at least one GPU." >&2
  exit 2
fi

run_scene() {
  local gpu="$1"
  local scene="$2"
  local sequence="$3"
  local scene_dir="${output_root}/${scene}"
  mkdir -p "$scene_dir"
  for layer in 0 1 2; do
    echo "gpu=${gpu} scene=${scene} sequence=${sequence} layer=${layer}"
    CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" \
      PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
      "$python_bin" "$root_dir/scripts/find_vggt_iterative_pruning_sparse_attention.py" \
        --layer "$layer" \
        --scene "$scene" \
        --sequence "$sequence" \
        --first-frame 0 \
        --num-frames 10 \
        --frame-gap 5 \
        --retention 0.25675675675675674 \
        --prune-stages 0.75 0.50 0.375 \
        --query-chunk-size 256 \
        --output-dir "$scene_dir" \
      >"${scene_dir}/layer_${layer}.log" 2>&1
  done
}

mkdir -p "$output_root"
pids=()
for worker in "${!gpus[@]}"; do
  (
    for ((index=worker; index<${#scenes[@]}; index+=${#gpus[@]})); do
      run_scene "${gpus[$worker]}" "${scenes[$index]}" "${sequences[$index]}"
    done
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
