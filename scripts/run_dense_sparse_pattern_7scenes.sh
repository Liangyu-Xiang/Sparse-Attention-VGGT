#!/usr/bin/env bash
# Dense-calibrate 25%-retention static sparse patterns for global layers 0--2.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
output_root="${1:-${root_dir}/outputs/dense_sparse_patterns/7scenes_n10_s5_layers_00_02}"
gpu_ids="${GPU_IDS:-4,5,6}"
IFS=',' read -r -a gpus <<<"$gpu_ids"
if (( ${#gpus[@]} == 0 )); then
  echo "GPU_IDS must provide at least one GPU." >&2
  exit 2
fi

# scene, sequence, and layer are independent calibration jobs.  Each worker
# consumes only one GPU to keep the profiling activations isolated.
jobs=()
for scene_sequence in \
  chess:seq-01 fire:seq-01 heads:seq-01 office:seq-01 \
  pumpkin:seq-01 redkitchen:seq-01 stairs:seq-02; do
  scene="${scene_sequence%%:*}"
  sequence="${scene_sequence#*:}"
  for layer in 0 1 2; do
    jobs+=("${scene}:${sequence}:${layer}")
  done
done

run_worker() {
  local gpu="$1"
  local worker="$2"
  local index
  for ((index=worker; index<${#jobs[@]}; index+=${#gpus[@]})); do
    local job="${jobs[$index]}"
    local scene="${job%%:*}"
    local rest="${job#*:}"
    local sequence="${rest%%:*}"
    local layer="${rest##*:}"
    local scene_dir="${output_root}/${scene}"
    mkdir -p "$scene_dir"
    echo "gpu=${gpu} scene=${scene} sequence=${sequence} layer=${layer}"
    CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" \
      PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
      "$python_bin" "$root_dir/scripts/find_vggt_dense_sparse_patterns.py" \
        --layer "$layer" \
        --scene "$scene" \
        --sequence "$sequence" \
        --first-frame 0 \
        --num-frames 10 \
        --frame-gap 5 \
        --retention 0.25 \
        --query-chunk-size 256 \
        --output-dir "$scene_dir" \
      >"${scene_dir}/layer_${layer}.log" 2>&1
  done
}

mkdir -p "$output_root"
pids=()
for worker in "${!gpus[@]}"; do
  run_worker "${gpus[$worker]}" "$worker" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
