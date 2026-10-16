#!/usr/bin/env bash
# Profile patch-value cosine similarity for every VGGT global attention layer.
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/mmc_syang/miniconda3/envs/fastvggt/bin/python}"
output_dir="${1:-${root_dir}/outputs/value_similarity/chess_n10_s5_all_layers}"
scene="${SCENE:-chess}"
sequence="${SEQUENCE:-seq-01}"
gpu_ids="${GPU_IDS:-0,1,2}"
last_layer="${LAST_GLOBAL_LAYER:-23}"
IFS=',' read -r -a gpus <<<"$gpu_ids"

if (( ${#gpus[@]} == 0 )); then
  echo "GPU_IDS must provide at least one GPU." >&2
  exit 2
fi
if (( last_layer < 0 )); then
  echo "LAST_GLOBAL_LAYER must be non-negative." >&2
  exit 2
fi

mkdir -p "$output_dir"
run_worker() {
  local gpu="$1"
  local worker="$2"
  local layer
  for ((layer=worker; layer<=last_layer; layer+=${#gpus[@]})); do
    echo "gpu=${gpu} scene=${scene} sequence=${sequence} layer=${layer}"
    CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" \
      PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
      "$python_bin" "$root_dir/scripts/measure_vggt_value_similarity.py" \
        --layer "$layer" \
        --scene "$scene" \
        --sequence "$sequence" \
        --first-frame 0 \
        --num-frames 10 \
        --frame-gap 5 \
        --output-dir "$output_dir" \
      >"${output_dir}/layer_${layer}.log" 2>&1
  done
}

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
