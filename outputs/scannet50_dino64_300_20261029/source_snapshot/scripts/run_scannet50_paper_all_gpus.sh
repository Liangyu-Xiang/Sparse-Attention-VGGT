#!/usr/bin/env bash
set -eu
cd /mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT
export LD_LIBRARY_PATH=/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for gpu in 0 1 2 3 4 5 6 7; do
 (
  export CUDA_VISIBLE_DEVICES="$gpu"
  for frames in 300 500 700; do
   export SCANNET_FRAMES="$frames"
   out="outputs/scannet50_dino64_${frames}_20261029"
   mkdir -p "$out"
   /mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/bin/python -u scripts/evaluate_scannet50_paper.py --worker "$gpu" > "$out/worker${gpu}.log" 2>&1
  done
 ) &
done
wait
for frames in 300 500 700; do
 SCANNET_FRAMES="$frames" /mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/bin/python scripts/evaluate_scannet50_paper.py
done
