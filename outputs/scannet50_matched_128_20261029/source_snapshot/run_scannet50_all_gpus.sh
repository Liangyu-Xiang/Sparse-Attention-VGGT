#!/usr/bin/env bash
set -eu
cd /mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT
export LD_LIBRARY_PATH=/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
out=outputs/scannet50_matched_128_20261029
mkdir -p "$out"
for gpu in 0 1 2 3 4 5 6 7; do
 CUDA_VISIBLE_DEVICES="$gpu" /mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/bin/python -u scripts/evaluate_scannet50_matched.py --worker "$gpu" > "$out/worker${gpu}.log" 2>&1 &
done
wait
/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/bin/python scripts/evaluate_scannet50_matched.py
