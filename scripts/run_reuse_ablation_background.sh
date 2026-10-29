#!/usr/bin/env bash
set -eu
cd /mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT
export LD_LIBRARY_PATH=/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
exec /mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/bin/python -u scripts/run_reuse_ablation_eval.py --output-dir outputs/reuse_single_change_7scenes128_20261029
