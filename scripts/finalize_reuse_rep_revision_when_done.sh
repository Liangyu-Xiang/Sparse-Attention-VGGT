#!/usr/bin/env bash
set -eu
cd /mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
while :; do
    if /mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/bin/python - <<'PY'
from pathlib import Path
import json,sys
p=Path('outputs/reuse_rep_revision_7scenes128_20261029/queue_status.json')
sys.exit(0 if p.exists() and json.loads(p.read_text())['all_terminal'] else 1)
PY
    then break; fi
    sleep 30
done
/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/bin/python -m scripts.summarize_reuse_rep_revision outputs/reuse_rep_revision_7scenes128_20261029
