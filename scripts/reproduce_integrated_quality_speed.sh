#!/usr/bin/env bash
set -euo pipefail

# Reproduce the two frozen methods without retuning or rerunning unchanged
# Dense/SAF3R controls. All geometric metrics and separate 7Scenes timing are
# retained. Run from this integrated worktree with its dataset/checkpoint links.
task_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${SAF3R_EVAL_PYTHON:-/data_SSD2/mmc_lyxiang/anaconda3/envs/sparse-vggt/bin/python}"
trial_output="$(realpath -m -- "${1:?Pass a fresh output directory}")"
if [[ -e "$trial_output" ]]; then
  echo "Use a fresh output directory: $trial_output" >&2
  exit 1
fi
cd "$task_project_root"
task_plan="outputs/integrated_quality_speed_20261105/final_plan.json"
mkdir -p "$trial_output"
"$task_python" - "$task_project_root" "$task_plan" "$trial_output/plan.json" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);plan=json.loads(Path(sys.argv[2]).read_text())
for name,spec in plan['methods'].items():
    spec['config']=str(root/'configs/evaluation/integrated_20261105'/f'{name}.json')
plan['user_approved_control_timing']=False
Path(sys.argv[3]).write_text(json.dumps(plan,indent=2)+'\n')
PY
"$task_python" scripts/run_integrated_final.py --plan "$trial_output/plan.json" --output "$trial_output/runs" --gpus 0 1 2 3 4 5
"$task_python" scripts/report_integrated_final.py --plan "$trial_output/plan.json" --runs "$trial_output/runs" --output "$trial_output/reports"
