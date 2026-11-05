"""Changed-method-only7Scenes full-forward timing; never rerun unchanged controls."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_camera_retention10_campaign import OUT,CONFIG
while not (OUT/'execution.json').exists():time.sleep(2)
assert not json.loads((OUT/'execution.json').read_text())['failures']
cmd=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu','6','--monitor',str(OUT/'speed_gpu.json'),'--log',str(OUT/'speed.log'),'--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset','7scenes','--scene','heads','--frames','128','--method','zdrop_history_states','--config',str(CONFIG),'--expected-input',str(ROOT/'outputs/global_state_loss_oracle_20261105/speed_heads128/layout_fixed/input.json'),'--output-dir',str(OUT/'speed_heads128'),'--warmup','2','--repeats','5','--timing-only']
subprocess.run(cmd,cwd=ROOT,check=True)
from scripts.report_camera_retention10_campaign import report
report()
