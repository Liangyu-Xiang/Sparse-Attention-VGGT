"""User-authorized RGB-only Dense and original SAF3R timing controls, GPU lease external."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/operator_repair_phase2_20261105/controls'
for method in ['dense','saf3r']:
 command=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset','7scenes','--scene','heads','--frames','128','--method',method,'--config',str(ROOT/'outputs/operator_repair_phase2_20261105/configs/baseline.json'),'--expected-input',str(ROOT/'outputs/global_state_loss_oracle_20261105/speed_heads128/layout_fixed/input.json'),'--output-dir',str(OUT/method),'--warmup','1','--repeats','3','--timing-only']
 with (OUT/f'{method}.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
