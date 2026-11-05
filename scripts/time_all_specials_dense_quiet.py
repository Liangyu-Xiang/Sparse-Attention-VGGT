"""Repeat matched timing after CPU quality completes; retain initial high-variance batch."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/eth3d_all_specials_dense_full_20261105/speed_heads128_quiet'
OUT.mkdir(parents=True,exist_ok=True)
for name,method,config in [('layout_fixed','zdrop_history_states','outputs/eth3d_layout_fixed_full_20261105/frozen_config.json'),('all_specials_dense','zdrop_history_states','outputs/eth3d_all_specials_dense_full_20261105/configs/all_specials_dense.json'),('dense','dense','outputs/eth3d_layout_fixed_full_20261105/frozen_config.json'),('saf3r','saf3r','outputs/eth3d_layout_fixed_full_20261105/frozen_config.json')]:
 command=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset','7scenes','--scene','heads','--frames','128','--method',method,'--config',str(ROOT/config),'--expected-input',str(ROOT/'outputs/global_state_loss_oracle_20261105/speed_heads128/layout_fixed/input.json'),'--output-dir',str(OUT/name),'--warmup','1','--repeats','3','--timing-only']
 print('TIMING_START',name,flush=True)
 with (OUT/f'{name}.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('TIMING_END',name,flush=True)
(OUT/'execution.json').write_text(json.dumps(dict(complete=True,methods=['all_specials_dense','layout_fixed','dense','saf3r'],authorization='Changed method evaluation; user previously explicitly authorized timing-only reruns of Dense, original SAF3R and layout-fixed baseline; no control quality reruns'))+'\n')
