"""Quiet matched2-warmup/5-repeat full forwards, after all quality CPU tasks finish."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rep_history_loss_20261105'
def main():
 p=argparse.ArgumentParser();p.add_argument('--leased',action='store_true');a=p.parse_args()
 if not a.leased:
  while not (OUT/'full/summary.json').exists():
   if (OUT/'full/execution.json').exists() and json.loads((OUT/'full/execution.json').read_text())['failures']:raise RuntimeError('Full quality campaign failed')
   time.sleep(3)
  # Driver only writes summary after all GPU/CPU children have exited.
  cmd=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu','6','--monitor',str(OUT/'speed_gpu.json'),'--log',str(OUT/'speed.log'),'--',sys.executable,str(Path(__file__).resolve()),'--leased']
  subprocess.run(cmd,cwd=ROOT,check=True);return
 speed=OUT/'speed_heads128';speed.mkdir(exist_ok=True,parents=True)
 for name,method in [('baseline','zdrop_history_states'),('A','zdrop_history_states'),('B','zdrop_history_states'),('dense','dense'),('saf3r','saf3r')]:
  config=OUT/'full/configs'/f"{name if name in ['baseline','A','B'] else 'baseline'}.json"
  cmd=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset','7scenes','--scene','heads','--frames','128','--method',method,'--config',str(config),'--expected-input',str(ROOT/'outputs/global_state_loss_oracle_20261105/speed_heads128/layout_fixed/input.json'),'--output-dir',str(speed/name),'--warmup','2','--repeats','5','--timing-only']
  print('TIMING_START',name,flush=True)
  with (speed/f'{name}.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('TIMING_END',name,flush=True)
 (speed/'execution.json').write_text(json.dumps(dict(complete=True,gpu=6,warmup=2,repeats=5,scope='heads128 only; after all own CPU quality and GPU inference tasks finish; baseline/A/B/Dense/SAF3R same hardware,inputs,checkpoint and protocol',control_authorization='User supplied experiment explicitly requests Baseline/A/B; prior explicit permission for timing-only Dense and original SAF3R persists; no Dense/SAF3R quality reruns'))+'\n')
if __name__=='__main__':main()
