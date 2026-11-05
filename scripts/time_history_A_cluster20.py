"""Matched A and A+all-specials timing after full ETH3D quality completes."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/rep_history_A_all_specials_cluster20_20261105'
def main():
 p=argparse.ArgumentParser();p.add_argument('--leased',action='store_true');a=p.parse_args()
 if not a.leased:
  while not (OUT/'quality_summary.json').exists():
   if (OUT/'execution.json').exists() and json.loads((OUT/'execution.json').read_text())['failures']:raise RuntimeError('Quality failed; retained results')
   time.sleep(3)
  subprocess.run([sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu','6','--monitor',str(OUT/'speed_gpu.json'),'--log',str(OUT/'speed.log'),'--',sys.executable,str(Path(__file__).resolve()),'--leased'],cwd=ROOT,check=True);return
 speed=OUT/'speed_heads128';speed.mkdir(parents=True,exist_ok=True)
 methods=[('cluster12','zdrop_history_states','configs/evaluation/reuse_rep_history_A_all_specials_dense.json'),('A_all_specials_cluster20','zdrop_history_states','outputs/rep_history_A_all_specials_cluster20_20261105/configs/A_all_specials_cluster20.json'),('dense','dense','configs/evaluation/reuse_rep_history_A_all_specials_dense.json'),('saf3r','saf3r','configs/evaluation/reuse_rep_history_A_all_specials_dense.json')]
 for name,method,cfg in methods:
  runner='scripts/evaluate_history_A_all_specials_profile.py' if name in ['cluster12','A_all_specials_cluster20'] else 'scripts/evaluate_zdrop_history_states.py'
  cmd=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/runner),'--dataset','7scenes','--scene','heads','--frames','128','--method',method,'--config',str(ROOT/cfg),'--expected-input',str(ROOT/'outputs/global_state_loss_oracle_20261105/speed_heads128/layout_fixed/input.json'),'--output-dir',str(speed/name),'--warmup','2','--repeats','5','--timing-only']
  print('TIMING_START',name,flush=True)
  with (speed/f'{name}.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('TIMING_END',name,flush=True)
 (speed/'execution.json').write_text(json.dumps(dict(complete=True,protocol='GPU6,heads128,2warmup+5repeat; same checkpoint/input/hardware; after CPU quality completes',control_authorization='Prior user authorization for timing-only A evaluation and Dense/SAF3R controls persists; no unchanged control quality reruns'))+'\n')
if __name__=='__main__':main()
