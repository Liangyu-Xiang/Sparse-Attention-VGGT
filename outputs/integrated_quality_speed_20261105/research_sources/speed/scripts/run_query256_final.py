"""Frozen deterministic reference / split-execution candidate on paired idle GPUs."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/query256_deterministic_split_20261105';SOURCE=OUT/'source';PYTHON=sys.executable

def worker(scene,gpu,profile):
 rows=[]
 for variant,config in [('reference','reuse_shared_value_loss_q256_deterministic_reference.json'),('optimized','reuse_shared_value_loss_q256_deterministic_split.json')]:
  label=variant+('_profile' if profile else '');out=OUT/scene/label
  if (out/'result.json').exists():continue
  if out.exists():raise RuntimeError('Incomplete result retained '+str(out))
  forward=[str(SOURCE/'scripts/profile_deep_saf3r_query256.py'),'--scene',scene,'--method','shared_value_q256','--output-dir',str(out),'--config',str(SOURCE/'configs/evaluation'/config),'--source-root',str(SOURCE)] if profile else ['scripts/evaluate_zdrop_history_states.py','--dataset','7scenes','--scene',scene,'--frames','128','--method','zdrop_history_states','--config','configs/evaluation/'+config,'--output-dir',str(out),'--warmup','1','--repeats','3']
  command=[PYTHON,str(SOURCE/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(OUT/f'{scene}_{label}_monitor.json'),'--log',str(OUT/f'{scene}_{label}.log'),'--',PYTHON,str(SOURCE/'scripts/launch_local_evaluation.py'),*forward]
  env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
  (OUT/f'{scene}_{label}_command.json').write_text(json.dumps(dict(cwd=str(SOURCE),argv=command,extra_env=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8')),indent=2)+'\n')
  print('START',scene,label,gpu,flush=True);proc=subprocess.run(command,cwd=SOURCE,env=env);print('END',scene,label,proc.returncode,flush=True)
  rows.append(dict(scene=scene,variant=label,gpu=gpu,returncode=proc.returncode))
  if proc.returncode:raise RuntimeError('Failed '+str(out))
  if not profile and variant=='optimized':
   a=json.loads((OUT/scene/'reference/output_sha256.json').read_text());b=json.loads((out/'output_sha256.json').read_text())
   if a!=b:raise RuntimeError('Rejected: full predictions differ '+scene)
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');a=p.parse_args()
 with ThreadPoolExecutor(max_workers=3) as pool:status=[f.result() for f in [pool.submit(worker,s,g,a.profile) for s,g in zip(('heads','chess','stairs'),(0,1,2))]]
 (OUT/('profile_status.json' if a.profile else 'quality_status.json')).write_text(json.dumps(status,indent=2)+'\n')
