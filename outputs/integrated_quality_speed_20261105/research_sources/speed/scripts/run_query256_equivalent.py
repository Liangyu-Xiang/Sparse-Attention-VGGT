"""Fresh paired Shared Value baseline/exact candidate, only the three authorized scenes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/query256_equivalent_20261105';PYTHON=sys.executable
FROZEN=ROOT/'outputs/query256_efficiency_20261105/frozen_baseline'
def worker(scene,gpu,profile=False):
 result=[]
 for variant,source,config in [('baseline',FROZEN,'reuse_shared_value_loss_q256.json'),('exact',ROOT,'reuse_shared_value_loss_q256_exact.json')]:
  label=variant+('_profile' if profile else '')
  out=OUT/scene/label
  if (out/'result.json').exists():continue
  if out.exists():raise RuntimeError('Preserving incomplete result '+str(out))
  if profile:
   forward=[str(ROOT/'scripts/profile_deep_saf3r_query256.py'),'--scene',scene,'--method','shared_value_q256','--output-dir',str(out),'--config',str(source/'configs/evaluation'/config),'--source-root',str(source)]
  else:forward=['scripts/evaluate_zdrop_history_states.py','--dataset','7scenes','--scene',scene,'--frames','128','--method','zdrop_history_states','--config','configs/evaluation/'+config,'--output-dir',str(out),'--warmup','1','--repeats','3']
  cmd=[PYTHON,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(OUT/f'{scene}_{label}_monitor.json'),'--log',str(OUT/f'{scene}_{label}.log'),'--',PYTHON,str(ROOT/'scripts/launch_local_evaluation.py'),*forward]
  (OUT/f'{scene}_{label}_command.json').write_text(json.dumps(dict(cwd=str(source),argv=cmd),indent=2)+'\n')
  print('START',scene,variant,gpu,flush=True);p=subprocess.run(cmd,cwd=source);print('END',scene,variant,p.returncode,flush=True)
  result.append(dict(scene=scene,variant=label,gpu=gpu,returncode=p.returncode))
  if p.returncode:raise RuntimeError('Failed '+str(out))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');a=p.parse_args()
 assert json.loads((OUT/'exact_graph.json').read_text())['passed']
 with ThreadPoolExecutor(max_workers=3) as pool:status=[f.result() for f in [pool.submit(worker,s,g,a.profile) for s,g in zip(('heads','chess','stairs'),(0,4,5))]]
 (OUT/('profile_status.json' if a.profile else 'quality_status.json')).write_text(json.dumps(status,indent=2)+'\n')
