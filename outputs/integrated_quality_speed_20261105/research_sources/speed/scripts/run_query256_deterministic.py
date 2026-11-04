"""User-authorized deterministic reference vs equivalent optimized execution."""
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/query256_deterministic_20261105';SOURCE=OUT/'frozen_source';PYTHON=sys.executable

def worker(scene,gpu,profile,candidate_source=None,candidate_label="exact"):
 source=SOURCE
 status=[]
 for variant,config in [('reference','reuse_shared_value_loss_q256_deterministic_reference.json'),('exact','reuse_shared_value_loss_q256_deterministic_exact.json')]:
  source=SOURCE if variant=='reference' else (candidate_source or SOURCE)
  method_label='reference' if variant=='reference' else candidate_label
  label=method_label+('_profile' if profile else '');out=OUT/scene/label
  if (out/'result.json').exists():continue
  if out.exists():raise RuntimeError('Preserving incomplete output '+str(out))
  forward=[str(source/'scripts/profile_deep_saf3r_query256.py'),'--scene',scene,'--method','shared_value_q256','--output-dir',str(out),'--config',str(source/'configs/evaluation'/config),'--source-root',str(source)] if profile else ['scripts/evaluate_zdrop_history_states.py','--dataset','7scenes','--scene',scene,'--frames','128','--method','zdrop_history_states','--config','configs/evaluation/'+config,'--output-dir',str(out),'--warmup','1','--repeats','3']
  command=[PYTHON,str(source/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(OUT/f'{scene}_{label}_monitor.json'),'--log',str(OUT/f'{scene}_{label}.log'),'--',PYTHON,str(source/'scripts/launch_local_evaluation.py'),*forward]
  env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
  (OUT/f'{scene}_{label}_command.json').write_text(json.dumps(dict(cwd=str(source),argv=command,extra_env=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8')),indent=2)+'\n')
  print('START',scene,label,gpu,flush=True);p=subprocess.run(command,cwd=source,env=env);print('END',scene,label,p.returncode,flush=True)
  status.append(dict(scene=scene,variant=label,gpu=gpu,returncode=p.returncode))
  if p.returncode:raise RuntimeError('Failed '+str(out))
  if not profile:
   if variant=='exact':
    left=json.loads((OUT/scene/'reference/output_sha256.json').read_text());right=json.loads((out/'output_sha256.json').read_text())
    if left!=right:raise RuntimeError('Full prediction bits differ; not accepted '+scene)
 return status
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');p.add_argument('--candidate-source',type=Path);p.add_argument('--candidate-label',default='exact');a=p.parse_args()
 assert all(x['predictions_equal'] for x in json.loads((OUT/'reference_repeatability/comparison.json').read_text()))
 with ThreadPoolExecutor(max_workers=3) as pool:
  status=[f.result() for f in [pool.submit(worker,s,g,a.profile,a.candidate_source,a.candidate_label) for s,g in zip(('heads','chess','stairs'),(1,3,4))]]
 (OUT/(a.candidate_label+('_profile_status.json' if a.profile else '_quality_status.json'))).write_text(json.dumps(status,indent=2)+'\n')
