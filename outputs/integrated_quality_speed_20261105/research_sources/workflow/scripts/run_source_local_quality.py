"""Predeclared three-scene clustering campaign plus exact selector Graph ablation."""
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source_local_quality_20261105';OUT.mkdir(exist_ok=True,parents=True)
configs=json.loads((ROOT/'outputs/source_local_unanchored_20261105/planned_configs.json').read_text())['configs']
variants=[('adaptive_graph','reuse_shared_value_loss_q256_adaptive_reference.json','graph'),('adaptive_eager','reuse_shared_value_loss_q256_adaptive_reference.json','eager')]
variants += [(c.removeprefix('reuse_shared_value_loss_q256_').removesuffix('.json'),c,'graph') for c in configs]
manifest=dict(scenes=['heads','chess','stairs'],frames=128,variants=variants,query_group_size=256,retention=.2,iterations=2,warmup=1,repeats=3,
              quality=['auc03','auc30','acc','comp','cd','f1'],selection='Report every predeclared variant; no tuning against test truth',
              controls='Dense/SAF3R unchanged: reuse historical quality, no rerun. Dense speedup N/A without new paired Dense timings.',
              graph_ablation='Same eager() selector mathematics; only bypass Window capture/replay. Both receive current inputs.')
(OUT/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')

def scene_worker(scene,gpu):
 results=[]
 for variant,config,execution in variants:
  target=OUT/scene/variant
  if (target/'result.json').exists():
   results.append(dict(scene=scene,variant=variant,existing=True));continue
  if target.exists():raise RuntimeError('Preserve incomplete attempt: '+str(target))
  prefix=OUT/f'{scene}_{variant}'
  command=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(prefix)+'.monitor.json','--log',str(prefix)+'.log','--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset','7scenes','--scene',scene,'--frames','128','--method','zdrop_history_states','--config',str(ROOT/'configs/evaluation'/config),'--output-dir',str(target),'--warmup','1','--repeats','3','--selector-execution',execution]
  print('START',scene,variant,'gpu',gpu,flush=True);proc=subprocess.run(command,cwd=ROOT)
  row=dict(scene=scene,variant=variant,returncode=proc.returncode,gpu=gpu);results.append(row)
  (OUT/f'{scene}_progress.json').write_text(json.dumps(results,indent=2)+'\n')
  print('END',row,flush=True)
  if proc.returncode:break
 return results
parser=argparse.ArgumentParser();parser.add_argument('--scenes',nargs='+',default=['heads','chess','stairs']);parser.add_argument('--gpus',nargs='+',type=int,default=[1,3,5]);args=parser.parse_args()
if len(args.scenes)!=len(args.gpus):raise ValueError('GPU count mismatch')
with ThreadPoolExecutor(max_workers=len(args.scenes)) as pool:
 futures=[pool.submit(scene_worker,s,g) for s,g in zip(args.scenes,args.gpus)]
 results=[f.result() for f in futures]
(OUT/('campaign_status_'+'_'.join(args.scenes)+'.json')).write_text(json.dumps(results,indent=2)+'\n')
if any(r.get('returncode',0) for group in results for r in group):sys.exit(1)
