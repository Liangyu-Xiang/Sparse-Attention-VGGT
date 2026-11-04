"""Three-scene, two-predeclared-layout operation profiling; strict GPU leases."""
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source_clustering_detail_20261105';OUT.mkdir(parents=True,exist_ok=True)
variants=('source_sequential_w32_t1','source_spatiotemporal_w32_t2')
(OUT/'plan.json').write_text(json.dumps(dict(scenes=['heads','chess','stairs'],frames=128,variants=variants,query_group_size=256,source_width=32,iterations=2,
 scope='Same existing full-forward deep profile as historical 2.098010 s; 3 separate event passes, choose one by median whole-wall per scene, then average scenes. New source subscopes preserve statements. Clustering excludes one-time source layout; report that separately.',quality='Diagnostics only: all six fields N/A; reuse completed quality campaign',dense_speedup=None,dense_speedup_reason='No paired Dense timing; unchanged control not rerun'),indent=2)+'\n')
def worker(scene,gpu):
 rows=[]
 for variant in variants:
  out=OUT/scene/variant
  if (out/'result.json').exists():continue
  if out.exists():raise RuntimeError('Preserve incomplete output '+str(out))
  prefix=OUT/f'{scene}_{variant}'
  cmd=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(prefix)+'.monitor.json','--log',str(prefix)+'.log','--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/profile_source_clustering_detail.py'),'--scene',scene,'--config',str(ROOT/'configs/evaluation'/f'reuse_shared_value_loss_q256_{variant}.json'),'--output-dir',str(out)]
  print('START',scene,variant,gpu,flush=True);p=subprocess.run(cmd,cwd=ROOT);rows.append(dict(scene=scene,variant=variant,gpu=gpu,returncode=p.returncode));print('END',rows[-1],flush=True)
  if p.returncode:break
 return rows
parser=argparse.ArgumentParser();parser.add_argument('--scenes',nargs='+',default=['heads','chess','stairs']);parser.add_argument('--gpus',nargs='+',type=int,default=[2,3,5]);args=parser.parse_args()
if len(args.scenes)!=len(args.gpus):raise ValueError('GPU count mismatch')
with ThreadPoolExecutor(max_workers=len(args.scenes)) as pool:
 results=list(pool.map(lambda pair:worker(*pair),zip(args.scenes,args.gpus)))
(OUT/('coordinator_status_'+'_'.join(args.scenes)+'.json')).write_text(json.dumps(results,indent=2)+'\n')
if any(row['returncode'] for group in results for row in group):sys.exit(1)
