"""Fresh selected-method audit on only the requested three 128-frame sequences."""
import argparse,json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/selected_source32_audit_20261105'
p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=['heads','chess','stairs']);p.add_argument('--gpus',nargs='+',type=int,default=[3,4,6]);a=p.parse_args()
assert len(a.scenes)==len(a.gpus)
def worker(scene,gpu):
 variant='source_sequential_w32_t1';target=OUT/scene/variant;prefix=OUT/f'{scene}_{variant}'
 if (target/'result.json').exists():return dict(scene=scene,existing=True)
 if target.exists():raise RuntimeError('Preserve incomplete attempt: '+str(target))
 cmd=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(prefix)+'.monitor.json','--log',str(prefix)+'.log','--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/profile_source_clustering_detail.py'),'--scene',scene,'--config',str(ROOT/'configs/evaluation/reuse_shared_value_loss_q256.json'),'--output-dir',str(target)]
 print('START',scene,gpu,flush=True);result=subprocess.run(cmd,cwd=ROOT);row=dict(scene=scene,gpu=gpu,returncode=result.returncode);print('END',row,flush=True);return row
with ThreadPoolExecutor(max_workers=len(a.scenes)) as pool:rows=list(pool.map(lambda pair:worker(*pair),zip(a.scenes,a.gpus)))
(OUT/('coordinator_'+'_'.join(a.scenes)+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
if any(r.get('returncode',0) for r in rows):sys.exit(1)
