"""Freeze and evaluate equivalent engineering changes; never inspect GT to tune."""
import argparse,hashlib,json,os,queue,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs')
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
 assert json.loads((out/'validation_v2/result.json').read_text())['passed']
 assert json.loads((out/'residual_validation_v2/result.json').read_text())['passed']
 assert '7 passed' in (out/'rope_validation.log').read_text()
 datasets=json.loads((BASE/'integrated_main_reeval_20261105/plan.json').read_text())['datasets']
 hashes={str(x.relative_to(ROOT)):digest(x) for folder in ('saf3r','evaluation') for x in (ROOT/folder).rglob('*.py')}
 configs={name:ROOT/f'configs/evaluation/cluster20_speed_20261105/{name}.json' for name in ('reference','optimized')}
 hashes.update({str(x.relative_to(ROOT)):digest(x) for x in configs.values()})
 plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),base_commit='e0107c38',datasets=datasets,gpus=[4,5,6],warmup=1,repeats=3,source_sha256=hashes,config={k:str(v) for k,v in configs.items()},frozen_before_quality=True,acceptance=dict(full_forward_seconds_lt=9,speed='per-scene paired warmed median method <= SAF3R; report all scenes and macro',pointcloud='absolute relative change <2% for ACC,Comp,CD,F1 equal-scene macros; report per-scene deviations too',reference='cluster_fraction .20 reference configuration; ETH3D historical complete;7Scenes newly completed because no matched .20 result exists'),dense_rerun=False,saf3r_timing_authorized=True,saf3r_quality_rerun=False,quality_tuning=False)
 if (out/'plan.json').exists():raise ValueError('Do not overwrite a frozen campaign')
 save(out/'plan.json',plan)
 def verify():
  bad=[k for k,v in hashes.items() if digest(ROOT/k)!=v]
  if bad:raise ValueError('Frozen source/config changed: '+str(bad))
 tasks=queue.Queue()
 for ds,scenes in datasets.items():
  for sc in scenes:tasks.put((ds,sc))
 completed=[];running={};lock=threading.Lock()
 env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONUNBUFFERED='1')
 def status():save(out/'progress.json',dict(completed=completed,running=running,remaining_scenes=tasks.qsize(),total_scenes=18,updated_utc=datetime.now(timezone.utc).isoformat()))
 def worker(gpu):
  while not tasks.empty():
   try:ds,sc=tasks.get_nowait()
   except queue.Empty:return
   methods=['reference','saf3r','optimized'] if ds=='7scenes' else ['saf3r','optimized']
   for name in methods:
    verify();target=out/'runs'/ds/sc/name;prefix=out/'logs'/f'{ds}_{sc}_{name}';monitor=out/'monitors'/f'{ds}_{sc}_{name}.json'
    match=(BASE/'integrated_main_reeval_20261105/runs'/ds/sc/'integrated_main/input.json') if ds=='7scenes' else BASE/'eth3d_cluster20_20261105/runs/eth3d'/sc/'integrated_main_f20/input.json'
    command=[sys.executable,str(ROOT/'scripts/run_profiler_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(monitor),'--log',str(prefix)+'.log','--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--method','saf3r' if name=='saf3r' else 'zdrop_history_states','--dataset',ds,'--scene',sc,'--frames','128','--warmup','1','--repeats','3','--matched-input',str(match),'--config',str(configs['optimized' if name=='optimized' else 'reference']),'--output-dir',str(target)]
    if name=='saf3r':command.append('--timing-only')
    row=dict(dataset=ds,scene=sc,method=name,gpu=gpu,output=str(target),monitor=str(monitor),log=str(prefix)+'.log',command=command)
    save(out/'commands'/f'{ds}_{sc}_{name}.json',row)
    with lock:running[str(gpu)]=row;status()
    print('START',ds,sc,name,gpu,flush=True);begin=time.monotonic();code=subprocess.call(command,cwd=ROOT,env=env)
    row.update(returncode=code,wall_seconds=time.monotonic()-begin)
    result=target/('timing_result.json' if name=='saf3r' else 'result.json')
    row['complete']=result.exists() and (name=='saf3r' or json.loads(result.read_text())['quality']['complete'])
    with lock:completed.append(row);running.pop(str(gpu),None);status()
    print('END',ds,sc,name,code,row['complete'],flush=True)
    if code or not row['complete']:raise RuntimeError('Run failed; preserve evidence: '+str(target))
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,[4,5,6]))
 verify();save(out/'completion.json',dict(complete=len(completed)==43 and all(x['complete'] for x in completed),runs=completed,source_unchanged=True))
 print('COMPLETE',len(completed),flush=True)
if __name__=='__main__':main()
