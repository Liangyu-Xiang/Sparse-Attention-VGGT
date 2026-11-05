"""Conservative equivalent candidate: fresh SAF3R timing, reused reference quality."""
import argparse,hashlib,json,os,queue,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();old=a.parent;out=a.output
 assert read(old/'completion.json')['complete'];out.mkdir(parents=True,exist_ok=True);prior=read(old/'plan.json');cfg=ROOT/'configs/evaluation/cluster20_speed_20261105/conservative.json'
 if (out/'plan.json').exists():raise ValueError('Fresh campaign required')
 hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for folder in ('saf3r','evaluation') for x in (ROOT/folder).rglob('*.py')};hashes[str(cfg.relative_to(ROOT))]=hashlib.sha256(cfg.read_bytes()).hexdigest()
 plan=dict(prior,config=dict(reference=prior['config']['reference'],optimized=str(cfg)),source_sha256=hashes,reuse_control_batch=str(old),new_runs=36,method_records=54,frozen_before_quality=True,correction_reason='Retain only exact-tested common-forward fusions and allocator reuse; restore original Attention pipeline/source stores/native convolution selection',quality_tuning=False)
 save(out/'plan.json',plan);tasks=queue.Queue()
 for ds,scenes in prior['datasets'].items():
  for sc in scenes:
   target=out/'runs'/ds/sc/'reference';target.parent.mkdir(parents=True,exist_ok=True)
   source=(old/'runs'/ds/sc/'reference') if ds=='7scenes' else Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs/eth3d_cluster20_20261105/runs/eth3d')/sc/'integrated_main_f20'
   target.symlink_to(source)
   if ds=='7scenes':
    for folder,suffix in [('monitors','.json'),('logs','.log')]:
     p=out/folder/f'{ds}_{sc}_reference{suffix}';p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(old/folder/p.name)
   tasks.put((ds,sc))
 completed=[];running={};lock=threading.Lock();env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONUNBUFFERED='1')
 def verify():
  bad=[k for k,v in hashes.items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
  if bad:raise ValueError('Frozen source changed: '+str(bad))
 def status():save(out/'progress.json',dict(completed=completed,running=running,total=36))
 def worker(gpu):
  while True:
   try:ds,sc=tasks.get_nowait()
   except queue.Empty:return
   for name in ('saf3r','optimized'):
    verify();target=out/'runs'/ds/sc/name;monitor=out/'monitors'/f'{ds}_{sc}_{name}.json';log=out/'logs'/f'{ds}_{sc}_{name}.log';matched=out/'runs'/ds/sc/'reference/input.json'
    config=cfg if name=='optimized' else Path(prior['config']['reference'])
    cmd=[sys.executable,str(ROOT/'scripts/run_profiler_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(monitor),'--log',str(log),'--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',sc,'--frames','128','--method','saf3r' if name=='saf3r' else 'zdrop_history_states','--warmup','1','--repeats','3','--config',str(config),'--output-dir',str(target),'--matched-input',str(matched)]
    if name=='saf3r':cmd.append('--timing-only')
    row=dict(dataset=ds,scene=sc,method=name,gpu=gpu,output=str(target),command=cmd)
    save(out/'commands'/f'{ds}_{sc}_{name}.json',row)
    with lock:running[str(gpu)]=row;status()
    print('START',ds,sc,name,gpu,flush=True);begin=time.monotonic();code=subprocess.call(cmd,cwd=ROOT,env=env);row.update(returncode=code,wall_seconds=time.monotonic()-begin)
    result=target/('timing_result.json' if name=='saf3r' else 'result.json');row['complete']=result.exists() and (name=='saf3r' or read(result)['quality']['complete'])
    with lock:completed.append(row);running.pop(str(gpu),None);status()
    print('END',ds,sc,name,code,row['complete'],flush=True)
    if code or not row['complete']:raise RuntimeError('Run failed: '+str(target))
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,(4,5,6)))
 verify();save(out/'completion.json',dict(complete=len(completed)==36 and all(x['complete'] for x in completed),runs=completed,source_unchanged=True,reference_quality_reused=18));print('COMPLETE',len(completed),flush=True)
if __name__=='__main__':main()
