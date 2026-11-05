"""Frozen ETH3D three-scene operator trials with independent GPU and CPU queues."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/operator_repair_20261105'
BASE=ROOT/'outputs/eth3d_layout_fixed_full_20261105'
SCENES=['pipes','courtyard','playground']
VARIANTS=['loss_only','states_only','both','both_mass_robust']
def read(p):return json.loads(p.read_text())
def save(p,v):
 t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 (OUT/'configs').mkdir(exist_ok=True,parents=True)
 base=read(BASE/'frozen_config.json')
 deltas={'loss_only':dict(operator_loss='normalized_moment'),'states_only':dict(partial_key_states=True),
  'both':dict(operator_loss='normalized_moment',partial_key_states=True),
  'both_mass_robust':dict(operator_loss='normalized_moment',partial_key_states=True,operator_mass_penalty=1.)}
 configs={'baseline':base}|{name:base|delta for name,delta in deltas.items()}
 for name,cfg in configs.items():save(OUT/'configs'/f'{name}.json',cfg)
 sources={str(p.relative_to(ROOT)):digest(p) for folder in ['saf3r','evaluation'] for p in (ROOT/folder).rglob('*.py')}
 for name in ['scripts/evaluate_zdrop_history_states.py','scripts/evaluate_saved_reuse_predictions.py','scripts/quality_metrics.py','scripts/launch_local_evaluation.py']:sources[name]=digest(ROOT/name)
 manifest=dict(scenes=SCENES,variants=VARIANTS,source_sha256=sources,configs={v:dict(config=str(OUT/'configs'/f'{v}.json'),config_sha256=digest(OUT/'configs'/f'{v}.json')) for v in VARIANTS},
  frozen_before_quality=True,selection='AUC3/AUC30 improvement across calibration scenes, retain reconstruction quality; no slower matched warmed7Scenes forward;7Scenes test truth never used to choose settings',
  scope='Three ETH3D calibration scenes, not full11-scene result; no oracle in production; unchanged16% average budget',gpus=[4,5,6])
 save(OUT/'frozen_manifest.json',manifest)
 for v in VARIANTS:
  out=OUT/'quality'/v;(out/'inputs').mkdir(parents=True);(out/'runs').mkdir()
  for s in SCENES:save(out/'inputs'/f'{s}.json',read(BASE/'runs'/s/'input.json'))
 save(OUT/'queue.json',dict(pending=[dict(variant=v,scene=s) for s in SCENES for v in VARIANTS],claims=[]))
 return manifest
def plan_for(m,v):return dict(m['configs'][v],source_sha256=m['source_sha256'])
def worker(gpu,m):
 from scripts.run_eth3d_layout_fix_evaluation import verify
 save(OUT/f'worker{gpu}_started.json',dict(gpu=gpu,pid=os.getpid()))
 while True:
  with (OUT/'queue.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);q=read(OUT/'queue.json')
   if not q['pending']:return
   job=q['pending'].pop(0);job.update(gpu=gpu,status='forward');q['claims'].append(job);save(OUT/'queue.json',q)
  v,s=job['variant'],job['scene'];plan=plan_for(m,v);verify(plan);out=OUT/'quality'/v
  cmd=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset','eth3d','--scene',s,'--frames','128','--method','zdrop_history_states','--config',plan['config'],'--expected-input',str(out/'inputs'/f'{s}.json'),'--output-dir',str(out/'runs'/s),'--warmup','0','--repeats','1','--defer-quality']
  print('START',gpu,v,s,flush=True)
  with (out/(s+'_forward.log')).open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  with (OUT/'queue.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);q=read(OUT/'queue.json')
   for item in q['claims']:
    if item['variant']==v and item['scene']==s:item.update(status='done' if code==0 else 'failed',returncode=code)
   save(OUT/'queue.json',q)
  print('END',gpu,v,s,code,flush=True)
  if code:raise SystemExit(code)
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[4,5,6]);a=p.parse_args()
 m=read(OUT/'frozen_manifest.json') if (OUT/'frozen_manifest.json').exists() else prepare()
 if a.worker is not None:worker(a.worker,m);return
 from scripts.run_eth3d_layout_fix_evaluation import cpu_quality
 import psutil
 processes={gpu:subprocess.Popen([sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(OUT/f'worker{gpu}_gpu.json'),'--log',str(OUT/f'worker{gpu}.log'),'--',sys.executable,str(Path(__file__).resolve()),'--worker',str(gpu)],cwd=ROOT) for gpu in (4,5,6)}
 submitted=set();futures={};failures=[];cancelled=set()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  while True:
   for v in VARIANTS:
    for s in SCENES:
     out=OUT/'quality'/v;folder=out/'runs'/s;key=(v,s)
     if (folder/'deferred_forward.json').exists() and key not in submitted and not (folder/'result.json').exists():
      futures[pool.submit(cpu_quality,out,plan_for(m,v),s)]=key;submitted.add(key)
   for future,key in list(futures.items()):
    if future.done():
     del futures[future]
     try:future.result()
     except Exception as exc:failures.append(dict(job=key,error=str(exc)))
   q=read(OUT/'queue.json');all_forward=all((OUT/'quality'/v/'runs'/s/'deferred_forward.json').exists() for v in VARIANTS for s in SCENES)
   if all_forward:
    for gpu,proc in processes.items():
     if proc.poll() is None and not (OUT/f'worker{gpu}_started.json').exists() and not psutil.Process(proc.pid).children():proc.terminate();cancelled.add(gpu)
   save(OUT/'progress.json',dict(queue=q,quality_active=list(futures.values()),complete=[(v,s) for v in VARIANTS for s in SCENES if (OUT/'quality'/v/'runs'/s/'result.json').exists()],failures=failures))
   if all(p.poll() is not None for p in processes.values()) and not futures:break
   time.sleep(2)
 for gpu,proc in processes.items():
  if proc.wait()!=0 and gpu not in cancelled:failures.append(dict(gpu=gpu,error='GPU worker failed; retained logs'))
 save(OUT/'execution.json',dict(failures=failures,cancelled_unused_waiters=list(cancelled)))
 if failures:raise RuntimeError(str(failures))
if __name__=='__main__':main()
