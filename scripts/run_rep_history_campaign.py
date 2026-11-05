"""Frozen A/B quality campaigns; staged calibration then complete ETH3D, GPU4/5/6."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/'outputs/eth3d_layout_fixed_full_20261105';ROOT_OUT=ROOT/'outputs/rep_history_loss_20261105'
VARIANTS=['baseline','A','B'];ALL=['facade','terrains','delivery_area','electro','courtyard','playground','kicker','office','relief_2','relief','pipes']
def read(p):return json.loads(p.read_text())
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(out,scenes):
 out.mkdir(parents=True,exist_ok=True);(out/'configs').mkdir(exist_ok=True)
 base=read(BASE/'frozen_config.json');configs={'baseline':base,'A':base|dict(rep_history_loss='A'),'B':base|dict(rep_history_loss='B')}
 for name,cfg in configs.items():save(out/'configs'/f'{name}.json',cfg)
 sources={str(p.relative_to(ROOT)):sha(p) for folder in ['saf3r','evaluation'] for p in (ROOT/folder).rglob('*.py')}
 for name in ['scripts/evaluate_zdrop_history_states.py','scripts/evaluate_saved_reuse_predictions.py','scripts/quality_metrics.py','scripts/launch_local_evaluation.py']:sources[name]=sha(ROOT/name)
 old_sources=read(BASE/'frozen_plan.json')['source_sha256'];protected=[k for k in old_sources if k.startswith(('saf3r/','evaluation/')) and k not in ['saf3r/patch/reuse_residual_exploration.py','saf3r/patch/reuse_window_graph.py']]
 assert all(sources.get(k)==v for k,v in old_sources.items() if k in protected),'Unexpected change to preexisting kernels/clustering/M5c/evaluation'
 m=dict(scenes=scenes,variants=VARIANTS,source_sha256=sources,configs={v:dict(config=str(out/'configs'/f'{v}.json'),config_sha256=sha(out/'configs'/f'{v}.json')) for v in VARIANTS},frozen_before_quality=True,base_reference=dict(path=str(BASE),macro=read(BASE/'summary.json')['macro']['layout_fixed']),single_factor='Only Rep-based History loss A/B; A union support/cost and count-based Rep residual weight are required semantic changes; B singleton only, original support/weights; camera-only special Query protection retained from explicit baseline',scope='ETH3D calibration3' if len(scenes)==3 else 'ETH3D full11,377 frames; unchanged terrace/meadow exclusion; max128 uniform seed0',gpus=[4,5,6],protected_source_files=protected,baseline_source_commit='8e1796530d4c49f04be8c3f4724bf378237992ac')
 save(out/'frozen_manifest.json',m)
 for v in VARIANTS:
  d=out/'quality'/v;(d/'inputs').mkdir(parents=True,exist_ok=True);(d/'runs').mkdir(exist_ok=True)
  for s in scenes:save(d/'inputs'/f'{s}.json',read(BASE/'runs'/s/'input.json'))
 save(out/'queue.json',dict(pending=[dict(variant=v,scene=s) for s in scenes for v in VARIANTS],claims=[]));return m
def plan(m,v):return dict(m['configs'][v],source_sha256=m['source_sha256'])
def worker(out,m,gpu):
 from scripts.run_eth3d_layout_fix_evaluation import verify
 save(out/f'worker{gpu}_started.json',dict(gpu=gpu,pid=os.getpid()))
 while True:
  with (out/'queue.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);q=read(out/'queue.json')
   if not q['pending']:return
   job=q['pending'].pop(0);job.update(gpu=gpu,status='forward');q['claims'].append(job);save(out/'queue.json',q)
  v,s=job['variant'],job['scene'];p=plan(m,v);verify(p);d=out/'quality'/v
  cmd=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset','eth3d','--scene',s,'--frames','128','--method','zdrop_history_states','--config',p['config'],'--expected-input',str(d/'inputs'/f'{s}.json'),'--output-dir',str(d/'runs'/s),'--warmup','0','--repeats','1','--defer-quality']
  print('START',gpu,v,s,flush=True)
  with (d/f'{s}_forward.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  with (out/'queue.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);q=read(out/'queue.json')
   for j in q['claims']:
    if j['variant']==v and j['scene']==s:j.update(status='done' if not code else 'failed',returncode=code)
   save(out/'queue.json',q)
  print('END',gpu,v,s,code,flush=True)
  if code:raise SystemExit(code)
def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',choices=['calibration','full'],default='calibration');p.add_argument('--worker',type=int,choices=[4,5,6]);a=p.parse_args()
 out=ROOT_OUT/a.scope;scenes=['pipes','courtyard','playground'] if a.scope=='calibration' else ALL
 m=read(out/'frozen_manifest.json') if (out/'frozen_manifest.json').exists() else prepare(out,scenes)
 if a.worker is not None:worker(out,m,a.worker);return
 from scripts.run_eth3d_layout_fix_evaluation import cpu_quality
 processes={gpu:subprocess.Popen([sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(out/f'worker{gpu}_gpu.json'),'--log',str(out/f'worker{gpu}.log'),'--',sys.executable,str(Path(__file__).resolve()),'--scope',a.scope,'--worker',str(gpu)],cwd=ROOT) for gpu in [4,5,6]}
 submitted=set();futures={};failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  while True:
   for v in VARIANTS:
    for s in scenes:
     d=out/'quality'/v;key=(v,s)
     if (d/'runs'/s/'deferred_forward.json').exists() and key not in submitted and not (d/'runs'/s/'result.json').exists():
      futures[pool.submit(cpu_quality,d,plan(m,v),s)]=key;submitted.add(key)
   for future,key in list(futures.items()):
    if future.done():
     del futures[future]
     try:future.result()
     except Exception as exc:failures.append(dict(job=key,error=str(exc)))
   complete=[(v,s) for v in VARIANTS for s in scenes if (out/'quality'/v/'runs'/s/'result.json').exists()]
   save(out/'progress.json',dict(complete=complete,quality_active=list(futures.values()),queue=read(out/'queue.json'),failures=failures))
   if all(proc.poll() is not None for proc in processes.values()) and not futures:
    # If the last child exited between the scan and poll, submit its saved prediction next iteration.
    ready=any((out/'quality'/v/'runs'/s/'deferred_forward.json').exists() and (v,s) not in submitted and not (out/'quality'/v/'runs'/s/'result.json').exists() for v in VARIANTS for s in scenes)
    if not ready:break
   time.sleep(2)
 for gpu,proc in processes.items():
  if proc.wait():failures.append(dict(gpu=gpu,error='GPU worker failed; logs retained'))
 save(out/'execution.json',dict(failures=failures))
 if failures:raise RuntimeError(str(failures))
 from scripts.report_rep_history_campaign import report_quality
 report_quality(out)
if __name__=='__main__':main()
