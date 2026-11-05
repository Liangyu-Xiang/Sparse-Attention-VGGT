"""Frozen user-requested 10%/25% retention, full matched ETH3D quality and camera features."""
import argparse,concurrent.futures,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/rep_history_A_cluster20_early10_late25_20261106'
BASE=ROOT/'outputs/rep_history_A_all_specials_cluster20_20261105'
CONFIG=ROOT/'configs/evaluation/reuse_rep_history_A_all_specials_cluster20_early10_late25.json'
def read(p):return json.loads(p.read_text())
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 assert not OUT.exists();cfg=read(ROOT/'configs/evaluation/reuse_rep_history_A_all_specials_cluster20.json');cfg['layer_retentions']=[.10]*6+[.25]*18;save(CONFIG,cfg)
 old=read(BASE/'frozen_manifest.json');names=set(old['source_sha256'])|{'scripts/evaluate_camera_retention.py','scripts/run_camera_retention10_campaign.py','scripts/camera_token_diagnostic.py','scripts/evaluate_zdrop_history_states.py','scripts/evaluate_saved_reuse_predictions.py','scripts/launch_local_evaluation.py','scripts/quality_metrics.py'}
 sources={n:sha(ROOT/n) for n in names}
 assert all(sources[k]==v for k,v in old['source_sha256'].items()),'Native source must match existing20% cluster baseline'
 scenes=read(ROOT/'outputs/camera_token_comparison_20261105/frozen_plan.json')['scenes'];sources[str(CONFIG.relative_to(ROOT))]=sha(CONFIG)
 plan=dict(scenes=scenes,config=str(CONFIG),config_sha256=sha(CONFIG),source_sha256=sources,method='A_all_specials_cluster20_early10_late25',gpus=[4,5,6],single_change='Only layer_retentions: [0.10]*6+[0.25]*18; all-specials,A history,cluster fraction0.20,all kernels frozen',sampling='ETH3D existing filtered11,377 frames,max128 uniform seed0; terrace/meadow excluded by unchanged protocol',dense_reference='outputs/camera_token_comparison_20261105/full/*/dense/camera_tensors.pt',controls='Reuse existing Dense features and original20% cluster quality; no unchanged control rerun')
 save(OUT/'frozen_plan.json',plan)
 for s in scenes:save(OUT/'inputs'/f'{s}.json',read(BASE/'quality/A_all_specials_cluster20/inputs'/f'{s}.json'))
 save(OUT/'queue.json',dict(pending=scenes,claims=[]))
def worker(gpu):
 from scripts.run_eth3d_layout_fix_evaluation import verify
 plan=read(OUT/'frozen_plan.json');verify(plan)
 while True:
  with (OUT/'queue.lock').open('w') as f:
   fcntl.flock(f,fcntl.LOCK_EX);q=read(OUT/'queue.json')
   if not q['pending']:return
   s=q['pending'].pop(0);q['claims'].append(dict(scene=s,gpu=gpu,status='forward'));save(OUT/'queue.json',q)
  cmd=[sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_camera_retention.py'),'--dataset','eth3d','--scene',s,'--frames','128','--method','zdrop_history_states','--config',str(CONFIG),'--expected-input',str(OUT/'inputs'/f'{s}.json'),'--output-dir',str(OUT/'runs'/s),'--warmup','0','--repeats','1','--defer-quality']
  print('START',gpu,s,flush=True)
  with (OUT/f'{s}_forward.log').open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  with (OUT/'queue.lock').open('w') as f:
   fcntl.flock(f,fcntl.LOCK_EX);q=read(OUT/'queue.json');next(c for c in q['claims'] if c['scene']==s).update(status='complete' if not code else 'failed',returncode=code);save(OUT/'queue.json',q)
  print('END',gpu,s,code,flush=True)
  if code:raise SystemExit(code)
def main():
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[4,5,6]);a=p.parse_args()
 if not (OUT/'frozen_plan.json').exists():prepare()
 plan=read(OUT/'frozen_plan.json')
 if a.worker is not None:worker(a.worker);return
 from scripts.run_eth3d_layout_fix_evaluation import cpu_quality
 procs={g:subprocess.Popen([sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(g),'--monitor',str(OUT/f'worker{g}_gpu.json'),'--log',str(OUT/f'worker{g}.log'),'--',sys.executable,str(Path(__file__).resolve()),'--worker',str(g)],cwd=ROOT) for g in [4,5,6]}
 submitted=set();futures={};failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  while True:
   for s in plan['scenes']:
    if (OUT/'runs'/s/'deferred_forward.json').exists() and s not in submitted and not (OUT/'runs'/s/'result.json').exists():futures[pool.submit(cpu_quality,OUT,plan,s)]=s;submitted.add(s)
   for future,s in list(futures.items()):
    if future.done():
     del futures[future]
     try:future.result()
     except Exception as exc:failures.append(dict(scene=s,error=str(exc)))
   save(OUT/'progress.json',dict(quality_complete=[s for s in plan['scenes'] if (OUT/'runs'/s/'result.json').exists()],camera_complete=[s for s in plan['scenes'] if (OUT/'runs'/s/'camera/result.json').exists()],quality_active=list(futures.values()),queue=read(OUT/'queue.json'),failures=failures))
   if all(p.poll() is not None for p in procs.values()) and not futures:
    if not any((OUT/'runs'/s/'deferred_forward.json').exists() and s not in submitted and not (OUT/'runs'/s/'result.json').exists() for s in plan['scenes']):break
   time.sleep(2)
 for g,p in procs.items():
  if p.wait():failures.append(dict(gpu=g,error='GPU worker failed'))
 save(OUT/'execution.json',dict(failures=failures))
 if failures:raise RuntimeError(str(failures))
 from scripts.report_camera_retention10_campaign import report
 report()
if __name__=='__main__':main()
