"""ETH3D 2x2 decomposition: estimator change versus attention execution change."""
import concurrent.futures as cf,datetime,hashlib,json,shutil,subprocess,sys,threading,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/headmean_execution_causal_20261102'
SCENES=['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 if OUT.exists():raise ValueError('Fresh output required')
 OUT.mkdir();parents={k:ROOT/'outputs'/v for k,v in {'perhead_two_qk':'ready_history_quality_20261102','headmean_one_qk':'headmean_single_qk_quality_20261102'}.items()}
 sources={};hashes={}
 for variant,parentkey in [('headmean_two_qk','perhead_two_qk'),('perhead_one_qk','headmean_one_qk')]:
  parent=parents[parentkey];m=json.loads((parent/'manifest.json').read_text());base=Path(m['source_root']);assert all(hashlib.sha256((base/n).read_bytes()).hexdigest()==h for n,h in m['source_sha256'].items())
  target=OUT/'source'/variant;shutil.copytree(base,target,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  p=target/'saf3r/patch/reuse_residual_exploration.py';s=p.read_text()
  if variant=='headmean_two_qk':
   for n in ['saf3r/patch/reuse_headmean_history.py','saf3r/triton/reuse_headmean_history.py']:shutil.copy2(parents['headmean_one_qk']/'source'/n,target/n)
   s=s.replace('from . import reuse_ready_history as m5c','from . import reuse_headmean_history as m5c').replace('history_estimator_head_aware=True','history_estimator_head_aware=False').replace("history_estimator='Ready D2 conditionals + M5c'","history_estimator='Head-mean D2 conditionals + M5c; original two-QK backend'")
  else:
   s=s.replace('from . import reuse_headmean_history as m5c','from . import reuse_ready_history as m5c').replace('history_estimator_head_aware=False','history_estimator_head_aware=True').replace("history_estimator='Head-mean conditionals + head-mean M5c'","history_estimator='Original per-head D2 conditionals + M5c; single-QK backend'")
  p.write_text(s)
  p=target/'scripts/evaluate_zdrop_history_states.py';s=p.read_text().replace('names=set(sources)|{',"names=set(sources)|{'saf3r/patch/reuse_headmean_history.py','saf3r/triton/reuse_headmean_history.py',") if variant=='headmean_two_qk' else p.read_text();p.write_text(s)
  sources[variant]=str(target);hashes[variant]={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ['.py','.json','.yaml','.yml']}
  with tarfile.open(OUT/(variant+'_source.tar'),'w') as archive:
   for n in hashes[variant]:archive.add(target/n,arcname=n,recursive=False)
 refs={scene:{'perhead_two_qk':str(parents['perhead_two_qk']/'eth3d'/scene/'ready_history_v5/result.json'),'headmean_one_qk':str(parents['headmean_one_qk']/'eth3d'/scene/'headmean_single_qk/result.json')} for scene in SCENES}
 save(OUT/'manifest.json',dict(scenes=SCENES,dataset='eth3d',sources=sources,source_sha256=hashes,references=refs,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),changed_methods_executed=list(sources),controls_reused_only=True,quality_tasks=['pose','recon_unposed'],quality_once_per_scene=True,no_quality_retuning=True,requested_frames=128,timing=dict(warmup=1,repeats=3,dense_speedup=None,reason='No Dense rerun'),causal_design='2x2: perhead/headmean estimator x original two-QK/physically grid-grouped single-QK; no compensation ablation mixed in'))
 queue=[(v,s) for s in SCENES for v in sources];lock=threading.Lock();results=[]
 def worker(gpu):
  while True:
   with lock:
    if not queue:return
    variant,scene=queue.pop(0)
   src=Path(sources[variant]);folder=OUT/'eth3d'/scene/variant
   cmd=[sys.executable,'-u',str(ROOT/'scripts/run_gpu_lease.py'),'--gpu',str(gpu),'--monitor',str(folder.parent/(variant+'_monitor.json')),'--log',str(OUT/'logs'/f'{scene}_{variant}.log'),'--',sys.executable,'-u',str(src/'scripts/evaluate_zdrop_history_states.py'),'--dataset','eth3d','--method','zdrop_history_states','--scene',scene,'--frames','128','--warmup','1','--repeats','3','--config',str(src/'configs/evaluation/reuse_zdrop_history_states.json'),'--output-dir',str(folder)]
   print('START',variant,scene,gpu,flush=True);code=subprocess.call(cmd,cwd=src)
   with lock:results.append(dict(variant=variant,scene=scene,gpu=gpu,returncode=code));save(OUT/'progress.json',results)
   print('DONE',variant,scene,code,flush=True)
 with cf.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
 save(OUT/'completion.json',dict(runs=results,all_processes_succeeded=len(results)==22 and all(r['returncode']==0 for r in results)))
 subprocess.run([sys.executable,str(ROOT/'scripts/report_headmean_execution_causal.py')],check=True)
if __name__=='__main__':main()
