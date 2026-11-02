"""Frozen ready-history quality on both datasets; reuse unchanged controls."""
import concurrent.futures as cf
import hashlib,json,os,subprocess,sys,threading,tarfile,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/no_mass_compensation_quality_v2_20261102'
SNAP=OUT/'source'
SCENES={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def main():
 if OUT.exists():raise ValueError('Fresh output required')
 OUT.mkdir()
 subprocess.run([sys.executable,str(ROOT/'scripts/setup_no_mass_compensation.py'),'--destination',str(SNAP)],check=True)
 assert json.loads((ROOT/'outputs/headmean_single_qk_validation_20261102/validation.json').read_text())['complete']
 references={}
 for ds,scenes in SCENES.items():
  references[ds]={}
  for s in scenes:
   if ds=='7scenes':r={'dense':ROOT/'outputs/reuse_compact_7scenes128_20261029'/s/'dense/result.json','saf3r':ROOT/'outputs/saf3r_matched_reuse_comparison_20261029'/s/'saf3r_lazy4/result.json'}
   else:r={m:ROOT/'outputs/reuse_relaxed_saf3r_eval_20261101/eth3d'/s/m/'result.json' for m in ['dense','saf3r']}
   r['ready_previous']=ROOT/'outputs/headmean_single_qk_quality_20261102'/ds/s/'headmean_single_qk/result.json'
   for f in r.values():assert json.loads(f.read_text())['quality']['complete']
   references[ds][s]={m:str(f) for m,f in r.items()}
 allhash={str(p.relative_to(SNAP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SNAP.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ['.py','.json','.yaml','.yml']}
 save(OUT/'manifest.json',dict(method='no_mass_compensation',datasets=SCENES,references=references,source_sha256=allhash,source_root=str(SNAP),config=json.loads((SNAP/'configs/evaluation/reuse_zdrop_history_states.json').read_text()),frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controls_executed=0,no_quality_retuning=True,quality_tasks=['pose','recon_unposed'],timing=dict(warmup=1,repeats=3,dense_speedup=None,reason='Unchanged controls reused; no contemporaneous Dense timing')))
 with tarfile.open(OUT/'source_snapshot.tar','w') as t:
  for n in allhash:t.add(SNAP/n,arcname=n,recursive=False)
 queue=[(d,s) for d,ss in SCENES.items() for s in ss];lock=threading.Lock();results=[]
 def worker(gpu):
  while True:
   with lock:
    if not queue:return
    ds,s=queue.pop(0)
   folder=OUT/ds/s/'no_mass_compensation'
   cmd=[sys.executable,'-u',str(ROOT/'scripts/run_gpu_lease.py'),'--gpu',str(gpu),'--monitor',str(folder.parent/'lease_monitor.json'),'--log',str(OUT/'logs'/f'{ds}_{s}.log'),'--',sys.executable,'-u',str(SNAP/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--method','zdrop_history_states','--scene',s,'--frames','128','--warmup','1','--repeats','3','--config',str(SNAP/'configs/evaluation/reuse_zdrop_history_states.json'),'--output-dir',str(folder)]
   print('START',ds,s,gpu,flush=True)
   code=subprocess.call(cmd,cwd=SNAP)
   with lock:results.append(dict(dataset=ds,scene=s,gpu=gpu,returncode=code));save(OUT/'progress.json',results)
   print('DONE',ds,s,code,flush=True)
 with cf.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
 save(OUT/'completion.json',dict(runs=results,all_processes_succeeded=len(results)==18 and all(r['returncode']==0 for r in results)))
 subprocess.run([sys.executable,str(ROOT/'scripts/report_no_mass_compensation_quality.py')],check=True)
if __name__=='__main__':main()
