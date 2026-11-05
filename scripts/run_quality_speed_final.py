"""Only frozen changed methods: matched complete pose+reconstruction evaluations."""
import argparse,hashlib,json,os,queue,subprocess,sys,threading,time,shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save
BASE=Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs/history_A_special5_cluster12_speed_20261106')
def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();f=json.loads(a.freeze.read_text());assert f['frozen_before_target_quality'] and (a.resume or not a.output.exists());out=a.output;out.mkdir(parents=True,exist_ok=a.resume)
 scope=json.loads((BASE/'scope.json').read_text())
 for ds,scenes in scope['datasets'].items():
  for sc in scenes:
   expected=json.loads((BASE/'expected_inputs'/ds/(sc+'.json')).read_text())
   assert expected['image_paths'] and all((ROOT/path).is_file() for path in expected['image_paths']),f'Missing frozen RGB files: {ds}/{sc}'
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['saf3r','evaluation'] for p in (ROOT/folder).rglob('*.py')}
 plan=dict(freeze=f,datasets=scope['datasets'],source_sha256=hashes,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gpus=[4,5,6],warmup=1,repeats=3,dense_rerun=False,saf3r_rerun=False,timing_reference_batch=BASE.name,selection_uses_test_gt=False)
 if a.resume:
  previous=json.loads((out/'plan.json').read_text());assert previous['freeze']==f and previous['source_sha256']==hashes and previous['datasets']==scope['datasets']
  save(out/'resumption.json',dict(reason='RFS ENOSPC before inference;output relocated to verified local copy;reuse every complete scene,archive incomplete records only',original_plan_unchanged=True,source_unchanged=True,config_unchanged=True))
 else:save(out/'plan.json',plan)
 tasks=queue.Queue();done=[];total=0
 for candidate in f['candidates']:
  for ds,scenes in scope['datasets'].items():
   for scene in scenes:
    total+=1;name=candidate['name'];target=out/'runs'/name/ds/scene;result=target/'result.json';key=f'{name}_{ds}_{scene}';monitor=out/'monitors'/(key+'.json');log=out/'logs'/(key+'.log')
    if a.resume and result.exists() and json.loads(result.read_text())['quality']['complete']:
     r=json.loads(result.read_text());m=json.loads(monitor.read_text());expected=json.loads((BASE/'expected_inputs'/ds/(scene+'.json')).read_text());cfg=json.loads((ROOT/candidate['config']).read_text())
     assert all(r['input'][k]==expected[k] for k in ('image_paths','frame_indices','shape','checkpoint_sha256')) and all(r['config'][k]==v for k,v in cfg.items())
     assert m['returncode']==0 and not m['same_gpu_overlap_detected'];done.append(dict(name=name,dataset=ds,scene=scene,output=str(target),complete=True,returncode=0,reused_complete=True));continue
    if a.resume:
     failure=out/'failures'/'rfs_quota_attempt'/key
     for old in [target,monitor,log]:
      if old.exists():failure.mkdir(parents=True,exist_ok=True);shutil.move(str(old),str(failure/old.name))
     if failure.exists():
      from scripts.quality_metrics import quality_record
      save(failure/'quality.json',quality_record({},missing_reason='Incomplete infrastructure attempt: RFS ENOSPC or controller stopped to relocate outputs;no completed quality score retained'))
    tasks.put((candidate,ds,scene))
 running={};lock=threading.Lock();env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONUNBUFFERED='1')
 def status():save(out/'progress.json',dict(completed=done,running=running,total=total))
 def worker(gpu):
  while True:
   try:c,ds,scene=tasks.get_nowait()
   except queue.Empty:return
   for path,sha in hashes.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
   cfg=ROOT/c['config'];assert hashlib.sha256(cfg.read_bytes()).hexdigest()==c['config_sha256']
   name=c['name'];target=out/'runs'/name/ds/scene;key=f'{name}_{ds}_{scene}';monitor=out/'monitors'/(key+'.json');log=out/'logs'/(key+'.log')
   cmd=[sys.executable,str(ROOT/'scripts/run_profiler_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(monitor),'--log',str(log),'--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method','zdrop_history_states','--frames','128','--config',str(cfg),'--output-dir',str(target),'--warmup','1','--repeats','3','--expected-input',str(BASE/'expected_inputs'/ds/(scene+'.json'))]
   row=dict(name=name,dataset=ds,scene=scene,gpu=gpu,output=str(target),command=cmd)
   with lock:running[str(gpu)]=row;status()
   print('START',name,ds,scene,gpu,flush=True);start=time.monotonic();code=subprocess.call(cmd,cwd=ROOT,env=env);result=target/'result.json';complete=result.exists() and json.loads(result.read_text())['quality']['complete'];row.update(returncode=code,wall_seconds=time.monotonic()-start,complete=complete)
   with lock:done.append(row);running.pop(str(gpu),None);status()
   print('END',name,ds,scene,code,complete,flush=True)
   if code or not complete:raise RuntimeError('Incomplete changed-method run: '+key)
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,[4,5,6]))
 save(out/'completion.json',dict(complete=len(done)==total and all(r['complete'] for r in done),runs=done));print('COMPLETE',total,flush=True)
if __name__=='__main__':main()
