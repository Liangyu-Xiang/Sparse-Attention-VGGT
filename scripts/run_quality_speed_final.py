"""Only frozen changed methods: matched complete pose+reconstruction evaluations."""
import argparse,hashlib,json,os,queue,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.search_quality_speed import save
BASE=Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs/history_A_special5_cluster12_speed_20261106')
def main():
 p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();f=json.loads(a.freeze.read_text());assert f['frozen_before_target_quality'] and not a.output.exists();out=a.output;out.mkdir(parents=True)
 scope=json.loads((BASE/'scope.json').read_text())
 for ds,scenes in scope['datasets'].items():
  for sc in scenes:
   expected=json.loads((BASE/'expected_inputs'/ds/(sc+'.json')).read_text())
   assert expected['image_paths'] and all((ROOT/path).is_file() for path in expected['image_paths']),f'Missing frozen RGB files: {ds}/{sc}'
 hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['saf3r','evaluation'] for p in (ROOT/folder).rglob('*.py')}
 plan=dict(freeze=f,datasets=scope['datasets'],source_sha256=hashes,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),gpus=[4,5,6],warmup=1,repeats=3,dense_rerun=False,saf3r_rerun=False,timing_reference_batch=BASE.name,selection_uses_test_gt=False);save(out/'plan.json',plan)
 tasks=queue.Queue()
 for candidate in f['candidates']:
  for ds,scenes in scope['datasets'].items():
   for scene in scenes:tasks.put((candidate,ds,scene))
 total=tasks.qsize();done=[];running={};lock=threading.Lock();env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONUNBUFFERED='1')
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
