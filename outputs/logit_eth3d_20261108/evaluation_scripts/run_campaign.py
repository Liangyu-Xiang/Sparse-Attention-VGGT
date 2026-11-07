"""Run only the changed candidate on frozen ETH3D inputs, GPU4/5/6."""
import concurrent.futures,json,os,queue,subprocess,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/logit_eth3d_20261108/eth3d'
PYTHON='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/bin/python'
MONITOR=Path('/tmp/sparse-vggt-history-teacher-20261106/scripts/run_profiler_gpu_exclusive.py')
def main():
 variant=json.loads((OUT.parent/'selection.json').read_text())['selected'];tasks=queue.Queue()
 for scene in ['facade','terrains','delivery_area','electro','courtyard','playground','kicker','office','relief_2','relief','pipes']:tasks.put(scene)
 lock=threading.Lock();done=[];running={}
 env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1')
 def status():
  temp=OUT/'progress.tmp';temp.write_text(json.dumps(dict(completed=done,running=running),indent=2));temp.replace(OUT/'progress.json')
 def worker(gpu):
  while True:
   try:scene=tasks.get_nowait()
   except queue.Empty:return
   folder=OUT/'runs'/scene
   command=[PYTHON,str(MONITOR),'--gpu',str(gpu),'--monitor',str(OUT/'monitors'/f'{scene}.json'),'--log',str(OUT/'logs'/f'{scene}.log'),'--',PYTHON,'-u',str(ROOT/'experiments/logit_eth3d/evaluate.py'),'--variant',variant,'--scene',scene,'--expected-input',str(OUT/'expected_inputs'/f'{scene}.json'),'--config',str(OUT/'config.json'),'--output',str(folder)]
   with lock:running[str(gpu)]=scene;status()
   start=time.monotonic();code=subprocess.call(command,cwd=ROOT,env=env)
   result=folder/'result.json';complete=result.exists() and json.loads(result.read_text())['quality']['complete']
   with lock:done.append(dict(scene=scene,gpu=gpu,returncode=code,complete=complete,wall_seconds=time.monotonic()-start));running.pop(str(gpu));status()
   print('DONE',scene,code,complete,flush=True)
   if code or not complete:raise RuntimeError('Incomplete run: '+scene)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,[4,5,6]))
 (OUT/'campaign_complete.json').write_text(json.dumps(dict(complete=all(x['complete'] for x in done),runs=done),indent=2))
if __name__=='__main__':main()
