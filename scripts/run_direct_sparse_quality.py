"""Frozen Direct backend;full quality changed method only,GPUs4–7 exclusive."""
import concurrent.futures,hashlib,json,os,subprocess,sys,tarfile,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/direct_sparse_quality_20261104';SCENES={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 for name in ['direct_sparse_validation_v2_20261104','direct_legacy_validation_v3_20261104','direct_sparse_forward_validation_v2_20261104','direct_sparse_all_windows_20261104']:assert json.loads((ROOT/f'outputs/{name}/result.json').read_text())['complete'],name
 OUT.mkdir(exist_ok=False,parents=True);hashes={}
 for folder in ['saf3r','configs','evaluation','scripts','tests']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.toml']:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 save(OUT/'source_sha256.json',hashes)
 with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
  for name in sorted(hashes):archive.add(ROOT/name,arcname=name,recursive=False)
 save(OUT/'manifest.json',dict(method='v8_direct_sparse',base='c965ff76 plus phase commits801b66f0/9b59254b/20656a68 and isolated-observation optimizations',config=json.loads((ROOT/'configs/evaluation/reuse_v8_direct_sparse.json').read_text()),scenes=SCENES,gpus=[4,5,6,7],quality_tasks=['pose','recon_unposed'],quality_retuning=False,controls_executed=0,main_attention='Direct Triton Flash-style online softmax,no Grid argument/routing,compact IDs,original bias,32-query Groups',history='Original v8 D2+M5c Grid-conditionals;actual sparse P exports A/B are separate API experiments,not substituted into old prior',legacy_observation='Independent full-query Grid summary QK and compact original conditional-cache computation,explicitly counted in forward',coarse='One FP32 pooled-Key/probe QK,full-Key and patch-only normalization;old residual BF16 path promoted to shared FP32 precision',history_fast_shift='Cauchy bound<=32 ensures exp argument in[-64,0];otherwise original stable per-Grid fallback,no probability floor',tile=dict(keys=128,warps=4,stages=2),benchmark='heads128,warmup1+three complete forwards+separate profile',dense_speedup=None,dense_reason='No unchanged Dense full-forward rerun this campaign;prior Dense timings are historical only'))
 jobs=[(ds,s) for ds,ss in SCENES.items() for s in ss];lock=threading.Lock();done=[]
 env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1')
 def launch(gpu,label,cmd):return subprocess.call([sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(OUT/f'monitors/{label}.json'),'--log',str(OUT/f'logs/{label}.log'),'--',*cmd],cwd=ROOT,env=env)
 def worker(gpu):
  if gpu==7:
   code=launch(gpu,'timing',[sys.executable,str(ROOT/'scripts/benchmark_direct_sparse_forward.py'),'--config',str(ROOT/'configs/evaluation/reuse_v8_direct_sparse.json'),'--input',str(ROOT/'outputs/inference_efficiency_20261103/v8/quality/7scenes/heads/efficiency/input.json'),'--output-dir',str(OUT/'timing'),'--profile']);save(OUT/'timing_status.json',dict(returncode=code))
  while True:
   with lock:
    if not jobs:return
    ds,scene=jobs.pop(0)
   code=launch(gpu,f'{ds}_{scene}',[sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method','zdrop_history_states','--frames','128','--warmup','1','--repeats','1','--config',str(ROOT/'configs/evaluation/reuse_v8_direct_sparse.json'),'--output-dir',str(OUT/f'quality/{ds}/{scene}')])
   with lock:done.append(dict(dataset=ds,scene=scene,gpu=gpu,returncode=code));save(OUT/'progress.json',done)
   print(ds,scene,code,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,[4,5,6,7]))
 unchanged=all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items());save(OUT/'completion.json',dict(complete=len(done)==18 and all(x['returncode']==0 for x in done),source_unchanged=unchanged,runs=done))
if __name__=='__main__':main()
