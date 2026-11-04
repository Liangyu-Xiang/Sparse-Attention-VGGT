"""Two changed methods only,36 complete pose+reconstruction evaluations."""
import concurrent.futures,hashlib,json,os,subprocess,sys,tarfile,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sampled_history_quality_20261104'
SCENES={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}
INTERVALS=[32,256]
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 assert json.loads((ROOT/'outputs/sampled_history_integration_validation_v2_20261104/numerical/result.json').read_text())['complete']
 for interval in INTERVALS:
  path=ROOT/f'outputs/sampled_history_model_smoke{interval}_20261104/diagnostic';assert json.loads((path/'status.json').read_text())['stage']=='diagnostic_complete';assert len(json.loads((path/'layers.json').read_text()))==24
 OUT.mkdir(parents=True,exist_ok=False);hashes={}
 for folder in ['saf3r','configs','evaluation','scripts','tests']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.toml']:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 save(OUT/'source_sha256.json',hashes)
 with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
  for name in sorted(hashes):archive.add(ROOT/name,arcname=name,recursive=False)
 save(OUT/'manifest.json',dict(base='b0819004',intervals=INTERVALS,scenes=SCENES,gpus=[4,5,6,7],configs={str(i):json.loads((ROOT/f'configs/evaluation/reuse_sampled_history_{i}.json').read_text()) for i in INTERVALS},
  quality_tasks=['pose','recon_unposed'],quality_retuning=False,controls_executed=0,
  history='Inline debiased log sparse scores,one Probe4 in each sampled Group;compact sampled-only signed FP16 cache;M5c beta(1,2) and coarse missing-token completion',
  fallback='Unsampled Groups use their own coarse-uniform prior and own sparse support. No neighbor Key/probability copy. Residual summary uses own coarse patch probability and selected count/GridSize.',
  coarse='Four pooled-Key coarse Probes retained;history prior consistently uses Probe0. Unchanged retention20%,clusters,states,16-round budget,loss and mass transfer.',
  timing='Both intervals fresh processes on GPU7,same heads128 input,one warmup+three full forwards+separate profile',
  dense_speedup=None,dense_reason='Changed methods only;unchanged Dense not rerun. Historical Dense timings are not paired.'))
 jobs=[(interval,ds,scene) for ds,scenes in SCENES.items() for scene in scenes for interval in INTERVALS];done=[];lock=threading.Lock()
 env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1')
 def launch(gpu,label,cmd):return subprocess.call([sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(OUT/f'monitors/{label}.json'),'--log',str(OUT/f'logs/{label}.log'),'--',*cmd],cwd=ROOT,env=env)
 def failure(path,reason):
  from scripts.quality_metrics import quality_record,quality_markdown
  path.mkdir(parents=True,exist_ok=True);q=quality_record({},reason);save(path/'failure.json',dict(complete=False,quality=q));(path/'quality.md').write_text(quality_markdown(q))
 def worker(gpu):
  if gpu==7:
   for interval in INTERVALS:
    cfg=ROOT/f'configs/evaluation/reuse_sampled_history_{interval}.json';target=OUT/f'timing/{interval}'
    code=launch(gpu,f'timing_{interval}',[sys.executable,str(ROOT/'scripts/benchmark_direct_sparse_forward.py'),'--config',str(cfg),'--input',str(ROOT/'outputs/inference_efficiency_20261103/v8/quality/7scenes/heads/efficiency/input.json'),'--output-dir',str(target),'--profile']);save(OUT/f'timing_status_{interval}.json',dict(returncode=code))
    if code:failure(target,'Full-forward timing failed,return code '+str(code)+';see campaign log;no geometry evaluation')
  while True:
   with lock:
    if not jobs:return
    interval,ds,scene=jobs.pop(0)
   target=OUT/f'quality/{interval}/{ds}/{scene}';cfg=ROOT/f'configs/evaluation/reuse_sampled_history_{interval}.json'
   code=launch(gpu,f'{interval}_{ds}_{scene}',[sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method','zdrop_history_states','--frames','128','--warmup','1','--repeats','1','--config',str(cfg),'--output-dir',str(target)])
   if code:failure(target,'Quality process failed,return code '+str(code)+';see campaign log')
   with lock:done.append(dict(interval=interval,dataset=ds,scene=scene,gpu=gpu,returncode=code));save(OUT/'progress.json',done)
   print(interval,ds,scene,code,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,[4,5,6,7]))
 unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items());save(OUT/'completion.json',dict(complete=len(done)==36 and all(x['returncode']==0 for x in done),source_unchanged=unchanged,runs=done))
if __name__=='__main__':main()
