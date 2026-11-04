"""Frozen changed-method campaign. One CPU-only exclusive lease per GPU."""
import concurrent.futures,hashlib,json,os,subprocess,sys,tarfile,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/direct_probability_quality_20261104'
SCENES={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}
CONFIG=ROOT/'configs/evaluation/reuse_direct_probability.json'
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 assert json.loads((ROOT/'outputs/direct_probability_validation_v3_20261104/numerical/result.json').read_text())['complete']
 assert json.loads((ROOT/'outputs/direct_probability_pilot_v2_20261104/timing/result.json').read_text())['complete']
 OUT.mkdir(exist_ok=False,parents=True);hashes={}
 for folder in ['saf3r','configs','evaluation','scripts','tests']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.toml']:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 save(OUT/'source_sha256.json',hashes)
 with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
  for name in sorted(hashes):archive.add(ROOT/name,arcname=name,recursive=False)
 save(OUT/'manifest.json',dict(method='direct_probability_history',base='6c9ca01c',config=json.loads(CONFIG.read_text()),scenes=SCENES,gpus=[4,5,6,7],quality_tasks=['pose','recon_unposed'],quality_retuning=False,controls_executed=0,
  history='log mean_h(exp(raw_QK-final_compensated_sparse_LSE)); shared assignment bias removed before exp; coarse missing-member scores in same sparse-LSE scale; per-Grid completion, M5c beta(1,2), signed FP16 log conditionals',
  changed_semantics=['ratio of head-mean scores instead of mean of per-head Grid conditionals','4-probe rather than all32-query residual Grid summary'],
  unchanged='retention20%,adaptive-local clustering,Drop/Rep/History/Exact,16-round budget,state loss,mass-transfer bias,checkpoint,frames and quality protocol',
  extra_qk='4 probes only including Specials; no legacy all-query observation, no second selected-key cache QK; current coarse pooled-Key QK retained',
  dense_speedup=None,dense_reason='Changed methods only; unchanged Dense not rerun. Historical timings are not paired speed comparisons.'))
 jobs=[(ds,s) for ds,ss in SCENES.items() for s in ss];lock=threading.Lock();done=[]
 env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1')
 def launch(gpu,label,cmd):return subprocess.call([sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(OUT/f'monitors/{label}.json'),'--log',str(OUT/f'logs/{label}.log'),'--',*cmd],cwd=ROOT,env=env)
 def unavailable(path,reason):
  from scripts.quality_metrics import quality_record,quality_markdown
  path.mkdir(parents=True,exist_ok=True);q=quality_record({},reason);save(path/'failure.json',dict(complete=False,quality=q));(path/'quality.md').write_text(quality_markdown(q))
 def worker(gpu):
  if gpu==7:
   code=launch(gpu,'timing',[sys.executable,str(ROOT/'scripts/benchmark_direct_sparse_forward.py'),'--config',str(CONFIG),'--input',str(ROOT/'outputs/inference_efficiency_20261103/v8/quality/7scenes/heads/efficiency/input.json'),'--output-dir',str(OUT/'timing'),'--profile']);save(OUT/'timing_status.json',dict(returncode=code))
   if code:unavailable(OUT/'timing','Timing process failed, return code '+str(code)+'; see timing log; no quality evaluation')
  while True:
   with lock:
    if not jobs:return
    ds,scene=jobs.pop(0)
   target=OUT/f'quality/{ds}/{scene}'
   code=launch(gpu,f'{ds}_{scene}',[sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method','zdrop_history_states','--frames','128','--warmup','1','--repeats','1','--config',str(CONFIG),'--output-dir',str(target)])
   if code:unavailable(target,'Quality process failed, return code '+str(code)+'; see corresponding campaign log')
   with lock:done.append(dict(dataset=ds,scene=scene,gpu=gpu,returncode=code));save(OUT/'progress.json',done)
   print(ds,scene,code,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,[4,5,6,7]))
 unchanged=all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items());save(OUT/'completion.json',dict(complete=len(done)==18 and all(x['returncode']==0 for x in done),source_unchanged=unchanged,runs=done))
if __name__=='__main__':main()
