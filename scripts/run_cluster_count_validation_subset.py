"""Default fixed-subset validation:three variants x four scenes;sampler32 and retention20% fixed."""
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,sys,tarfile,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cluster_count_quality_20261104'
SCENES=json.loads((ROOT/'configs/evaluation/validation_subset.json').read_text())['scenes']
INTERVALS=[15,10,5]
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 global OUT
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpus',help='Comma-separated physical GPU IDs;default:all currently idle GPUs');args=parser.parse_args();OUT=args.output_dir.resolve()
 if args.gpus:gpus=[int(x) for x in args.gpus.split(',')]
 else:
  query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
  gpus=[int(line.split(',')[0]) for line in query.splitlines() if int(line.split(',')[1])<100]
 if not gpus:raise RuntimeError('No idle GPUs;choose a known available card or retry later')
 timing_gpu=gpus[-1];expected=len(INTERVALS)*sum(map(len,SCENES.values()))
 assert json.loads((ROOT/'outputs/cluster_count_validation_20261104/result.json').read_text())['complete']
 for interval in INTERVALS:
  path=ROOT/f'outputs/cluster_count_smoke_20261104/{interval}/diagnostic';assert json.loads((path/'status.json').read_text())['stage']=='diagnostic_complete';assert len(json.loads((path/'layers.json').read_text()))==24
 OUT.mkdir(parents=True,exist_ok=False);hashes={}
 for folder in ['saf3r','configs','evaluation','scripts','tests']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.toml']:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 save(OUT/'source_sha256.json',hashes)
 with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
  for name in sorted(hashes):archive.add(ROOT/name,arcname=name,recursive=False)
 save(OUT/'manifest.json',dict(base='c85543dbc365c16a746829731a53e374aae14020',cluster_percentages=INTERVALS,intervals=INTERVALS,history_query_interval=32,attention_retention=.2,scenes=SCENES,gpus=gpus,timing_gpu=timing_gpu,scope='fixed difficult-sequence validation subset',wide_cluster_fix='History tile unchanged through width128;tile elements capped8192 for widths256/512;exact geometry/drop bins extended to512 without repartitioning clusters',configs={str(i):json.loads((ROOT/f'configs/evaluation/reuse_sampled32_clusters_{i}.json').read_text()) for i in INTERVALS},
  quality_tasks=['pose','recon_unposed'],quality_retuning=False,controls_executed=0,
  history='Inline debiased log sparse scores,one Probe4 in each sampled Group;compact sampled-only signed FP16 cache;M5c beta(1,2) and coarse missing-token completion',
  fallback='Unsampled Groups use their own coarse-uniform prior and own sparse support. No neighbor Key/probability copy. Residual summary uses own coarse patch probability and selected count/GridSize.',
  coarse='Four pooled-Key coarse Probes retained;history prior uses Probe0. Only cluster_fraction changes;initial layer preserves budget with deterministic extra real members and conservative rep weights. States/16-round budget/loss/mass transfer remain fixed.',
  timing=f'All fractions fresh processes on physical GPU{timing_gpu},same heads128 timing-only input,one warmup+three full forwards+separate profile',
  dense_speedup=None,dense_reason='Changed methods only;unchanged Dense not rerun. Historical Dense timings are not paired.'))
 jobs=[(interval,ds,scene) for ds,scenes in SCENES.items() for scene in scenes for interval in INTERVALS];done=[];lock=threading.Lock()
 env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1')
 def launch(gpu,label,cmd):return subprocess.call([sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(OUT/f'monitors/{label}.json'),'--log',str(OUT/f'logs/{label}.log'),'--',*cmd],cwd=ROOT,env=env)
 def failure(path,reason):
  from scripts.quality_metrics import quality_record,quality_markdown
  path.mkdir(parents=True,exist_ok=True);q=quality_record({},reason);save(path/'failure.json',dict(complete=False,quality=q));(path/'quality.md').write_text(quality_markdown(q))
 def worker(gpu):
  if gpu==timing_gpu:
   for interval in INTERVALS:
    cfg=ROOT/f'configs/evaluation/reuse_sampled32_clusters_{interval}.json';target=OUT/f'timing/{interval}'
    code=launch(gpu,f'timing_{interval}',[sys.executable,str(ROOT/'scripts/benchmark_direct_sparse_forward.py'),'--config',str(cfg),'--input',str(ROOT/'outputs/inference_efficiency_20261103/v8/quality/7scenes/heads/efficiency/input.json'),'--output-dir',str(target),'--profile']);save(OUT/f'timing_status_{interval}.json',dict(returncode=code))
    if code:failure(target,'Full-forward timing failed,return code '+str(code)+';see campaign log;no geometry evaluation')
  while True:
   with lock:
    if not jobs:return
    interval,ds,scene=jobs.pop(0)
   target=OUT/f'quality/{interval}/{ds}/{scene}';cfg=ROOT/f'configs/evaluation/reuse_sampled32_clusters_{interval}.json'
   code=launch(gpu,f'{interval}_{ds}_{scene}',[sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method','zdrop_history_states','--frames','128','--warmup','1','--repeats','1','--config',str(cfg),'--output-dir',str(target)])
   if code:failure(target,'Quality process failed,return code '+str(code)+';see campaign log')
   with lock:done.append(dict(interval=interval,dataset=ds,scene=scene,gpu=gpu,returncode=code));save(OUT/'progress.json',done)
   print(interval,ds,scene,code,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:list(pool.map(worker,gpus))
 unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items());save(OUT/'completion.json',dict(complete=len(done)==expected and all(x['returncode']==0 for x in done),source_unchanged=unchanged,runs=done))
if __name__=='__main__':main()
