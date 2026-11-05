"""Corrected History Loss A+all-special Queries+12% clusters; changed quality only."""
import argparse,hashlib,json,os,queue,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(p.read_text())
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(16*1024*1024),b''):h.update(data)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;scope=read(out/'scope.json')
 checkpoint=Path(scope['checkpoint']);assert sha(checkpoint)==scope['checkpoint_sha256'];assert read(out/'semantic_source_audit.json')['passed']
 assert '17 passed' in (out/'validation.log').read_text();assert read(out/'residual_validation/result.json')['passed']
 cfg=ROOT/'configs/evaluation/history_a12_speed_20261106/optimized.json';reference=ROOT/'configs/evaluation/history_a12_speed_20261106/reference.json'
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig
 from dataclasses import asdict
 effective=asdict(ReuseExploreConfig(**read(cfg)));base=asdict(ReuseExploreConfig(**read(reference)))
 assert effective['rep_history_loss']=='A' and effective['dense_special_queries']==5 and effective['cluster_fraction']==.12
 diff={k:dict(before=base[k],after=v) for k,v in effective.items() if base[k]!=v}
 assert set(diff)=={'fused_rope','autotune_dense_convolutions','fused_residual_add','reuse_cuda_allocations','efficient_source_scatter','attention_stages'},diff
 hashes={str(x.relative_to(ROOT)):sha(x) for folder in ('saf3r','evaluation') for x in (ROOT/folder).rglob('*.py')};hashes[str(cfg.relative_to(ROOT))]=sha(cfg);hashes[str(reference.relative_to(ROOT))]=sha(reference)
 if (out/'plan.json').exists():raise ValueError('Fresh campaign manifest required')
 plan=dict(created_utc=datetime.now(timezone.utc).isoformat(),root=str(ROOT),base_commit='10e5656f',model_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),datasets=scope['datasets'],gpus=[4,5,6],source_sha256=hashes,config=str(cfg),reference_config=str(reference),effective_config=effective,engineering_diff=diff,checkpoint=str(checkpoint),checkpoint_sha256=scope['checkpoint_sha256'],warmup=1,repeats=3,frozen_before_quality=True,new_quality_runs=18,saf3r_timing_only_runs=18,dense_rerun=False,quality_tuning=False,historical_ETH3D_reference='rep_history_A_all_specials_20261105; RTX4090/Torch2.8, not same-hardware reproduction',quality_gate='Report absolute relative ETH3D six-metric changes against historical A_all_specials;pointcloud <2% separately. 7Scenes has no existing complete matched A_all_specials reference; do not substitute legacy-main results.')
 save(out/'plan.json',plan);tasks=queue.Queue()
 for ds,scenes in plan['datasets'].items():
  for sc in scenes:tasks.put((ds,sc))
 done=[];running={};lock=threading.Lock();env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONUNBUFFERED='1')
 def verify():
  bad=[k for k,v in hashes.items() if sha(ROOT/k)!=v]
  if bad:raise ValueError('Frozen model/config changed: '+str(bad))
 def status():save(out/'progress.json',dict(completed=done,running=running,total=36,updated_utc=datetime.now(timezone.utc).isoformat()))
 def worker(gpu):
  while True:
   try:ds,sc=tasks.get_nowait()
   except queue.Empty:return
   for name in ('saf3r','history_A_special5_cluster12_fast'):
    verify();target=out/'runs'/ds/sc/name;log=out/'logs'/f'{ds}_{sc}_{name}.log';monitor=out/'monitors'/f'{ds}_{sc}_{name}.json';config=reference if name=='saf3r' else cfg
    cmd=[sys.executable,str(ROOT/'scripts/run_profiler_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(monitor),'--log',str(log),'--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',sc,'--method','saf3r' if name=='saf3r' else 'zdrop_history_states','--frames','128','--config',str(config),'--output-dir',str(target),'--warmup','1','--repeats','3','--expected-input',str(out/'expected_inputs'/ds/(sc+'.json'))]
    if name=='saf3r':cmd.append('--timing-only')
    row=dict(dataset=ds,scene=sc,method=name,gpu=gpu,output=str(target),command=cmd)
    save(out/'commands'/f'{ds}_{sc}_{name}.json',row)
    with lock:running[str(gpu)]=row;status()
    print('START',ds,sc,name,gpu,flush=True);begin=time.monotonic();code=subprocess.call(cmd,cwd=ROOT,env=env);row.update(returncode=code,wall_seconds=time.monotonic()-begin)
    result=target/('timing_result.json' if name=='saf3r' else 'result.json');row['complete']=result.exists() and (name=='saf3r' or read(result)['quality']['complete'])
    with lock:done.append(row);running.pop(str(gpu),None);status()
    print('END',ds,sc,name,code,row['complete'],flush=True)
    if code or not row['complete']:raise RuntimeError('Run failed: '+str(target))
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,(4,5,6)))
 verify();save(out/'completion.json',dict(complete=len(done)==36 and all(x['complete'] for x in done),source_unchanged=True,runs=done));print('COMPLETE',len(done),flush=True)
if __name__=='__main__':main()
