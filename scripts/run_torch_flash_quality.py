"""Frozen modified-method quality only; no unchanged controls rerun."""
import concurrent.futures,hashlib,json,os,subprocess,sys,tarfile,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/torch_flash_quality_20261104';SCENES={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 for name in ['torch_flash_validation_v3_20261104','torch_flash_forward_validation_v3_20261104']:assert json.loads((ROOT/f'outputs/{name}/result.json').read_text())['complete'],name
 OUT.mkdir(exist_ok=False,parents=True);hashes={}
 for folder in ['saf3r','configs','evaluation','scripts','tests']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml','.toml']:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 save(OUT/'source_sha256.json',hashes)
 with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
  for name in sorted(hashes):archive.add(ROOT/name,arcname=name,recursive=False)
 save(OUT/'manifest.json',dict(method='v8_torch_flash',base='6285125b,original frozen v8 with P0 disabled',config=json.loads((ROOT/'configs/evaluation/reuse_v8_torch_flash.json').read_text()),scenes=SCENES,gpus=[4,5,6],quality_tasks=['pose','recon_unposed'],quality_retuning=False,controls_executed=0,flash_backend='PyTorch SDPA forced FLASH_ATTENTION only,no fallback',bias='Three FP16/BF16 residual channels encode FP32 bias/scale',padding='Duplicate Special0 and divide its unnormalized mass equally among copies',evidence='Original raw probes and all-query Grid statistics,computed by QK-only kernel with original FP32 bias',group_chunk=16,quality_acceptance='Report full quality and timing;do not automatically promote',dense_speedup=None,dense_reason='No unchanged Dense rerun in this modified-method campaign;prior paired timing batch is historical only'))
 jobs=[(ds,s) for ds,ss in SCENES.items() for s in ss];lock=threading.Lock();done=[]
 def worker(gpu):
  while True:
   with lock:
    if not jobs:return
    ds,scene=jobs.pop(0)
   cmd=[sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(OUT/f'monitors/{ds}_{scene}.json'),'--log',str(OUT/f'logs/{ds}_{scene}.log'),'--',sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method','zdrop_history_states','--frames','128','--warmup','1','--repeats','1','--config',str(ROOT/'configs/evaluation/reuse_v8_torch_flash.json'),'--output-dir',str(OUT/f'quality/{ds}/{scene}')]
   code=subprocess.call(cmd,cwd=ROOT,env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTORCH_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1'))
   with lock:done.append(dict(dataset=ds,scene=scene,gpu=gpu,returncode=code));save(OUT/'progress.json',done)
   print(ds,scene,code,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(worker,[4,5,6]))
 unchanged=all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in hashes.items());save(OUT/'completion.json',dict(complete=len(done)==18 and all(x['returncode']==0 for x in done),source_unchanged=unchanged,runs=done))
if __name__=='__main__':main()
