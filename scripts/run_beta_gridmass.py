"""Frozen changed-method-only full evaluation, GPUs4-7, no auxiliary contexts."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/beta_gridmass_20261104'
SCENES={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],
 'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    assert json.loads((ROOT/'outputs/beta_gridmass_validation_v2_20261104/result.json').read_text())['complete']
    OUT.mkdir(exist_ok=False,parents=True)
    hashes={}
    for directory in ['saf3r','scripts','configs','evaluation','tests']:
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix in {'.py','.json','.yaml','.yml','.toml'}:
                hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(OUT/'source_sha256.json',hashes)
    with tarfile.open(OUT/'source_snapshot.tar.gz','w:gz') as archive:
        for name in sorted(hashes):archive.add(ROOT/name,arcname=name,recursive=False)
    save(OUT/'manifest.json',dict(method='beta_gridmass',config=json.loads((ROOT/'configs/evaluation/reuse_beta_gridmass.json').read_text()),
        scenes=SCENES,gpus=[4,5,6,7],source_base='99bbb99b with exact frozen v8 window implementation',controls_executed=0,
        quality_tasks=['pose','recon_unposed'],quality_timing='One warmup and one measured output per scene',
        benchmark='Heads128,one warmup + three complete forwards + separate event profile',
        dense_speedup=None,dense_reason='Unchanged Dense not rerun; no contemporaneous paired timing',
        quality_retuning=False,single_cuda_process=True,beta='original ratio<1 ? ratio : ratio^-2 confidence, capped by current Grid mass',
        missing_mass='uniform on unselected members; fully-selected Grid rescale/fallback; Special handled by current coarse distribution',
        cache='head-mean true sparse P including compensation bias and Special normalization,FP16-u14',
        initial_validation_failure='v1 compared FP16 cache directly with too-small FP32 absolute tolerance;v2 checks quantized reference; model unchanged'))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    def launch(gpu,label,command):
        code=subprocess.call([sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu',str(gpu),
            '--monitor',str(OUT/'monitors'/f'{label}.json'),'--log',str(OUT/'logs'/f'{label}.log'),'--',*command],cwd=ROOT,env=env)
        print(label,code,flush=True);return code
    jobs=[(dataset,scene) for dataset,scenes in SCENES.items() for scene in scenes]
    lock=threading.Lock();done=[]
    def worker(gpu):
        if gpu==7:
            code=launch(gpu,'timing',[sys.executable,str(ROOT/'scripts/benchmark_beta_gridmass.py'),
                '--config',str(ROOT/'configs/evaluation/reuse_beta_gridmass.json'),
                '--input',str(ROOT/'outputs/inference_efficiency_20261103/v8/quality/7scenes/heads/efficiency/input.json'),
                '--output-dir',str(OUT/'timing'),'--profile'])
            save(OUT/'timing_status.json',dict(returncode=code))
        while True:
            with lock:
                if not jobs:return
                dataset,scene=jobs.pop(0)
            code=launch(gpu,f'{dataset}_{scene}',[sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),
                '--dataset',dataset,'--scene',scene,'--method','zdrop_history_states','--frames','128',
                '--warmup','1','--repeats','1','--config',str(ROOT/'configs/evaluation/reuse_beta_gridmass.json'),
                '--output-dir',str(OUT/'quality'/dataset/scene)])
            with lock:
                done.append(dict(dataset=dataset,scene=scene,gpu=gpu,returncode=code));save(OUT/'progress.json',done)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,[4,5,6,7]))
    unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items())
    save(OUT/'completion.json',dict(runs=done,complete=len(done)==18 and all(x['returncode']==0 for x in done),source_unchanged=unchanged))
    assert unchanged


if __name__=='__main__':main()
