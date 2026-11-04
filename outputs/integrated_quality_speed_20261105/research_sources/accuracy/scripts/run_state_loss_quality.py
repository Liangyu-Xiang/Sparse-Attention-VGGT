"""Evaluate two frozen History semantics; never select losses using test quality."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',type=int,nargs='+',required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    variants={'A_history_self_residual_rep':'centered_residual_energy','B_history_group':'group_centered'}
    scenes=[('eth3d',s) for s in ['kicker','playground','courtyard']]+[('7scenes',s) for s in ['heads','chess','stairs','fire','office','pumpkin','redkitchen']]
    jobs=[(name,loss,dataset,scene) for dataset,scene in scenes for name,loss in variants.items()]
    files=[ROOT/'saf3r/patch/state_loss_reference.py',ROOT/'saf3r/patch/state_loss_controller.py',ROOT/'saf3r/triton/state_loss_controller.py',
        ROOT/'saf3r/patch/reuse_residual_exploration.py',ROOT/'saf3r/patch/reuse_window_graph.py']+[ROOT/f'configs/evaluation/state_loss_{loss}.json' for loss in variants.values()]
    manifest=dict(selection_basis='Frozen using Dense attention-output diagnostics on ETH3D kicker/playground; no test pose/reconstruction metric used to choose losses',
        variants=variants,scope=[dict(method=n,dataset=d,scene=s) for n,_,d,s in jobs],
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        dense_speedup=None,dense_speedup_reason='Unchanged Dense not rerun for timing; matched historical quality controls reused',
        timing_protocol=dict(warmup=1,repeats=3,scope='Full model forward including all reset/clustering/routing/history and output heads'))
    path=a.output/'frozen_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Frozen experiment manifest changed; fresh output required')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    jobs=[j for j in jobs if not (a.output/f'{j[0]}/{j[2]}/{j[3]}/result.json').exists()]
    lock=threading.Lock();completed=[]
    def worker(gpu):
        while True:
            with lock:
                if not jobs:return
                name,loss,dataset,scene=jobs.pop(0)
            tag=f'{name}_{dataset}_{scene}';folder=a.output/name/dataset/scene
            if folder.exists():raise ValueError(f'Incomplete existing run requires inspection: {folder}')
            cmd=[sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu',str(gpu),
                '--monitor',str(a.output/f'{tag}_monitor.json'),'--log',str(a.output/f'{tag}.log'),'--',
                sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',dataset,'--scene',scene,
                '--frames','128','--method','zdrop_history_states','--config',str(ROOT/f'configs/evaluation/state_loss_{loss}.json'),
                '--output-dir',str(folder),'--warmup','1','--repeats','3']
            print('START',tag,gpu,flush=True);code=subprocess.call(cmd,cwd=ROOT)
            with lock:
                completed.append(dict(method=name,dataset=dataset,scene=scene,gpu=gpu,returncode=code))
                (a.output/'progress.json').write_text(json.dumps(completed,indent=2)+'\n')
            print('DONE',tag,code,flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(a.gpus)) as pool:list(pool.map(worker,a.gpus))


if __name__=='__main__':main()
