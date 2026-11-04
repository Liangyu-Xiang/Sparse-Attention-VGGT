"""Frozen repaired method only; seven 7Scenes scenes and ETH3D courtyard.

ETH3D kicker/playground were evaluated separately. No unchanged controls rerun.
"""
import concurrent.futures
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/shared_value_accuracy_20261105'
CONFIG=ROOT/'configs/evaluation/reuse_shared_value_loss_256_repaired.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs='+',default=[4,5])
    parser.add_argument('--exclude',nargs='*',default=[])
    args=parser.parse_args()
    queue=[('7scenes',s) for s in ['heads','chess','stairs','fire','office','pumpkin','redkitchen']]+[('eth3d','courtyard')]
    queue=[(d,s) for d,s in queue if s not in args.exclude and not (OUT/f'quality/{d}/{s}/result.json').exists()]
    files=[*ROOT.glob('saf3r/patch/*shared*repair*.py'),*ROOT.glob('saf3r/triton/*shared*repair*.py'),
           ROOT/'saf3r/patch/reuse_residual_exploration.py',ROOT/'saf3r/patch/reuse_window_graph.py',CONFIG]
    manifest=dict(config=json.loads(CONFIG.read_text()),scope=queue,selection_basis='Frozen using Dense attention fidelity on ETH3D kicker8; no 7Scenes test-truth tuning',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        dense_speedup=None,dense_reason='Unchanged Dense quality references reused; no new paired Dense timings',gpus=args.gpus)
    (OUT/'frozen_quality_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lock=threading.Lock();results=[]
    def worker(gpu):
        while True:
            with lock:
                if not queue:return
                dataset,scene=queue.pop(0)
            folder=OUT/f'quality/{dataset}/{scene}'
            cmd=[sys.executable,str(ROOT/'scripts/run_gpu_exclusive.py'),'--gpu',str(gpu),
                '--monitor',str(OUT/f'quality_{dataset}_{scene}_monitor.json'),'--log',str(OUT/f'quality_{dataset}_{scene}.log'),'--',
                sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',dataset,'--scene',scene,
                '--frames','128','--method','zdrop_history_states','--config',str(CONFIG),'--output-dir',str(folder),'--warmup','1','--repeats','3']
            print('START',dataset,scene,gpu,flush=True)
            code=subprocess.call(cmd,cwd=ROOT)
            with lock:
                results.append(dict(dataset=dataset,scene=scene,gpu=gpu,returncode=code))
                (OUT/'quality_progress.json').write_text(json.dumps(results,indent=2)+'\n')
            print('DONE',dataset,scene,code,flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(args.gpus)) as pool:list(pool.map(worker,args.gpus))


if __name__=='__main__':main()
