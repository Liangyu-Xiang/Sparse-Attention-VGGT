"""Idle-GPU queue for the final shared policy, without changing mask semantics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    pending = list(args.scenes)
    active, failed = {}, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu, (proc, log, scene) in list(active.items()):
            if proc.poll() is not None:
                if proc.returncode: failed.append(scene)
                log.close(); del active[gpu]
                print('finished', scene, proc.returncode, flush=True)
        status = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        idle = [int(line.split(',')[0]) for line in status.splitlines()
                if int(line.split(',')[1]) < 200 and int(line.split(',')[2]) == 0]
        for gpu in idle:
            if gpu in active or not pending: continue
            scene = pending.pop(0)
            if (args.output_dir/scene/'result.json').exists():
                raise ValueError('Use a fresh output directory rather than mixing experiments')
            command = [sys.executable, 'scripts/evaluate_layer_shared_av.py', '--scene', scene,
                '--frames', '128', '--warmup', '1', '--repeats', '3',
                '--logit-scale', str(args.logit_scale), '--dominance', str(args.dominance), '--sampling', '.0025',
                '--query-chunk', str(args.query_chunk), '--output-dir', str(args.output_dir)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='.')
            log = (args.output_dir/(scene+'.log')).open('w')
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = proc, log, scene
            print('launched', gpu, scene, proc.pid, flush=True)
        (args.output_dir/'queue_status.json').write_text(json.dumps(dict(
            pending=pending, active={g:s for g,(_,_,s) in active.items()}, failed=failed), indent=2))
        if pending or active: time.sleep(5)
    if failed: raise RuntimeError(f'Failed validation: {failed}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/layer_shared_av/selected_joint_v3_n128'))
    p.add_argument('--logit-scale', type=float, default=32.)
    p.add_argument('--dominance', type=float, default=.95)
    p.add_argument('--scenes', nargs='+', default=['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs'])
    p.add_argument('--query-chunk', type=int, default=1024)
    main(p.parse_args())
