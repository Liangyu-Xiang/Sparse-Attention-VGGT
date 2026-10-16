"""Bounded 3x3 Chess-only joint scale/threshold check on the CORRECT policy."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('outputs/layer_shared_av/joint_v3_chess_n128')


def main():
    # (32,.9) already has three measured forwards in the seven-scene suite.
    pending = [(m, t) for m in (16, 32, 64) for t in (.8, .9, .95) if (m, t) != (32, .9)]
    active, failed = {}, []
    ROOT.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu, (proc, log, name) in list(active.items()):
            if proc.poll() is not None:
                if proc.returncode: failed.append(name)
                log.close(); del active[gpu]
                print('finished', name, proc.returncode, flush=True)
        status = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        idle = [int(line.split(',')[0]) for line in status.splitlines()
            if int(line.split(',')[1]) < 200 and int(line.split(',')[2]) == 0]
        for gpu in idle:
            if gpu in active or not pending: continue
            multiplier, tau = pending.pop(0)
            name = f'scale{multiplier}_tau{tau}'
            directory = ROOT/name
            if (directory/'chess/result.json').exists(): raise ValueError('Use fresh results')
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, 'scripts/evaluate_layer_shared_av.py', '--scene', 'chess',
                '--frames', '128', '--warmup', '0', '--repeats', '1', '--logit-scale', str(multiplier),
                '--dominance', str(tau), '--sampling', '.0025', '--query-chunk', '1024',
                '--output-dir', str(directory)]
            log = (directory/'run.log').open('w')
            proc = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='.'),
                stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = proc, log, name
            print('launched', gpu, name, proc.pid, flush=True)
        (ROOT/'status.json').write_text(json.dumps(dict(pending=pending,
            active={g:n for g,(_,_,n) in active.items()}, failed=failed), indent=2))
        if pending or active: time.sleep(5)
    if failed: raise RuntimeError(failed)


if __name__ == '__main__': main()
