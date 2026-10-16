"""Run one-forward Chess sensitivity checks, using only idle GPUs.

These are explicitly cold single-forward accuracy checks, NOT warm timings.
Parameter selection used no pose metrics; held-out validation remains separate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TRIALS = [
    ('scale16', ['--logit-scale', '16']),
    ('scale64', ['--logit-scale', '64']),
    ('tau09', ['--dominance', '.9']),
    ('tau095', ['--dominance', '.95']),
    ('continuous', ['--dominance', '0']),
    ('eta01', ['--sampling', '.01']),
    ('eta005', ['--sampling', '.005']),
    ('eta001', ['--sampling', '.001']),
    ('fine020', ['--fine-ratio', '.20']),
    ('coarse064', ['--coarse-size', '64']),
    ('coarse256', ['--coarse-size', '256']),
]


def main(args):
    pending = list(TRIALS)
    active, failed = {}, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
            name, options = pending.pop(0)
            directory = args.output_dir/name
            if (directory/'chess/result.json').exists(): continue
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, 'scripts/evaluate_layer_shared_av.py', '--scene', 'chess',
                '--frames', '128', '--warmup', '0', '--repeats', '1',
                '--logit-scale', '32', '--dominance', '.8', '--sampling', '.0025',
                '--output-dir', str(directory), *options]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='.')
            log = (directory/'run.log').open('w')
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = proc, log, name
            print('launched', gpu, name, proc.pid, flush=True)
        (args.output_dir/'status.json').write_text(json.dumps(dict(
            pending=[n for n, _ in pending], active={g: n for g, (_, _, n) in active.items()}, failed=failed), indent=2))
        if pending or active: time.sleep(5)
    if failed: raise RuntimeError(f'Failed sensitivity checks: {failed}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/layer_shared_av/ablation_chess_n128'))
    main(p.parse_args())
