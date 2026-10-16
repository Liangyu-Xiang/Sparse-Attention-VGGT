"""Fresh Dense/SAF3R128 Chess timing, after a GPU becomes genuinely idle."""
import os
import subprocess
import sys
import time


while True:
    status = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    idle = [int(line.split(',')[0]) for line in status.splitlines()
        if int(line.split(',')[1]) < 200 and int(line.split(',')[2]) == 0]
    if idle:
        gpu = idle[0]
        print('Fresh baseline GPU', gpu, flush=True)
        subprocess.run([sys.executable, 'scripts/benchmark_adaptive_block_value.py',
            '--scene', 'chess', '--frames', '128', '--warmup', '1', '--repeats', '3',
            '--variants', 'dense', 'saf3r', '--output-dir',
            'outputs/layer_shared_av/paired_fresh_topk_chess_n128'],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH='.'), check=True)
        break
    time.sleep(5)
