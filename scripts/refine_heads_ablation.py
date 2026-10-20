"""Quality-only old-scheme substage interventions after heads stage diagnosis."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_cells,LABELS

PARTS={'middle_10_12':tuple(range(10,13)),'middle_13_16':tuple(range(13,17)),
       'deep_17_20':tuple(range(17,21)),'deep_21_23':tuple(range(21,24))}


def intervention(name):
    return PARTS[name]


def child(args):
    from saf3r.patch.av_stage_ablation import STAGES
    from scripts.evaluate_attention_first_residual_value import run
    STAGES[args.part]=intervention(args.part)
    # All ordinary mathematical selection settings remain unchanged. Only
    # which attention outputs are returned as sparse vs Dense is intervened on.
    run(argparse.Namespace(output_dir=args.output_dir/args.part,mode='old',active_stage=args.part,
        dataset='7scenes',scene='heads',frames=128,config=None,baseline_result=None,
        capture=False,warmup=0,repeats=1,tasks=['pose','depth','recon_unposed']))


def queue(args):
    from scripts.prioritize_heads_ablation import gpu_state
    root=args.output_dir;root.mkdir(parents=True,exist_ok=True)
    pending=list(PARTS);jobs={};lock=Lock()
    save(root/'manifest.json',dict(parts=PARTS,mode='old',frames=128,warmup=0,repeats=1,
        reason='Initial stage ablations localized degradation to middle/deep, not shallow',timing_comparison=False))
    def report():
        rows=[]
        for part in PARTS:
            file=root/part/'old/heads/result.json'
            if file.exists():
                d=json.loads(file.read_text());q=d['quality'];q.update(speedup_vs_vggt=None,
                    speedup_unavailable='Concurrent accuracy-only ablation; not a speed benchmark')
                d.update(quality=q,benchmark_timing=False);save(file,d);write_quality(file.parent,q)
                rows.append(dict(part=part,layers=PARTS[part],quality=q,input=d['input']))
        save(root/'comparison.json',dict(rows=rows,complete=len(rows)==4))
        lines=['| Sparse layers | '+' | '.join(LABELS)+' |','|---|'+'---:|'*6]
        for row in rows:lines.append('| '+str(row['layers'])+' | '+' | '.join(quality_cells(row['quality']))+' |')
        lines+=['','All remaining layers return original Dense output. Same128 frames, old manual/history80 policy.',
                'No runtime comparisons; all six task metrics evaluated.']
        (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    def status():
        save(root/'status.json',dict(stage='running',jobs=jobs));report()
    def worker(gpu):
        while True:
            with lock:
                if not pending:return
            used,total,pids=gpu_state(gpu)
            if len(pids)>=2 or total-used<20*1024:time.sleep(10);continue
            with lock:
                if not pending:return
                part=pending.pop(0)
                if (root/part/'old/heads/result.json').exists():jobs[part]=dict(stage='complete');status();continue
                jobs[part]=dict(stage='running',gpu=gpu,started_unix=time.time());status()
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(ROOT),
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            log=root/'logs'/f'{part}.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as stream:
                result=subprocess.run([sys.executable,'-u',__file__,'--output-dir',str(root),'--part',part],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            with lock:
                jobs[part].update(stage='complete' if result.returncode==0 else 'failed',returncode=result.returncode);status()
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        for future in [pool.submit(worker,gpu) for gpu in args.gpus]:future.result()
    save(root/'status.json',dict(stage='complete' if all(r['stage']=='complete' for r in jobs.values()) else 'failed',jobs=jobs));report()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--part',choices=list(PARTS));p.add_argument('--gpus',type=int,nargs='+',default=[4,5,6])
    args=p.parse_args()
    if args.part:child(args)
    else:queue(args)
