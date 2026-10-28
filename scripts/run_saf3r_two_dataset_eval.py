"""Matched Dense/SAF3R Top-4, two-GPU, two-dataset quality evaluation."""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/saf3r_top4_two_datasets_20261027'
SCENES={'7scenes':['chess','fire','heads','office','pumpkin','redkitchen','stairs'],
        'eth3d':['courtyard','electro','kicker','pipes','relief','delivery_area','facade','office','playground','relief_2','terrains']}
GPUS=['GPU-f0913c13-448b-e88d-92fe-8d47264d279f','GPU-8e332fc5-55f4-0d85-1a2d-40bd87342c1f']
LOCK=threading.Lock()
FIELDS=['auc03','auc30','acc','comp','cd','f1']

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2)+'\n')

def report():
    rows=[]
    for ds,scenes in SCENES.items():
        for scene in scenes:
            for method in ['dense','saf3r']:
                p=OUT/ds/scene/method/'result.json'
                if p.exists():
                    d=json.loads(p.read_text())
                    rows.append(dict(dataset=ds,scene=scene,method=method,**d['quality']['values'],
                        frames=d['frames'],seconds=d['timing']['median_seconds'],
                        speedup=d['quality']['speedup_vs_vggt'],
                        peak_gib=max(x['peak_allocated_gib'] for x in d['timing']['samples']),
                        unavailable=d['quality']['unavailable']))
    macro=[]
    for ds,scenes in SCENES.items():
        for method in ['dense','saf3r']:
            group=[x for x in rows if x['dataset']==ds and x['method']==method]
            if len(group)==len(scenes) and all(all(x[k] is not None for k in FIELDS) for x in group):
                macro.append(dict(dataset=ds,method=method,**{k:sum(x[k] for x in group)/len(group) for k in FIELDS}))
    save(OUT/'summary.json',dict(per_scene=rows,complete_dataset_macro=macro))
    lines=['# SAF3R Top-4 / matched Dense', '',
        '128-frame cap, uniform sampling; actual frame lists and counts saved per run. '
        'Lazy first-correspondence-head per-frame Top-4, ETH3D profile; not a paper DINO-feature Top-K reproduction. '
        'Same checkpoint and inputs within each pair. One warmup, three full forwards; all sparse costs included.',
        'AUC/F1 in percent; ACC/Comp/CD in metres. CD=(ACC+Comp)/2. '
        '7Scenes threshold/downsample=0.05/0.0078125 m; ETH3D=0.25/0.0390625 m.', '',
        '| Dataset | Scene | Method | Frames | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Seconds | Speedup | Peak GiB |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in rows:
        vals=['N/A' if x[k] is None else f"{x[k]*(100 if k in ['auc03','auc30','f1'] else 1):.5f}" for k in FIELDS]
        speed='N/A' if x['speedup'] is None else f"{x['speedup']:.3f}x"
        lines.append('| '+' | '.join([x['dataset'],x['scene'],x['method'],str(x['frames']),*vals,f"{x['seconds']:.3f}",speed,f"{x['peak_gib']:.3f}"])+' |')
    lines+=['','Dataset macro averages are emitted only when all configured scenes have all six metrics.', '','```json',json.dumps(macro,indent=2),'```']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')

def worker(gpu,jobs,state):
    while True:
        try:ds,scene=jobs.get_nowait()
        except queue.Empty:return
        key=f'{ds}/{scene}'
        for method in ['dense','saf3r']:
            out=OUT/ds/scene/method
            with LOCK:
                state['jobs'][key]=dict(state='running',method=method,gpu_uuid=gpu)
                save(OUT/'progress.json',state)
            cmd=[sys.executable,'scripts/evaluate_all_edge_history.py','--dataset',ds,'--scene',scene,
                 '--method',method,'--frames','128','--warmup','1','--repeats','3',
                 '--checkpoint','/data_NVME1/mmc_lyxiang/3D/SAF3R/checkpoints/vggt/model_tracker_fixed_e20.pt',
                 '--output-dir',str(out)]
            if method=='saf3r':cmd+=['--baseline',str(out.parent/'dense/result.json'),'--expected-input',str(out.parent/'dense/input.json')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            log=OUT/'logs'/f'{ds}_{scene}_{method}.log'
            with log.open('w') as stream:
                rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            if rc:
                with LOCK:
                    state['jobs'][key]=dict(state='failed',method=method,returncode=rc,log=str(log))
                    save(OUT/'progress.json',state)
                break
        else:
            with LOCK:
                state['jobs'][key]=dict(state='complete',gpu_uuid=gpu)
                save(OUT/'progress.json',state)
        with LOCK:report()
        jobs.task_done()

def main():
    if OUT.exists():raise ValueError('Fresh output root required')
    (OUT/'logs').mkdir(parents=True)
    state=dict(started_unix=time.time(),topk=4,profile='configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json',jobs={})
    jobs=queue.Queue()
    # Interleave datasets so both start immediately on the two cards.
    for i in range(max(map(len,SCENES.values()))):
        for ds,scenes in SCENES.items():
            if i<len(scenes):
                jobs.put((ds,scenes[i]));state['jobs'][f'{ds}/{scenes[i]}']=dict(state='queued')
    save(OUT/'progress.json',state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu,jobs,state) for gpu in GPUS]
        for f in futures:f.result()
    state['finished_unix']=time.time()
    state['complete']=all(x['state']=='complete' for x in state['jobs'].values())
    save(OUT/'progress.json',state)
    report()

if __name__=='__main__':main()
