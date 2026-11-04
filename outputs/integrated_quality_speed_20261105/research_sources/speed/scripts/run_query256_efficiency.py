"""Paired changed-method evaluation; never launches Dense or SAF3R controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/query256_efficiency_20261105'
PYTHON=sys.executable

def worker(scene,gpu,variant,profile):
    if variant=='optimized':
        dependency='baseline_profile' if profile else 'baseline'
        while not (OUT/scene/dependency/'result.json').exists():
            monitor=OUT/f'{scene}_{dependency}_monitor.json'
            if monitor.exists() and json.loads(monitor.read_text()).get('returncode'):raise RuntimeError('Baseline failed '+scene)
            time.sleep(1)
    source=OUT/'frozen_baseline' if variant=='baseline' else ROOT
    label=variant+('_profile' if profile else '')
    out=OUT/scene/label
    if (out/'result.json').exists():return {'scene':scene,'variant':label,'existing':True}
    if out.exists():raise RuntimeError('Preserving incomplete output '+str(out))
    config='configs/evaluation/reuse_shared_value_loss_q256'+('_efficient' if variant=='optimized' else '')+'.json'
    if profile:
        command=[PYTHON,str(ROOT/'scripts/profile_deep_saf3r_query256.py'),'--scene',scene,'--method','shared_value_q256','--output-dir',str(out),'--config',str(source/config),'--source-root',str(source)]
    else:
        command=[PYTHON,'scripts/evaluate_zdrop_history_states.py','--dataset','7scenes','--scene',scene,'--frames','128','--method','zdrop_history_states','--config',config,'--output-dir',str(out),'--warmup','1','--repeats','3']
    command=[PYTHON,str(ROOT/'scripts/launch_local_evaluation.py'),*command[1:]]
    cmd=[PYTHON,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(OUT/f'{scene}_{label}_monitor.json'),'--log',str(OUT/f'{scene}_{label}.log'),'--',*command]
    print('START',scene,label,gpu,flush=True)
    proc=subprocess.run(cmd,cwd=source)
    print('END',scene,label,proc.returncode,flush=True)
    return dict(scene=scene,variant=label,gpu=gpu,returncode=proc.returncode)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['baseline','optimized'],required=True);p.add_argument('--profile',action='store_true');a=p.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(worker,s,g,a.variant,a.profile) for s,g in zip(('heads','chess','stairs'),(0,1,0))]
        status=[j.result() for j in jobs]
    (OUT/(a.variant+('_profile' if a.profile else '')+'_status.json')).write_text(json.dumps(status,indent=2)+'\n')
    if any(s.get('returncode',0) for s in status):sys.exit(1)
