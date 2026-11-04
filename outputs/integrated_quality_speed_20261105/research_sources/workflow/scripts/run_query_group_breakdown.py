"""Three matched128-frame sequences, serial group32/128 pair per exclusive GPU."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/query_group_breakdown_20261104'
out.mkdir(exist_ok=True,parents=True)
manifest=dict(scenes=['heads','chess','stairs'],groups=[32,128],frames=128,gpus=[3,4,5],
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    warmup=1,repeats=3,event_profile_repeats=3,kernel_trace_scenes=['heads'],
    quality='Explicit performance diagnostic; all six quality fields N/A. Not a full quality evaluation.',
    dense_speedup='N/A: unchanged Dense not rerun per AGENTS.md')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def worker(scene,gpu):
    for size in ((128,32) if scene=='chess' else (32,128)):
        output=out/scene/f'q{size}'
        if (output/'result.json').exists():continue
        command=[sys.executable,'scripts/run_gpu_lease.py','--strict-contexts','--gpu',str(gpu),
            '--monitor',str(out/f'{scene}_q{size}_monitor.json'),'--log',str(out/f'{scene}_q{size}.log'),'--',
            sys.executable,'scripts/profile_query_group_breakdown.py','--scene',scene,'--group-size',str(size),'--output-dir',str(output)]
        if scene=='heads':command+=['--trace']
        print('START',scene,size,gpu,flush=True)
        result=subprocess.run(command,cwd=ROOT)
        print('END',scene,size,result.returncode,flush=True)
        if result.returncode:raise RuntimeError(f'Failed {scene} q{size}: see log')
with ThreadPoolExecutor(max_workers=3) as pool:
    jobs=[pool.submit(worker,scene,gpu) for scene,gpu in zip(manifest['scenes'],manifest['gpus'])]
    status=[]
    for job in jobs:
        try:job.result();status.append(dict(complete=True))
        except Exception as exc:status.append(dict(complete=False,error=str(exc)))
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print('ALL_JOBS_FINISHED',status,flush=True)
