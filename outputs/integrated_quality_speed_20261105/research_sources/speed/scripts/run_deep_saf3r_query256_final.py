"""Matched deep operation audit: Query256 vs SAF3R, three128-frame sequences."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/deep_saf3r_query256_20261105_final'
SCENES=('heads','chess','stairs')
OUT.mkdir(parents=True,exist_ok=True)
validation=json.loads((ROOT/'outputs/deep_saf3r_query256_20261105/validation/result.json').read_text())
assert validation['graph_validation']['passed'] and validation['clustering_scopes_preserve_outputs']
(OUT/'manifest.json').write_text(json.dumps(dict(scenes=SCENES,frames=128,methods=['saf3r','shared_value_q256'],
    gpus=[2,5,6],query_group_size=256,retention=.2,history_interval=256,warmup=1,repeats=3,
    profiling='One instrumentation warmup after normal timing,then3deep passes; graph-safe timestamps; no Kineto',
    additive_requirement='Every scene table uses one SAME profiled forward. Hierarchy children plus explicit remainder equal parent. Root includes wall-vs-device boundary time. No mixing normal benchmark medians and profiled intervals.',
    quality='Explicit performance diagnostic; six quality fields N/A with reason; no new Dense comparison',
    validation=validation),indent=2)+'\n')
def worker(scene,gpu):
    order=('shared_value_q256','saf3r') if scene=='chess' else ('saf3r','shared_value_q256')
    for method in order:
        out=OUT/scene/method
        if (out/'result.json').exists():continue
        if out.exists():raise RuntimeError('Incomplete output preserved: '+str(out))
        cmd=[sys.executable,'scripts/run_gpu_lease.py','--strict-contexts','--gpu',str(gpu),
            '--monitor',str(OUT/f'{scene}_{method}_monitor.json'),'--log',str(OUT/f'{scene}_{method}.log'),'--',
            sys.executable,'scripts/profile_deep_saf3r_query256.py','--scene',scene,'--method',method,'--output-dir',str(out)]
        print('START',scene,method,gpu,flush=True);proc=subprocess.run(cmd,cwd=ROOT)
        print('END',scene,method,proc.returncode,flush=True)
        if proc.returncode:raise RuntimeError(f'{scene}/{method} failed; see log')
with ThreadPoolExecutor(max_workers=3) as pool:
    jobs=[pool.submit(worker,s,g) for s,g in zip(SCENES,(2,5,6))];status=[]
    for job in jobs:
        try:job.result();status.append(dict(complete=True))
        except Exception as exc:status.append(dict(complete=False,error=str(exc)))
(OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print('COMPLETE',status,flush=True)
