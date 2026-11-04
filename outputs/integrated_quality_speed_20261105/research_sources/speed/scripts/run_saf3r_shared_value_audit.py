"""Six same-GPU performance diagnostics, no Dense rerun or quality retuning."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/saf3r_shared_value_audit_20261105'
OUT.mkdir(parents=True,exist_ok=True)
SCENES=('heads','chess','stairs')
(OUT/'manifest.json').write_text(json.dumps(dict(scenes=SCENES,frames=128,methods=['saf3r','shared_value_q128'],
    gpus=[3,4,5],warmup=1,repeats=3,event_repeats=3,quality='Explicit timing diagnostic; six quality fields N/A',
    prior_complete_quality_report='outputs/dense_saf3r_query128_20261104/report.md',
    unchanged=['checkpoint','inputs','preprocessing','Key retention','head profile','Top-K','Value clustering'],
    purpose='Test whether only support construction is slower, or attention execution/history/other model operations also differ'),indent=2)+'\n')
def worker(scene,gpu):
    order=('shared_value_q128','saf3r') if scene=='chess' else ('saf3r','shared_value_q128')
    for method in order:
        out=OUT/scene/method
        if (out/'result.json').exists():continue
        if out.exists():raise RuntimeError('Incomplete output preserved:'+str(out))
        cmd=[sys.executable,'scripts/run_gpu_lease.py','--strict-contexts','--gpu',str(gpu),'--monitor',str(OUT/f'{scene}_{method}_monitor.json'),
            '--log',str(OUT/f'{scene}_{method}.log'),'--',sys.executable,'scripts/profile_saf3r_vs_shared_value.py',
            '--scene',scene,'--method',method,'--output-dir',str(out)]
        print('START',scene,method,gpu,flush=True);proc=subprocess.run(cmd,cwd=ROOT)
        print('END',scene,method,proc.returncode,flush=True)
        if proc.returncode:raise RuntimeError(f'{scene}/{method} failed')
with ThreadPoolExecutor(max_workers=3) as pool:
    jobs=[pool.submit(worker,s,g) for s,g in zip(SCENES,(3,4,5))]
    status=[]
    for job in jobs:
        try:job.result();status.append(dict(complete=True))
        except Exception as exc:status.append(dict(complete=False,error=str(exc)))
(OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print('FINISHED',status,flush=True)
