"""Continue only never-started scenes on idle GPUs; preserve the existing stairs worker."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/deep_saf3r_query256_20261105'
plan={'heads':3,'chess':6}
(OUT/'rescheduled_plan.json').write_text(json.dumps(dict(scene_gpu=plan,reason='GPUs4/5 became occupied before leases acquired; no run started there. GPU3 became idle. Existing stairs remains onGPU6; leases serialize all work.'),indent=2)+'\n')
def worker(scene,gpu):
 for method in ('saf3r','shared_value_q256'):
  out=OUT/scene/method
  if (out/'result.json').exists():continue
  if out.exists():raise RuntimeError('Incomplete existing run:'+str(out))
  call=[sys.executable,'scripts/run_gpu_lease.py','--strict-contexts','--gpu',str(gpu),
    '--monitor',str(OUT/f'{scene}_{method}_monitor.json'),'--log',str(OUT/f'{scene}_{method}.log'),'--',
    sys.executable,'scripts/profile_deep_saf3r_query256.py','--scene',scene,'--method',method,'--output-dir',str(out)]
  print('START',scene,method,gpu,flush=True);p=subprocess.run(call,cwd=ROOT)
  print('END',scene,method,p.returncode,flush=True)
  if p.returncode:raise RuntimeError(f'{scene}/{method} failed')
with ThreadPoolExecutor(max_workers=2) as pool:
 jobs=[pool.submit(worker,s,g) for s,g in plan.items()]
 for j in jobs:j.result()
print('RESCHEDULED_SCENES_COMPLETE',flush=True)
