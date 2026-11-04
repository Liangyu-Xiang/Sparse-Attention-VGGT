"""Live per-node proof for the final, bit-exact split execution."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/query256_deterministic_split_20261105'
def run(scene,gpu):
 out=OUT/('live_'+scene)
 cmd=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(OUT/f'live_{scene}_monitor.json'),'--log',str(OUT/f'live_{scene}.log'),'--',sys.executable,str(ROOT/'scripts/launch_local_evaluation.py'),'scripts/validate_query256_live_equivalence.py','--scene',scene,'--config','configs/evaluation/reuse_shared_value_loss_q256_deterministic_split.json','--output-dir',str(out)]
 p=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8'})
 if p.returncode:raise RuntimeError('Live proof failed '+scene)
with ThreadPoolExecutor(max_workers=3) as pool:
 for result in [pool.submit(run,s,g) for s,g in zip(('heads','chess','stairs'),(0,1,2))]:result.result()
