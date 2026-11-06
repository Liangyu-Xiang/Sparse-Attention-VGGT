"""Do not interleave with the already-authorized14% paired measurement batch."""
import json,time,subprocess,sys,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=Path('/tmp/sparse-vggt-history-sparse90-results-20261106');prior=Path('/tmp/sparse-vggt-history-speed-results-20261106/quality/completion.json')
while not prior.exists():time.sleep(2)
assert json.loads(prior.read_text())['complete']
env=dict(os.environ,LD_LIBRARY_PATH='/mnt/rfs/users/xiangliangyu/conda-envs/vggt-baseline/lib',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
def run(pair):
 shard,gpu=pair;cmd=[sys.executable,str(ROOT/'scripts/run_profiler_gpu_exclusive.py'),'--gpu',str(gpu),'--monitor',str(out/f'screen{shard}_monitor.json'),'--log',str(out/f'screen{shard}.log'),'--',sys.executable,str(ROOT/'scripts/screen_history_sparse90.py'),'--plan',str(out/'search_plan.json'),'--output',str(out/'screen'),'--shard',str(shard)];return subprocess.call(cmd,cwd=ROOT,env=env)
with ThreadPoolExecutor(max_workers=3) as pool:codes=list(pool.map(run,enumerate([4,5,6])))
assert all(v==0 for v in codes),codes
print('ALL_SCREEN_COMPLETE',flush=True)
