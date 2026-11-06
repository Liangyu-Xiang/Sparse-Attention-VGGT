"""Do not interrupt the paired scene queue for supplementary real-head tests."""
import subprocess,sys,time,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=Path('/tmp/sparse-vggt-history-sparse90-results-20261106')
while not (out/'quality/completion.json').exists():time.sleep(2)
cmd=[sys.executable,str(ROOT/'scripts/run_profiler_gpu_exclusive.py'),'--gpu','5','--monitor',str(out/'final_kernel_tests_monitor.json'),'--log',str(out/'final_kernel_tests.log'),'--',sys.executable,'-m','pytest','-q','tests/test_teacher_history_raw_fast.py','tests/test_total_edge_retention.py']
assert subprocess.call(cmd,cwd=ROOT,env=os.environ)==0
