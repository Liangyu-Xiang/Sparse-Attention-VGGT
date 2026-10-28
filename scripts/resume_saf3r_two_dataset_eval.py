"""Resume failed pairs without overwriting completed quality results."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from scripts.run_saf3r_two_dataset_eval import OUT,ROOT,report,save

LOCK=threading.Lock()
STATE={}
JOBS={
 'GPU-f0913c13-448b-e88d-92fe-8d47264d279f':[('7scenes','chess'),('7scenes','heads')],
 'GPU-a67db5fc-c14e-4884-1909-a7c7050ef267':[('7scenes','fire')],
 'GPU-917c95c3-9023-f077-f577-dae0f7f9d717':[('eth3d','electro')],
}

def active(gpu):
    s=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {int(parts[1].strip()) for line in s.splitlines() if len(parts:=line.split(','))==2 and parts[0].strip()==gpu}

def update(key,**data):
    with LOCK:
        STATE[key]=data
        save(OUT/'retry_progress.json',STATE)

def worker(gpu,pairs):
    for ds,scene in pairs:
        for method in ['dense','saf3r']:
            out=OUT/ds/scene/method
            if (out/'result.json').exists():continue
            key=f'{ds}/{scene}/{method}'
            while active(gpu):
                update(key,state='waiting_for_idle_gpu',gpu_uuid=gpu)
                time.sleep(15)
            if out.exists():
                archived=out.with_name(method+'.failed_attempt_'+str(time.time_ns()))
                out.rename(archived)
            cmd=[sys.executable,'scripts/evaluate_all_edge_history.py','--dataset',ds,'--scene',scene,
                 '--method',method,'--frames','128','--warmup','1','--repeats','3',
                 '--checkpoint','/data_NVME1/mmc_lyxiang/3D/SAF3R/checkpoints/vggt/model_tracker_fixed_e20.pt',
                 '--output-dir',str(out)]
            if method=='saf3r':cmd+=['--baseline',str(out.parent/'dense/result.json'),'--expected-input',str(out.parent/'dense/input.json')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONPATH=str(ROOT),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            log=OUT/'logs'/f'{ds}_{scene}_{method}_retry.log'
            with log.open('w') as stream:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                update(key,state='running',pid=p.pid,gpu_uuid=gpu,log=str(log))
                foreign=set()
                while p.poll() is None:
                    if not (out/'forward.json').exists():foreign.update(active(gpu)-{p.pid})
                    time.sleep(3)
            save(out/'gpu_contention.json',dict(gpu_uuid=gpu,foreign_pids=sorted(foreign),monitored=True))
            if p.returncode:
                update(key,state='failed',returncode=p.returncode,log=str(log))
                break
            if method=='saf3r':
                result=json.loads((out/'result.json').read_text())
                baseline_monitor=out.parent/'dense/gpu_contention.json'
                verified=baseline_monitor.exists() and not json.loads(baseline_monitor.read_text())['foreign_pids'] and not foreign
                if not verified:
                    result['quality']['speedup_vs_vggt']=None
                    result['quality']['speedup_unavailable']='Dense baseline lacks contention monitoring, or paired timing encountered GPU contention; see gpu_contention.json'
                    save(out/'result.json',result)
                    from scripts.evaluate_attention_first_residual_value import write_quality
                    write_quality(out,result['quality'])
            update(key,state='complete',gpu_uuid=gpu)
            with LOCK:report()
    with LOCK:report()

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(JOBS)) as pool:
        tasks=[pool.submit(worker,gpu,pairs) for gpu,pairs in JOBS.items()]
        for task in tasks:task.result()
