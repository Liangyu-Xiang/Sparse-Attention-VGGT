"""Frozen changed method only; one quality evaluation per scene, no GT tuning."""
import argparse,concurrent.futures as cf,hashlib,json,subprocess,sys,threading
from pathlib import Path

def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--config',required=True);a=p.parse_args();out=a.output_dir;source=out/'source'
    hashes=json.loads((out/'source_sha256.json').read_text());assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    scenes={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}
    cfg=json.loads((source/a.config).read_text());save(out/'manifest.json',dict(source_root=str(source),source_sha256=hashes,config=cfg,datasets=scenes,quality_tasks=['pose','recon_unposed'],controls_executed=0,quality_retuning=False,timing_protocol='1 warmup + 1 quality forward; separate paired speed benchmark uses 3 repeats',dense_speedup=None,dense_reason='User explicitly requested no Dense rerun'))
    queue=[(ds,s) for ds,ss in scenes.items() for s in ss];lock=threading.Lock();done=[]
    def worker(gpu):
        while True:
            with lock:
                if not queue:return
                ds,scene=queue.pop(0)
            folder=out/ds/scene/'efficiency'
            cmd=[sys.executable,str(source/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(folder.parent/'lease_monitor.json'),'--log',str(out/'logs'/f'{ds}_{scene}.log'),'--',sys.executable,str(source/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method','zdrop_history_states','--frames','128','--warmup','1','--repeats','1','--config',str(source/a.config),'--output-dir',str(folder)]
            code=subprocess.call(cmd,cwd=source)
            with lock:done.append(dict(dataset=ds,scene=scene,gpu=gpu,returncode=code));save(out/'progress.json',done)
            print(ds,scene,code,flush=True)
    with cf.ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(worker,[0,1,2,3,4,6,7]))
    assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    save(out/'completion.json',dict(runs=done,all_processes_succeeded=len(done)==18 and all(d['returncode']==0 for d in done)))

if __name__=='__main__':main()
