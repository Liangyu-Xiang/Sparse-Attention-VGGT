"""Frozen local-normalization method only, complete 7Scenes and ETH3D quality."""
import argparse,concurrent.futures as cf,datetime,hashlib,json,os,subprocess,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--retry-from',type=Path);args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    scenes={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}
    names=['saf3r/patch/reuse_residual_exploration.py','saf3r/patch/reuse_grid_local_history.py','saf3r/triton/reuse_grid_local_history.py','saf3r/patch/reuse_ready_history.py','saf3r/triton/reuse_single_qk_grid.py','configs/evaluation/reuse_grid_local_history.json']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    save(out/'manifest.json',dict(source_root=str(ROOT),source_sha256=hashes,datasets=scenes,config=json.loads((ROOT/names[-1]).read_text()),frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),quality_tasks=['pose','recon_unposed'],controls_executed=0,quality_retuning=False,timing=dict(warmup=1,repeats=1,reason='Quality execution only; full-forward paired benchmark separately uses 1 warmup + 3 repeats')))
    queue=[(ds,s) for ds,ss in scenes.items() for s in ss]
    if args.retry_from:
        failed=json.loads((args.retry_from/'progress.json').read_text())
        queue=[(r['dataset'],r['scene']) for r in failed if r['returncode']!=0]
    count=len(queue);save(out/'scheduled.json',dict(runs=queue,retry_from=str(args.retry_from) if args.retry_from else None))
    lock=threading.Lock();done=[]
    def worker(gpu):
        while True:
            with lock:
                if not queue:return
                ds,scene=queue.pop(0)
            folder=out/ds/scene/'grid_local'
            cmd=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(folder.parent/'lease_monitor.json'),'--log',str(out/'logs'/f'{ds}_{scene}.log'),'--',sys.executable,str(ROOT/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method','zdrop_history_states','--frames','128','--warmup','1','--repeats','1','--config',str(ROOT/names[-1]),'--output-dir',str(folder)]
            code=subprocess.call(cmd,cwd=ROOT)
            with lock:
                done.append(dict(dataset=ds,scene=scene,gpu=gpu,returncode=code));save(out/'progress.json',done)
            print(ds,scene,code,flush=True)
    with cf.ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(worker,range(7)))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    save(out/'completion.json',dict(runs=done,all_processes_succeeded=len(done)==count and all(d['returncode']==0 for d in done)))


if __name__=='__main__':main()
