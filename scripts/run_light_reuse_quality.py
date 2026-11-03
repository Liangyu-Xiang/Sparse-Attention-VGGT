"""Frozen, predeclared modeling ladder; only changed methods are evaluated."""
import argparse,concurrent.futures as cf,hashlib,json,subprocess,sys,threading
from pathlib import Path

def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir;source=out/'source';hashes=json.loads((out/'source_sha256.json').read_text())
 assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
 variants={name:dict(method='zdrop_history_states' if name=='calibration' else 'light_reuse',config=f'configs/evaluation/reuse_light_{name}.json') for name in ['calibration','simple','all_exact','full','global','single_probe']}
 scenes={'7scenes':['heads','chess','stairs','fire','office','pumpkin','redkitchen'],'eth3d':['courtyard','delivery_area','electro','facade','kicker','office','pipes','playground','relief','relief_2','terrains']}
 save(out/'manifest.json',dict(variants=variants,configs={k:json.loads((source/v['config']).read_text()) for k,v in variants.items()},scenes=scenes,quality_tasks=['pose','recon_unposed'],quality_retuning=False,controls_executed=0,source_root=str(source),timing_protocol='One warmup and one quality forward per scene; independent speed jobs use 3 repeats',dense_speedup=None,dense_reason='User prohibited unchanged Dense rerun'))
 queue=[(name,ds,scene) for name in variants for ds,ss in scenes.items() for scene in ss];lock=threading.Lock();done=[]
 def worker(gpu):
  while True:
   with lock:
    if not queue:return
    name,ds,scene=queue.pop(0)
   variant=variants[name];folder=out/name/ds/scene
   cmd=[sys.executable,str(source/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),'--monitor',str(folder.parent/(scene+'_monitor.json')),'--log',str(out/'logs'/f'{name}_{ds}_{scene}.log'),'--',sys.executable,str(source/'scripts/evaluate_zdrop_history_states.py'),'--dataset',ds,'--scene',scene,'--method',variant['method'],'--frames','128','--warmup','1','--repeats','1','--config',str(source/variant['config']),'--output-dir',str(folder)]
   code=subprocess.call(cmd,cwd=source)
   with lock:done.append(dict(variant=name,dataset=ds,scene=scene,gpu=gpu,returncode=code));save(out/'progress.json',done)
   print(name,ds,scene,code,flush=True)
 with cf.ThreadPoolExecutor(max_workers=7) as pool:list(pool.map(worker,[0,1,2,3,4,6,7]))
 assert all(hashlib.sha256((source/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
 save(out/'completion.json',dict(runs=done,all_processes_succeeded=len(done)==108 and all(x['returncode']==0 for x in done)))
if __name__=='__main__':main()
