"""Evaluate frozen methods on all configured scenes without quality retuning."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3,4,5])
    p.add_argument('--control-timing',action='store_true',help='Only with explicit user approval for unchanged Dense/SAF3R timing')
    a=p.parse_args();plan=json.loads(a.plan.read_text())
    if not plan.get('frozen_before_quality'):raise ValueError('Final methods must be frozen before geometric quality')
    if a.control_timing and not plan.get('user_approved_control_timing'):raise ValueError('Explicit control timing approval missing')
    a.output.mkdir(parents=True,exist_ok=True)
    groups=[(ds,s) for ds,scenes in plan['datasets'].items() for s in scenes]
    def worker(index,gpu):
        progress=[]
        for dataset,scene in groups[index::len(a.gpus)]:
            methods=list(plan['methods'].items())
            # All methods for a scene use the same physical GPU and protocol.
            if a.control_timing and dataset=='7scenes':methods=[('dense',None),('saf3r',None)]+methods
            for name,spec in methods:
                target=a.output/dataset/scene/name
                control=spec is None
                result=target/('timing_result.json' if control else 'result.json')
                if result.exists():
                    progress.append(dict(dataset=dataset,scene=scene,method=name,reused=True));continue
                if target.exists():raise ValueError('Keep failed attempt for audit: '+str(target))
                for rel,digest in plan['model_source_sha256'].items():
                    if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen model source changed: '+rel)
                prefix=a.output/(dataset+'_'+scene+'_'+name)
                command=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),
                         '--monitor',str(prefix)+'.monitor.json','--log',str(prefix)+'.log','--',sys.executable,
                         str(ROOT/'scripts/launch_local_evaluation.py'),str(ROOT/'scripts/evaluate_zdrop_history_states.py'),
                         '--dataset',dataset,'--scene',scene,'--frames','128','--method',name if control else 'zdrop_history_states',
                         '--output-dir',str(target),'--warmup','1' if dataset=='7scenes' else '0',
                         '--repeats','3' if dataset=='7scenes' else '1']
                if control:command.append('--timing-only')
                else:
                    config=Path(spec['config'])
                    if hashlib.sha256(config.read_bytes()).hexdigest()!=spec['config_sha256']:raise ValueError('Frozen config changed')
                    command+=['--config',str(config)]
                print('START',dataset,scene,name,gpu,flush=True)
                code=subprocess.call(command,cwd=ROOT)
                row=dict(dataset=dataset,scene=scene,method=name,gpu=gpu,returncode=code,result=str(result))
                if result.exists() and not control:row['quality_complete']=json.loads(result.read_text())['quality']['complete']
                progress.append(row)
                (a.output/f'worker_{gpu}_progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                print('END',row,flush=True)
                if code:break
        return progress
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures=[pool.submit(worker,i,g) for i,g in enumerate(a.gpus)]
        rows=[r for f in futures for r in f.result()]
    (a.output/'campaign.json').write_text(json.dumps(rows,indent=2)+'\n')
    if any(r.get('returncode',0) or r.get('quality_complete') is False for r in rows):sys.exit(1)


if __name__=='__main__':main()
