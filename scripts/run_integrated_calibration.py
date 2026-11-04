"""Run frozen RGB calibration candidates with strict independent GPU leases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,default=[3,4])
    a=p.parse_args();plan=json.loads(a.plan.read_text());scenes=plan['inputs']
    if len(a.gpus)!=len(scenes):raise ValueError('One GPU per calibration input')
    a.output.mkdir(parents=True,exist_ok=True)
    def worker(scene,gpu):
        rows=[]
        for name,config in plan['configs'].items():
            out=a.output/scene/name
            if (out/'result.json').exists():
                rows.append(dict(scene=scene,name=name,reused=True));continue
            if out.exists():raise ValueError('Preserve failed or interrupted attempt: '+str(out))
            prefix=a.output/(scene+'_'+name)
            command=[sys.executable,str(ROOT/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu',str(gpu),
                     '--monitor',str(prefix)+'.monitor.json','--log',str(prefix)+'.log','--',sys.executable,
                     str(ROOT/'scripts/calibrate_integrated_rgb.py'),'--input',str(ROOT/'outputs/integrated_quality_speed_20261105/calibration_inputs'/f'{scene}.json'),
                     '--method','candidate','--config',config,'--local-fidelity','--output-dir',str(out)]
            print('START',scene,name,gpu,flush=True)
            code=subprocess.call(command,cwd=ROOT)
            rows.append(dict(scene=scene,name=name,returncode=code,gpu=gpu))
            (a.output/(scene+'_progress.json')).write_text(json.dumps(rows,indent=2)+'\n')
            print('END',rows[-1],flush=True)
            if code:break
        return rows
    with ThreadPoolExecutor(max_workers=len(scenes)) as pool:
        futures=[pool.submit(worker,s,g) for s,g in zip(scenes,a.gpus)]
        rows=[r for f in futures for r in f.result()]
    (a.output/'campaign.json').write_text(json.dumps(rows,indent=2)+'\n')
    if any(r.get('returncode',0) for r in rows):sys.exit(1)


if __name__=='__main__':main()
