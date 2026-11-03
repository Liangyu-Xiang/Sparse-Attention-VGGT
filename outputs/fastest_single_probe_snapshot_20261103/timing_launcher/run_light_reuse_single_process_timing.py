"""One fixed sequence, one benchmark process on a GPU, serial variants."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--outputs',type=Path,required=True);p.add_argument('--gpu',type=int,default=5);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.outputs/'light_reuse_single_process_timing_20261103';out.mkdir(exist_ok=True);done=[]
 for name in ['calibration','simple','all_exact','full','global','single_probe']:
  folder=out/name
  cmd=[sys.executable,str(root/'scripts/run_gpu_lease.py'),'--single-process','--strict-contexts','--gpu',str(a.gpu),'--monitor',str(out/(name+'_monitor.json')),'--log',str(out/(name+'.log')),'--',sys.executable,str(a.source/'scripts/benchmark_light_reuse.py'),'--config',str(a.source/f'configs/evaluation/reuse_light_{name}.json'),'--input',str(a.outputs/'efficiency_quality_v8_20261103/7scenes/heads/efficiency/input.json'),'--output-dir',str(folder),'--profile']
  code=subprocess.call(cmd,cwd=a.source);done.append(dict(variant=name,returncode=code));(out/'progress.json').write_text(json.dumps(done,indent=2));print(name,code,flush=True)
 (out/'completion.json').write_text(json.dumps(dict(runs=done,complete=all(r['returncode']==0 for r in done)),indent=2))
if __name__=='__main__':main()
