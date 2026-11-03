"""Exclusive-GPU timing and sampled Attention references; no quality GT."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--quality-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=True,parents=True);source=a.quality_root/'source';parent=a.quality_root.parent
 variants=['calibration','simple','all_exact','full','global','single_probe'];done=[]
 for name in variants:
  cases=[('timing','benchmark_light_reuse.py','7scenes','heads',['--profile'])]+[('diagnostic','diagnose_light_reuse.py',ds,scene,[]) for ds,scene in [('7scenes','heads'),('eth3d','terrains')]]
  for kind,script,ds,scene,extra in cases:
   folder=a.output_dir/name/(kind+'_'+ds+'_'+scene)
   input_file=parent/'efficiency_quality_v8_20261103'/ds/scene/'efficiency/input.json'
   cmd=[sys.executable,str(source/'scripts/run_gpu_lease.py'),'--strict-contexts','--gpu','5','--monitor',str(folder.parent/(folder.name+'_monitor.json')),'--log',str(folder.parent/(folder.name+'.log')),'--',sys.executable,str(source/'scripts'/script),'--config',str(source/f'configs/evaluation/reuse_light_{name}.json'),'--input',str(input_file),'--output-dir',str(folder),*extra]
   code=subprocess.call(cmd,cwd=source);done.append(dict(variant=name,kind=kind,dataset=ds,scene=scene,returncode=code));(a.output_dir/'progress.json').write_text(json.dumps(done,indent=2)+'\n');print(name,kind,ds,scene,code,flush=True)
 (a.output_dir/'completion.json').write_text(json.dumps(dict(runs=done,all_processes_succeeded=all(x['returncode']==0 for x in done)),indent=2)+'\n')
if __name__=='__main__':main()
