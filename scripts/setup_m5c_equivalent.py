"""Create an isolated optimized fork of the tested historical-state cleanup."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('/tmp/zdrop_history_cleanup_20261102'))
    parser.add_argument('--destination',type=Path,default=Path('/tmp/zdrop_m5c_equivalent_20261102'))
    args=parser.parse_args()
    if args.destination.exists():raise ValueError('Destination must be new')
    manifest=json.loads((ROOT/'outputs/zdrop_cleanup_heads128_20261102/manifest.json').read_text())
    hashes=manifest['source']['after']['sha256']
    assert all(hashlib.sha256((args.source/n).read_bytes()).hexdigest()==h for n,h in hashes.items()),'Source differs from tested cleanup'
    shutil.copytree(args.source,args.destination,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for n in ['saf3r/patch/reuse_m5c_equivalent.py','saf3r/triton/reuse_m5c_equivalent.py']:
        shutil.copy2(ROOT/n,args.destination/n)
    p=args.destination/'saf3r/patch/reuse_residual_exploration.py';s=p.read_text()
    needle='from . import reuse_m5c as m5c';assert s.count(needle)==1
    s=s.replace(needle,'from . import reuse_m5c_equivalent as m5c')
    s=s.replace("history_cache_device='cpu_grid_statistics_gpu_previous_qk'","history_cache_device='gpu_grid_statistics_gpu_previous_qk'")
    s=s.replace('history_cpu_bytes=m5c.host_bytes(s.m5c_previous),',
        'history_cpu_bytes=m5c.host_bytes(s.m5c_previous),history_calibration_gpu_bytes=m5c.device_bytes(s.m5c_previous),')
    p.write_text(s)
    evaluator=args.destination/'scripts/evaluate_zdrop_history_states.py'
    text=evaluator.read_text().replace("names=set(sources)|{", "names=set(sources)|{'saf3r/patch/reuse_m5c_equivalent.py','saf3r/triton/reuse_m5c_equivalent.py',")
    evaluator.write_text(text)
    print(args.destination)
if __name__=='__main__':main()
