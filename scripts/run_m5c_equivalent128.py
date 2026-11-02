"""Before/after same-method profile under one exclusive GPU lease."""
import hashlib,io,json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/m5c_equivalent_heads128_20261102'
def main():
 if OUT.exists():raise ValueError('Fresh output required')
 assert 'PASS SIX SAME-INPUT PROBABILITY/CACHE/NORMALIZATION CASES' in (ROOT/'outputs/m5c_equivalent_validation_20261102.log').read_text()
 OUT.mkdir();hashes={}
 for variant,folder in [('before','/tmp/zdrop_history_cleanup_20261102'),('after','/tmp/zdrop_m5c_equivalent_20261102')]:
  base=Path(folder);d={};buffer=io.BytesIO()
  with tarfile.open(fileobj=buffer,mode='w') as archive:
   for directory in ['saf3r','scripts','configs','evaluation']:
    for p in sorted((base/directory).rglob('*')):
     if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml']:
      name=str(p.relative_to(base));data=p.read_bytes();d[name]=hashlib.sha256(data).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
  (OUT/(variant+'_source.tar')).write_bytes(buffer.getvalue());hashes[variant]=dict(root=str(base),sha256=d)
 (OUT/'manifest.json').write_text(json.dumps(dict(method='repaired RMS Drop historical states: fused M5c + device FP32 calibration cache',dataset='7scenes',scene='heads',frames=128,source=hashes,unchanged='checkpoint/input/config/method/RMS/squaredL2/Drop/history policy/budget/clustering/query32',controls_executed=0,quality_evaluated=False,timing='Each variant1warmup3 uninstrumented full forwards; after one additional transparent CUDA-event profile',validation='outputs/m5c_equivalent_validation_20261102.log'),indent=2)+'\n')
 for variant in ['before','after']:
  cmd=[sys.executable,'-u',str(ROOT/'scripts/profile_m5c_equivalent128.py'),'--variant',variant,'--out',str(OUT/variant)]
  with (OUT/(variant+'.log')).open('w') as log:code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if code:raise RuntimeError(f'{variant} diagnostic failed with exit{code}')
 (OUT/'completion.json').write_text(json.dumps(dict(complete=True,controls_executed=0),indent=2)+'\n')
if __name__=='__main__':main()
