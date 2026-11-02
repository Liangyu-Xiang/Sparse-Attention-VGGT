"""Only the changed ready-history method is run; reference is historical."""
import hashlib,io,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/ready_history_v4_heads128_20261102'

def main():
 assert 'PASS READY HISTORY' in (ROOT/'outputs/ready_history_validation_20261102.log').read_text()
 if OUT.exists():raise ValueError('Fresh output required')
 OUT.mkdir();prior=ROOT/'outputs/m5c_pipeline_dynamic_heads128_20261102'
 shutil.copytree(prior/'after',OUT/'before');shutil.copy2(prior/'after_source.tar',OUT/'before_source.tar')
 old=json.loads((prior/'manifest.json').read_text());base=Path('/tmp/zdrop_ready_history_v4_20261102');hashes={};buffer=io.BytesIO()
 with tarfile.open(fileobj=buffer,mode='w') as archive:
  for folder in ['saf3r','scripts','configs','evaluation']:
   for p in sorted((base/folder).rglob('*')):
    if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml']:
     name=str(p.relative_to(base));data=p.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();item=tarfile.TarInfo(name);item.size=len(data);archive.addfile(item,io.BytesIO(data))
 (OUT/'after_source.tar').write_bytes(buffer.getvalue())
 manifest=dict(method='Ready previous-layer conditionals, fused attention probe emission, FP16 edge storage',source=dict(before=old['source']['after'],after=dict(root=str(base),sha256=hashes)),reference=str(prior/'after/forward.json'),comparison='Historical reference only, not contemporaneous paired timing',cache_quantization='FP16 per-head/probe edge conditionals; FP32 calculation and grid data',whole_grid_pooling_unchanged=True,allocator='expandable_segments:True',controls_executed=0,quality_evaluated=False)
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 tensors=Path('/tmp/ready_history_outputs_20261102');tensors.mkdir(exist_ok=True)
 if not (tensors/'before_first.pt').exists():os.link('/tmp/m5c_pipeline_dynamic_outputs_20261102/after_first.pt',tensors/'before_first.pt')
 with (OUT/'after.log').open('w') as log:
  code=subprocess.call([sys.executable,'-u',str(ROOT/'scripts/profile_ready_history_v4_128.py'),'--variant','after','--out',str(OUT/'after')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
 (OUT/'completion.json').write_text(json.dumps(dict(complete=code==0,returncode=code,controls_executed=0),indent=2)+'\n')
 if code:raise RuntimeError(f'Ready-history diagnostic exit {code}')
if __name__=='__main__':main()
