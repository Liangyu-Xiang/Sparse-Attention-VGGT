"""Corrected runtime-COUNT candidate; reuse the completed same-batch reference."""
import hashlib,io,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/m5c_pipeline_dynamic_heads128_20261102'
PREVIOUS=ROOT/'outputs/m5c_pipeline_heads128_20261102'


def main():
    assert 'PASS TWELVE PIPELINE' in (ROOT/'outputs/m5c_pipeline_dynamic_validation_20261102.log').read_text()
    assert 'PASS' in (ROOT/'outputs/drop_moment_dynamic_20261102.log').read_text()
    if OUT.exists():raise ValueError('Fresh output required')
    OUT.mkdir()
    shutil.copytree(PREVIOUS/'before',OUT/'before')
    shutil.copy2(PREVIOUS/'before_source.tar',OUT/'before_source.tar')
    manifest=json.loads((PREVIOUS/'manifest.json').read_text())
    manifest.update(method='M5c pipeline + exact binned Drop, runtime bucket COUNT',
        reused_reference_timing=str(PREVIOUS/'before/forward.json'),
        reference_reuse_reason='Completed same-batch same-GPU same-input 1-warmup/3-repeat reference; do not rerun unchanged reference')
    base=Path('/tmp/zdrop_m5c_pipeline_dynamic_20261102');hashes={};buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as archive:
        for folder in ['saf3r','scripts','configs','evaluation']:
            for p in sorted((base/folder).rglob('*')):
                if p.is_file() and p.suffix in ['.py','.json','.yaml','.yml']:
                    name=str(p.relative_to(base));data=p.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    (OUT/'after_source.tar').write_bytes(buffer.getvalue())
    manifest['source']['after']=dict(root=str(base),sha256=hashes)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    tensors=Path('/tmp/m5c_pipeline_dynamic_outputs_20261102');tensors.mkdir(exist_ok=True)
    if not (tensors/'before_first.pt').exists():
        os.link('/tmp/m5c_pipeline_outputs_20261102/before_first.pt',tensors/'before_first.pt')
    command=[sys.executable,'-u',str(ROOT/'scripts/profile_m5c_pipeline_dynamic128.py'),'--variant','after','--out',str(OUT/'after')]
    with (OUT/'after.log').open('w') as log:
        code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if code:raise RuntimeError(f'Corrected candidate diagnostic failed: {code}')
    (OUT/'completion.json').write_text(json.dumps(dict(complete=True,controls_executed=0,reference_reused=True),indent=2)+'\n')


if __name__=='__main__':main()
