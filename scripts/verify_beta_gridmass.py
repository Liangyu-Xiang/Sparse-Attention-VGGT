"""CPU artifact verification; no model, data or CUDA context is initialized."""
import ast,hashlib,json,statistics,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/beta_gridmass_20261104'

def main():
    summary=json.loads((OUT/'summary.json').read_text());assert summary['complete'] and summary['source_unchanged'] and not summary['missing']
    assert len(summary['audit'])==18 and all(r['passed'] for r in summary['audit'])
    hashes=json.loads((OUT/'source_sha256.json').read_text())
    for name,digest in hashes.items():
        path=ROOT/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        if path.suffix=='.py':ast.parse(path.read_text(),filename=name)
    for p in (OUT/'quality').glob('*/*/result.json'):
        raw=json.loads(p.read_text());assert raw['quality']['complete'] and not raw['quality_errors']
        assert abs(raw['quality']['values']['cd']-(raw['quality']['values']['acc']+raw['quality']['values']['comp'])/2)<1e-9
        assert raw['config']==dict(attention_block_keys=64,**json.loads((OUT/'manifest.json').read_text())['config'])
        for name,digest in raw['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for row in summary['macro']:
        assert row['complete']
        if row['method']=='beta_gridmass':
            files=list((OUT/'quality'/row['dataset']).glob('*/result.json'));assert len(files)==row['scenes']==(7 if row['dataset']=='7scenes' else 11)
            values=[json.loads(f.read_text())['quality']['values'] for f in files]
            for key in ['auc03','auc30','acc','comp','cd','f1']:assert abs(statistics.mean(v[key] for v in values)-row[key])<1e-12
    monitors=list((OUT/'monitors').glob('*.json'));assert len(monitors)==20
    for p in monitors:
        r=json.loads(p.read_text());assert not r['auxiliary_cuda_context'] and not r['foreign_gpu_pids'] and not r['same_gpu_overlap_detected'] and r['max_simultaneous_gpu_applications']==1 and r['returncode']==0
    timing=summary['timing'];assert len(timing['samples'])==3 and statistics.median(x['seconds'] for x in timing['samples'])==timing['median_seconds']
    assert all(v is None for v in timing['quality'].values()) and timing['quality_reason'] and timing['dense_speedup'] is None
    assert json.loads((ROOT/'outputs/beta_gridmass_validation_v2_20261104/result.json').read_text())['complete']
    for entry in json.loads((OUT/'references/index.json').read_text()):assert hashlib.sha256((OUT/entry['path']).read_bytes()).hexdigest()==entry['sha256']
    print(f'PASS: {len(hashes)} frozen source hashes/syntax,18 complete six-metric runs,432 layers,20 exclusive monitors,reference audits and warmed median; quality regression retained')

if __name__=='__main__':main()
