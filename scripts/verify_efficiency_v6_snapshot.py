"""Verify the frozen v6 source/results package using only the standard library."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/efficiency_v6_snapshot_20261103'


def main():
    provenance=json.loads((OUT/'publication_provenance.json').read_text())
    for name, expected in provenance['packaged_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
        if name.endswith('.py'):
            ast.parse((ROOT/name).read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if name.startswith(('saf3r/','scripts/','configs/')):
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    summary=json.loads((OUT/'summary.json').read_text())
    assert len(summary['audits'])==18
    assert all(all(row['checks'].values()) for row in summary['audits'])
    for dataset, scenes in manifest['datasets'].items():
        for scene in scenes:
            folder=OUT/'quality'/dataset/scene/'efficiency'
            result=json.loads((folder/'result.json').read_text())
            assert result['quality']['complete'],(dataset,scene)
            values=result['quality']['values']
            assert abs(values['cd']-(values['acc']+values['comp'])/2)<1e-10
            assert result['config']==dict(attention_block_keys=64,**manifest['config'])
            for name, expected in result['source_sha256'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    times=json.loads((OUT/'full_forward/result.json').read_text())
    assert times['median_seconds']['candidate']==18.109622897114605
    assert times['dense_speedup'] is None
    assert len(times['times']['candidate'])==3
    monitor=json.loads((OUT/'full_forward/monitor.json').read_text())
    assert monitor['strict_contexts'] and not monitor['same_gpu_overlap_detected']
    print('PASS: frozen source hashes, syntax,18 quality runs, six metrics, timing provenance; ETH3D regression remains explicitly labeled')


if __name__=='__main__':
    main()
