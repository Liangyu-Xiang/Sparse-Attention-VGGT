"""Verify published raw metrics, frozen runtime source and single-GPU monitors."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gpt6_history_fix_eval_20261103'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['complete'] and summary['source_unchanged'] and not summary['missing']
    hashes=json.loads((OUT/'source_sha256.json').read_text())
    for name,digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert len(summary['protocol_audit'])==36 and all(x['passed'] for x in summary['protocol_audit'])
    assert len(summary['monitor_audit'])==40 and all(x['passed'] for x in summary['monitor_audit'])
    metrics=['auc03','auc30','acc','comp','cd','f1']
    runs=list((OUT/'quality').glob('*/*/*/result.json'))
    assert len(runs)==36
    for path in runs:
        raw=json.loads(path.read_text())
        assert raw['quality']['complete'] and not raw['quality_errors'],path
        values=raw['quality']['values']
        assert all(values[k] is not None for k in metrics)
        assert abs(values['cd']-(values['acc']+values['comp'])/2)<1e-10
        assert raw['quality_protocol']['tasks']==['pose','recon_unposed']
        for name,digest in raw['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,(str(path),name)
        layers=json.loads((path.parent/'layers.json').read_text())
        assert len(layers)==24 and all(x['counts_mean']<=x['patch_budget'] for x in layers)
    for macro in summary['macro']:
        if macro['method'] not in {'single_probe_repaired','full_repaired'}:
            continue
        values=[json.loads(p.read_text())['quality']['values'] for p in (OUT/'quality'/macro['method']/macro['dataset']).glob('*/result.json')]
        assert len(values)==macro['scenes']==(7 if macro['dataset']=='7scenes' else 11)
        for key in metrics:
            assert abs(statistics.mean(x[key] for x in values)-macro[key])<1e-12
    for timing in summary['timing']:
        assert len(timing['samples'])==3 and statistics.median(timing['samples'])==timing['median_seconds']
        assert timing['gpu']==7 and timing['dense_speedup'] is None
    for reference in json.loads((OUT/'references/index.json').read_text()):
        assert hashlib.sha256((OUT/reference['packaged']).read_bytes()).hexdigest()==reference['sha256']
    detail=summary['detailed_budget_profile']
    assert detail['complete'] and abs(sum(detail['exclusive_seconds'].values())-detail['wall_seconds'])<1e-8
    assert detail['instrumentation_sha256']==hashlib.sha256((ROOT/'scripts/profile_history_fix_budget.py').read_bytes()).hexdigest()
    print(f'PASS: {len(hashes)} frozen source hashes,36 complete six-metric runs,864 layer records,40 exclusive monitors,paired input audits and medians')


if __name__=='__main__':
    main()
