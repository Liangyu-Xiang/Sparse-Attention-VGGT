"""CPU-only verification of the published experimental snapshot."""
import ast
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'outputs/fastest_single_probe_snapshot_20261103'


def main():
    hashes = json.loads((ARTIFACT / 'published_source_sha256.json').read_text())
    for name, digest in hashes.items():
        path = ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
    summary = json.loads((ARTIFACT / 'summary.json').read_text())
    config = json.loads((ROOT / 'configs/evaluation/reuse_light_single_probe.json').read_text())
    for macro in summary['macro']:
        results = list((ARTIFACT / 'quality' / macro['dataset']).glob('*/result.json'))
        assert len(results) == macro['scenes'] == (7 if macro['dataset'] == '7scenes' else 11)
        values = []
        for path in results:
            result = json.loads(path.read_text())
            assert result['quality']['complete'] and not result['quality_errors'], path
            assert result['config'] == config, path
            metrics = result['quality']['values']
            assert all(metrics[k] is not None for k in ['auc03', 'auc30', 'acc', 'comp', 'cd', 'f1'])
            assert abs(metrics['cd'] - (metrics['acc'] + metrics['comp']) / 2) < 1e-10
            values.append(metrics)
        for key in values[0]:
            assert abs(statistics.mean(x[key] for x in values) - macro[key]) < 1e-12, key
    audit = json.loads((ARTIFACT / 'protocol_audit.json').read_text())
    assert len(audit) == 18 and all(x['passed'] for x in audit)
    timing = json.loads((ARTIFACT / 'timing/single_probe/result.json').read_text())
    monitor = json.loads((ARTIFACT / 'timing/single_probe_monitor.json').read_text())
    assert monitor['single_process'] and monitor['strict_contexts']
    assert not monitor['same_gpu_overlap_detected'] and not monitor['foreign_gpu_pids']
    assert not monitor['aborted'] and monitor['returncode'] == 0
    assert len(timing['samples']) == 3
    assert statistics.median(x['seconds'] for x in timing['samples']) == timing['median_seconds']
    assert timing['config'] == config and timing['dense_speedup'] is None
    assert all(v is None for v in timing['quality'].values()) and timing['quality_reason']
    print(f"PASS: {len(hashes)} source hashes, syntax, 18 quality runs, macros, protocol audits, single-process timing {timing['median_seconds']:.9f}s")


if __name__ == '__main__':
    main()
