"""CPU-only verification; does not require Torch, CUDA or external datasets."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ARTIFACT=ROOT/'outputs/v8_five_bucket_profile_20261104'


def main():
    manifest=json.loads((ARTIFACT/'publication.json').read_text())
    for name,digest in manifest['repository_files'].items():
        path=ROOT/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        if path.suffix=='.py':ast.parse(path.read_text(),filename=name)
    hashes=json.loads((ARTIFACT/'original_source_sha256.json').read_text())
    with tarfile.open(ARTIFACT/'v8_source_snapshot.tar.gz','r:gz') as archive:
        members={member.name:member for member in archive.getmembers()}
        assert set(hashes)==set(members)
        for name,digest in hashes.items():
            assert members[name].isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts,name
            assert hashlib.sha256(archive.extractfile(members[name]).read()).hexdigest()==digest,name
    raw=json.loads((ARTIFACT/'result/profile.json').read_text())
    summary=json.loads((ARTIFACT/'summary.json').read_text())
    assert abs(sum(x['seconds'] for x in summary['categories'])-summary['sparse_chain_seconds'])<1e-7
    assert abs(summary['sparse_chain_seconds']+summary['base_model_and_outside_global_seconds']-raw['wall_seconds'])<1e-7
    assert abs(sum(x['percent_sparse_chain'] for x in summary['categories'])-100)<1e-7
    assert abs(sum(x['percent_full_forward'] for x in summary['categories'])+summary['base_model_percent_full_forward']-100)<1e-7
    assert len(summary['per_layer'])==24
    for category in summary['categories']:
        assert abs(sum(row[category['category']+'_ms'] for row in summary['per_layer'])/1000-category['seconds'])<1e-7
    assert len(raw['graph_windows'])==1522 and summary['logical_selection_windows']==1518
    for row in raw['graph_windows']:
        assert abs(sum(row[k+'_seconds'] for k in ['gate','loss','selection','pack','transfer','diagnostics'])-row['total_seconds'])<1e-8
    monitor=json.loads((ARTIFACT/'monitor.json').read_text())
    assert monitor['gpu']==7 and not monitor['auxiliary_cuda_context']
    assert monitor['max_simultaneous_gpu_applications']==1 and not monitor['same_gpu_overlap_detected']
    assert not monitor['foreign_gpu_pids'] and monitor['returncode']==0 and not monitor['aborted']
    assert json.loads((ARTIFACT/'result/clock_validation.json').read_text())['passed']
    assert all(v is None for v in raw['quality'].values()) and raw['quality_reason']
    assert raw['dense_speedup'] is None and raw['dense_reason']
    assert manifest['proposed_simplification_status']=='documented_not_implemented_not_evaluated'
    assert raw['instrumentation_sha256']==hashlib.sha256((ARTIFACT/'diagnostics/profile_v8_five_buckets.py').read_bytes()).hexdigest()
    print(f"PASS: {len(hashes)} archived source hashes, package hashes/syntax,24 layers,1522 graph measurements,single-CUDA-process monitor,100% accounting; quality and Dense speedup correctly N/A")


if __name__=='__main__':main()
