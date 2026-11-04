"""Verify reporting completeness; explicitly retain rejected equivalence results."""
import ast,hashlib,json,statistics,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';Q=OUT/'v8_p0_quality_v3_20261104'
def read(p):return json.loads(p.read_text())
def main():
 summary=read(Q/'summary.json');assert summary['complete'] and summary['source_unchanged'];assert len(summary['audits'])==18 and all(x['passed'] for x in summary['audits']);assert all(x['complete'] for x in summary['macro'])
 hashes=read(Q/'source_sha256.json')
 for name,h in hashes.items():
  p=ROOT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
  if p.suffix=='.py':ast.parse(p.read_text(),filename=name)
 with tarfile.open(Q/'source_snapshot.tar.gz') as t:
  for name,h in hashes.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==h
 for ds,scenes in read(Q/'manifest.json')['scenes'].items():
  for scene in scenes:
   folder=Q/f'quality/{ds}/{scene}';d=read(folder/'result.json');v=d['quality']['values'];assert d['quality']['complete'] and all(v[k] is not None for k in ['auc03','auc30','acc','comp','cd','f1']);assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-8
   for name,h in read(folder/'source_sha256.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 for row in read(Q/'references/index.json'):assert hashlib.sha256((Q/row['path']).read_bytes()).hexdigest()==row['sha256']
 for name in ['dense_sparse_comparison_20261104','dense_p0_v2_comparison_20261104']:
  folder=OUT/name;methods=read(folder/'completion.json')['methods'];results={m:read(folder/m/'result.json') for m in methods}
  for m,d in results.items():
   assert len(d['samples'])==3 and statistics.median(x['seconds'] for x in d['samples'])==d['median_seconds'];assert abs(d['dense_speedup']-results['dense']['median_seconds']/d['median_seconds'])<1e-12
   assert all(x is None for x in d['quality'].values());mon=read(folder/f'{m}_monitor.json');assert mon['returncode']==0 and not mon['foreign_gpu_pids'] and not mon['same_gpu_overlap_detected'] and mon['max_simultaneous_gpu_applications']==1
  assert all(x['input']['image_paths']==results['dense']['input']['image_paths'] for x in results.values())
 diag=read(OUT/'beta_regression_diagnostic_v3_20261104/result.json');assert diag['complete'] and len(diag['rows'])==24
 l1=next(r for r in diag['rows'] if r['layer']==1);assert l1['beta_previous_support_mass']>3*l1['dense_previous_support_mass'];assert l1['compensation_logbias_max']>60 and l1['sparse_cache_zero_fraction']>.98
 same=read(OUT/'attention_same_qkv_20261104/result.json');assert same['complete'] and len(same['rows'])==24
 assert not read(OUT/'v8_p0_all_windows_20261104/result.json')['complete'];mismatch=read(OUT/'v8_p0_first_mismatch_20261104/result.json');assert mismatch['complete'] and not mismatch['equivalence_passed']
 print(f'PASS: {len(hashes)} frozen hashes/archive/syntax,18 complete six-metric quality runs,54 historical controls,6 paired full-forward measurements,24-layer core/profile accounting,exclusive monitors and rejected-P0 evidence. No P0 equivalence/speed acceptance claimed.')
if __name__=='__main__':main()
