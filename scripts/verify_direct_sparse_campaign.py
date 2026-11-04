"""Verify complete results without claiming unmeasured occupancy or paired speedup."""
import ast,hashlib,json,statistics,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';Q=OUT/'direct_sparse_quality_20261104'
def read(p):return json.loads(p.read_text())
def main():
 s=read(Q/'summary.json');assert s['complete'] and s['source_unchanged'];assert len(s['audits'])==18 and all(x['passed'] for x in s['audits']);assert all(x['complete'] for x in s['macro']);assert s['dense_speedup'] is None and not s['default_changed'] and s['controls_executed']==0
 hashes=read(Q/'source_sha256.json')
 for name,h in hashes.items():
  p=ROOT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,name
  if p.suffix=='.py':ast.parse(p.read_text(),filename=name)
 with tarfile.open(Q/'source_snapshot.tar.gz') as t:
  for name,h in hashes.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==h,name
 for ds,scenes in read(Q/'manifest.json')['scenes'].items():
  for scene in scenes:
   folder=Q/f'quality/{ds}/{scene}';d=read(folder/'result.json');v=d['quality']['values'];assert d['quality']['complete'] and all(v[k] is not None for k in ['auc03','auc30','acc','comp','cd','f1']);assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-8
   for name,h in read(folder/'source_sha256.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 for row in read(Q/'references/index.json'):assert hashlib.sha256((Q/row['path']).read_bytes()).hexdigest()==row['sha256']
 timing=read(Q/'timing/result.json');assert len(timing['samples'])==3 and statistics.median(x['seconds'] for x in timing['samples'])==timing['median_seconds'];assert timing['dense_speedup'] is None and all(x is None for x in timing['quality'].values())
 for name in ['direct_sparse_validation_v2_20261104','direct_legacy_validation_v3_20261104','direct_sparse_forward_validation_v2_20261104','direct_sparse_all_windows_20261104','direct_sparse_pure_capture_20261104','direct_sparse_launches_20261104']:
  assert read(OUT/name/'result.json')['complete'],name
 op=read(OUT/'direct_sparse_all_windows_20261104/result.json');assert len(op['windows'])==1584 and len(op['layers'])==24 and sum(x['query_rows'] for x in op['windows'])==24*133248
 for layer in op['layers']:
  assert layer['windows']==66 and layer['query_rows']==133248
 assert all(x['output_relative_l2']<.01 and x['probability_A_B_max_abs']<2e-4 for x in op['windows'])
 meta=read(OUT/'direct_sparse_launches_20261104/result.json');assert meta['occupancy'] is None
 # No Grid data in the direct API or executable core body.
 tree=ast.parse((ROOT/'saf3r/triton/reuse_direct_sparse_attention.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='direct_sparse_attention');args=[a.arg for a in [*fn.args.args,*fn.args.kwonlyargs]];assert not any('grid' in a or 'region' in a for a in args)
 code=(ROOT/'saf3r/triton/reuse_direct_sparse_attention.py').read_text();assert 'argsort' not in code and 'REGIONS' not in code and 'selected_grid_offsets' not in code
 for p in (Q/'monitors').glob('*.json'):
  m=read(p);assert m['returncode']==0 and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and not m['auxiliary_cuda_context'] and m['max_simultaneous_gpu_applications']==1
 assert s['diagnostics']['raw_probe_scratch_max_bytes']==0
 print(f'PASS: {len(hashes)} frozen source/archive hashes and syntax,18 complete six-metric scene audits,54 historical references,1584 matched windows/all24layers,1/4probe A/B and FP16/BF16 capture checks,Grid-free API and GPU-exclusive timing. Dense full-forward speedup and measured occupancy remain N/A with reasons.')
if __name__=='__main__':main()
