import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];keys=['auc03','auc30','acc','comp','cd','f1']
for folder in (ROOT/'outputs').iterdir():
 if not folder.is_dir() or not folder.name.startswith(('direct_sparse_','direct_legacy_')):continue
 for p in folder.rglob('result.json'):
  if 'quality' in p.relative_to(folder).parts or 'references' in p.relative_to(folder).parts:continue
  d=json.loads(p.read_text());q=d.get('quality',{})
  if not all(k in q and q[k] is None for k in keys):continue
  if 'peak_allocated_gib' not in d and 'peak_memory_gib' not in d:d.update(peak_memory_gib=None,peak_reason='Numerical/availability diagnostic did not isolate peak;full-forward/all-window diagnostics record peak separately')
  if not any(k in d for k in ['runtime_seconds','diagnostic_wall_seconds','median_seconds']):d.update(runtime_seconds=None,runtime_reason='Non-benchmark diagnostic;no trustworthy isolated runtime recorded')
  p.write_text(json.dumps(d,indent=2)+'\n')
  (p.parent/'diagnostic.md').write_text('|AUC@3 fraction|AUC@30 fraction|ACC m|Comp m|CD m|F1 fraction|\n|---|---|---|---|---|---|\n|N/A|N/A|N/A|N/A|N/A|N/A|\n\n'+str(d.get('quality_reason','Diagnostic,no geometry evaluation'))+'\n\nDense full-forward speedup N/A: '+str(d.get('dense_reason','No paired timing'))+'\n')
print('Direct diagnostic six-field availability tables written.')
