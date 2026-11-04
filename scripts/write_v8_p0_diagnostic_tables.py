"""Human-readable six-field tables for every diagnostic, including failures."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';KEYS=['auc03','auc30','acc','comp','cd','f1']
for folder in OUT.iterdir():
 if folder.is_dir() and (folder.name.startswith(('v8_p0_','beta_regression_diagnostic','attention_same_qkv','original_v8_graph','dense_sparse_comparison','dense_p0_v2_comparison'))):
  for p in folder.rglob('result.json'):
   if 'quality' in p.relative_to(folder).parts or 'references' in p.relative_to(folder).parts:continue
   d=json.loads(p.read_text());q=d.get('quality');q=q.get('values',q) if isinstance(q,dict) else None
   if not isinstance(q,dict) or not all(k in q for k in KEYS):continue
   reason=d.get('quality_reason',d.get('reason','Diagnostic only; no GT quality evaluation'))
   text=['# Diagnostic quality availability','', '| AUC@3 fraction | AUC@30 fraction | ACC m | Comp m | CD m | F1 fraction |','|---|---|---|---|---|---|','| '+' | '.join('N/A' if q[k] is None else str(q[k]) for k in KEYS)+' |','',reason,'','Diagnostic completion: '+str(d.get('complete'))+'. This does not imply a complete quality evaluation.','', 'End-to-end Dense speedup: '+str(d.get('dense_speedup'))+'. '+str(d.get('dense_reason',''))]
   (p.parent/'diagnostic.md').write_text('\n'.join(text)+'\n')
# Initial environment failure happened before evaluator output setup.
failed=OUT/'v8_p0_quality_20261104'
if failed.exists():
 manifest=json.loads((failed/'manifest.json').read_text())
 for ds,scenes in manifest['scenes'].items():
  for scene in scenes:
   p=failed/f'quality/{ds}/{scene}';p.mkdir(exist_ok=True,parents=True)
   d=dict(complete=False,dataset=ds,scene=scene,quality=dict(complete=False,values={k:None for k in KEYS},unavailable={k:'Open3D import failed: missing libX11.so.6 loader path; no forward or GT evaluation' for k in KEYS}),timing=None,timing_reason='Environment import failed before forward',dense_speedup=None,dense_reason='No timing measured')
   (p/'result.json').write_text(json.dumps(d,indent=2)+'\n');(p/'quality.md').write_text('|AUC@3 fraction|AUC@30 fraction|ACC m|Comp m|CD m|F1 fraction|\n|---|---|---|---|---|---|\n|N/A|N/A|N/A|N/A|N/A|N/A|\n\nOpen3D import failed: missing libX11.so.6 loader path; no forward or GT evaluation.\n')
print('Diagnostic six-field tables written; failed environment runs retained.')
