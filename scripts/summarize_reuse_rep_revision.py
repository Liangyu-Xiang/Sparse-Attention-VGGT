"""Audit the complete combined revision and reused reference results."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.run_reuse_rep_revision import SCENES,METHODS,report
from scripts.evaluate_attention_first_residual_value import save

def summarize(root):
 report(root)
 # Historical per-scene speedup belongs to its own earlier Dense pairing.
 # Keep it in raw source JSON, but do not display it as this cohort's pairing.
 report_path=root/'report.md'
 lines=report_path.read_text().splitlines()
 for i,line in enumerate(lines):
  if line.startswith('| ') and any('| '+m+' |' in line for m in ('original','cosine_distortion')):
   cells=line.split('|')
   if len(cells)>10:cells[-2]=' N/A (historical) ';lines[i]='|'.join(cells)
 report_path.write_text('\n'.join(lines)+'\n')
 data=json.loads((root/'summary.json').read_text());macro={x['method']:x for x in data['macro']}
 audit=[];behavior=[]
 for scene in SCENES:
  base=root/scene/'dense/result.json'
  if not base.exists():continue
  reference=json.loads(base.read_text())
  for method in METHODS:
   path=root/scene/method/'result.json'
   if not path.exists():continue
   run=json.loads(path.read_text())
   same={key:run[key]==reference[key] for key in ('input','hardware','quality_protocol')}
   same['timing_protocol']=all(run['timing'][k]==reference['timing'][k] for k in ('warmup','repeats','capture'))
   historical=method in ('original','cosine_distortion')
   shared=[k for k in reference['source_sha256'] if k.startswith('saf3r/') and k in run['source_sha256']]
   same['core_source']=historical or all(run['source_sha256'][k]==reference['source_sha256'][k] for k in shared)
   if method!='saf3r_dino64' and not historical:same['reuse_source_coverage']=all(k in run['source_sha256'] for k in reference['source_sha256'] if k.startswith('saf3r/'))
   audit.append(dict(scene=scene,method=method,reused_historical=historical,checks=same,passed=all(same.values())))
   layers=root/scene/method/'layers.json'
   if layers.exists():
    rows=json.loads(layers.read_text())
    for row in rows:
     f=row['fine_clusters'];budget=row['patch_budget']
     behavior.append(dict(scene=scene,method=method,layer=row['layer'],
      history_per_cluster=row['reuse_edges_mean']/f,
      dropped_fraction=row['dropped_clusters_mean']/f,
      rep_fraction=row['representatives_mean']/f,
      exact_edge_fraction=row['exact_edges_mean']/budget,
      unused_budget=row['unused_budget_mean'],
      **{k:v for k,v in row.items() if any(term in k for term in ('observed_history','history_rep_overlap','history_candidates','gain_','_gain_','state_'))}))
 pairs=[('rep_combined','cosine_distortion')]
 comparisons=[]
 lines=['# Complete combined Rep-baseline revision comparison','',
  'Original and previous cosine QUALITY results are reused, with identical input/protocol. Their historical timings are not contemporaneous comparisons. Only the complete combined revision is evaluated; superseded staged runs are excluded.',
  'AUC/F1 are percentages; ACC/Comp/CD are unsquared metres. CD=(ACC+Comp)/2.',
  '7Scenes all seven scenes, uniform 128 frames, F1 threshold 0.05 m, voxel 0.0078125 m.',
  'SAF3R uses DINO per-frame Top-64 with the ETH3D calibrated profile, distinct from lazy first-correspondence Top-4.',
  '', '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Warm full-forward s | Peak GiB | Dense speedup |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for method in METHODS:
  item=macro[method];v=item['values']
  if v is None:continue
  cells=[f"{v[k]*(100 if k in ('auc03','auc30','f1') else 1):.5f}" for k in ('auc03','auc30','acc','comp','cd','f1')]
  lines.append('| '+' | '.join([method]+cells+[f"{item['mean_scene_median_seconds']:.3f}",f"{item['mean_scene_peak_allocated_gib']:.3f}",('N/A (historical)' if method in ('original','cosine_distortion') else f"{item['speedup_ratio_of_mean_scene_medians']:.3f}x")])+' |')
 lines+=['','## Combined revision against the previous cosine implementation','','| Method | Reference | dAUC@3 pp | dAUC@30 pp | dCD m | dF1 pp | Reference/stage time |','|---|---|---:|---:|---:|---:|---:|']
 for method,reference in pairs:
  a,b=macro[method],macro[reference]
  if a['values'] is None or b['values'] is None:continue
  diff={k:a['values'][k]-b['values'][k] for k in a['values']}
  ratio=None if reference=='cosine_distortion' else b['mean_scene_median_seconds']/a['mean_scene_median_seconds']
  comparisons.append(dict(method=method,reference=reference,delta=diff,runtime_ratio=ratio))
  lines.append(f"| {method} | {reference} | {100*diff['auc03']:+.5f} | {100*diff['auc30']:+.5f} | {diff['cd']:+.5f} | {100*diff['f1']:+.5f} | {('N/A (historical)' if ratio is None else format(ratio,'.3f')+'x')} |")
 lines+=['','## State diagnostics','','Equal-scene/equal-layer averages; seed layer included. History is selected non-Rep history state edges, not the full observed bank. Exact edge fraction uses budget as denominator.',
  '', '| Method | History / cluster | Drop % | Rep / clusters % | Exact edges / budget % | Unused budget | Observed previous keys | Candidate keys | Rep overlap |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 means=[]
 for method in METHODS:
  rows=[r for r in behavior if r['method']==method]
  if len(rows)!=7*24:continue
  fields=['history_per_cluster','dropped_fraction','rep_fraction','exact_edge_fraction','unused_budget']
  out=dict(method=method,**{k:float(np.mean([r[k] for r in rows])) for k in fields})
  for k in ('observed_history_mean','history_candidates_mean','history_rep_overlap_mean'):
   out[k]=float(np.mean([r[k] for r in rows])) if all(k in r for r in rows) else None
  means.append(out)
  cells=[f"{out[k]*(100 if 'fraction' in k else 1):.5f}" for k in fields]
  cells += ['N/A' if out[k] is None else f"{out[k]:.2f}" for k in ('observed_history_mean','history_candidates_mean','history_rep_overlap_mean')]
  lines.append('| '+' | '.join([method]+cells)+' |')
 save(root/'pair_audit.json',audit)
 save(root/'stage_comparisons.json',comparisons)
 save(root/'state_behavior.json',dict(per_scene_layer=behavior,macro=means))
 complete=sum(x['quality_complete'] for x in data['per_run'] if x['method'].startswith('rep_'))
 save(root/'completion.json',dict(required_new_runs=7,complete_new_quality=complete,all_new_complete=complete==7,reference_rows=28,
  matched_audit_passed=len(audit)==35 and all(x['passed'] for x in audit),scope='7Scenes all seven scenes, 128 frames, pose+recon_unposed'))
 (root/'combined_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output_dir',type=Path);args=p.parse_args();summarize(args.output_dir)
