"""Final seven-scene compact comparison with explicitly reused controls."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scripts.run_reuse_compact_eval import SCENES,METHODS,report
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record,quality_cells

def summarize(root):
 report(root);summary=json.loads((root/'summary.json').read_text());macro={x['method']:x for x in summary['macro']}
 runs=[];audit=[];snapshots=[];source_ref=None
 for scene in SCENES:
  out=root/scene/'rep_compact';path=out/'result.json'
  if not path.exists():continue
  run=json.loads(path.read_text());ref=json.loads((root/scene/'rep_combined/result.json').read_text())
  checks={k:run[k]==ref[k] for k in ('input','hardware','quality_protocol')}
  checks['timing_protocol']=all(run['timing'][k]==ref['timing'][k] for k in ('warmup','repeats','capture'))
  keys=(set(run['config'])|set(ref['config']))-{'compact','window_groups'}
  checks['algorithm_config']=all(run['config'].get(k)==ref['config'].get(k) for k in keys)
  sha=run['source_sha256']
  if source_ref is None:source_ref=sha
  checks['same_new_sources']=sha==source_ref
  audit.append(dict(scene=scene,checks=checks,passed=all(checks.values())))
  for name,expected in sha.items():
   p=out/'source_snapshot'/name;snapshots.append(dict(scene=scene,path=name,passed=p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==expected))
  runs.append(run)
 complete=sum(r['quality']['complete'] for r in runs)
 save(root/'completion.json',dict(required_new_runs=7,complete_new_quality=complete,all_new_complete=complete==7,all_prior_methods_reused=True,
  matched_input_protocol_config_audit_passed=len(audit)==7 and all(x['passed'] for x in audit),snapshot_hashes_checked=len(snapshots),snapshot_audit_passed=bool(snapshots) and all(x['passed'] for x in snapshots)))
 save(root/'pair_audit.json',audit);save(root/'source_snapshot_audit.json',snapshots)
 lines=['# Complete combined revision: compact acceleration','',
  '7Scenes all seven scenes, uniform128 frames, identical checkpoint/input/preprocessing/pose/recon_unposed protocol.',
  'AUC/F1 in percent; ACC/Comp/CD unsquared metres. CD=(ACC+Comp)/2. F1 threshold0.05m; voxel0.0078125m.',
  'Only rep_compact is newly evaluated. All prior methods, including Dense/SAF3R and the unoptimized combined implementation, are reused as requested.',
  'Time ratios use stored matched same-input/GPU/warmup/repeat protocol medians; they are NOT contemporaneously remeasured control comparisons. No parameters selected using test truth.',
  'SAF3R DINO per-frame Top-64 uses the ETH3D calibrated profile, distinct from lazy first-correspondence Top-4.',
  '', '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % | Full-forward seconds | Peak GiB | Ratio vs stored Dense |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 dense=macro['dense']
 for method in METHODS:
  item=macro[method]
  if item['values'] is None:continue
  q=quality_record({});q['values']=item['values']
  ratio=dense['mean_scene_median_seconds']/item['mean_scene_median_seconds']
  if method in ('original','cosine_distortion'):ratio_cell='N/A (historical cohort)'
  else:ratio_cell=f'{ratio:.4f}x'+(' (stored)' if method!='dense' else '')
  lines.append('| '+' | '.join([method]+quality_cells(q)+[f"{item['mean_scene_median_seconds']:.4f}",f"{item['mean_scene_peak_allocated_gib']:.3f}",ratio_cell])+' |')
 comparisons={}
 if complete==7:
  item=macro['rep_compact'];v=item['values']
  q=quality_record(dict(pose=dict(auc03=v['auc03'],auc30=v['auc30']),recon_unposed=dict(acc=v['acc'],comp=v['comp'],overall=v['cd'],fscore=v['f1'])))
  ratio=dense['mean_scene_median_seconds']/item['mean_scene_median_seconds']
  q.update(method='rep_compact',speedup_vs_vggt=ratio,speedup_unavailable=None,timing_reference='Stored same-input/GPU/protocol Dense timing, not remeasured')
  write_quality(root,q)
  p=root/'quality.md';p.write_text(p.read_text().replace('Speedup vs VGGT: N/A until matched Dense full-forward timing is available; see comparison report.',f'Speedup ratio vs stored paired Dense timing: {ratio:.6f}x. Control not remeasured per user instruction.'))
  for name in ('dense','rep_combined','original'):
   ref=macro[name]
   comparisons[name]=dict(delta_values={k:v[k]-ref['values'][k] for k in v},runtime_ratio_of_stored_medians=ref['mean_scene_median_seconds']/item['mean_scene_median_seconds'],reference_reused=True,contemporaneous_control=False)
  delta=comparisons['rep_combined']['delta_values'];old_speed=comparisons['rep_combined']['runtime_ratio_of_stored_medians']
  lines+=['',f"Stored unoptimized combined median mean / compact median mean = {old_speed:.4f}x. This is a stored-reference ratio, not a fresh paired control measurement.",
   f"Quality differences against unoptimized combined: AUC@3 {100*delta['auc03']:+.5f}pp, AUC@30 {100*delta['auc30']:+.5f}pp, ACC {delta['acc']:+.6f}m, Comp {delta['comp']:+.6f}m, CD {delta['cd']:+.6f}m, F1 {100*delta['f1']:+.5f}pp.",
   'Floating-point reduction/atomic order and larger execution batching can cause discrete state/network drift despite identical state definitions. No claim of bitwise identical full-network output.',
   'All complete-forward costs, geometry preparation, history/routing, sixteen-step solve, packing, transfer and diagnostic summaries count. The existing second evidence QK pass remains included.']
  save(root/'comparisons.json',comparisons)
 lines+=['','Per-scene raw result/input/config/source snapshots are under scene/rep_compact/. Detailed six-metric scene table: report.md. Numerical checks and instrumented profile are separate diagnostic-only artifacts; their overhead is not a warmed timing measurement.']
 (root/'compact_report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output_dir',type=Path);a=p.parse_args();summarize(a.output_dir)
