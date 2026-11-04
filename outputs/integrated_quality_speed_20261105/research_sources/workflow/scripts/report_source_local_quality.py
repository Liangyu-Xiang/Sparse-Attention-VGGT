"""Matched full-forward and six-metric tables for all predeclared variants."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/source_local_quality_20261105'
plan=json.loads((OUT/'plan.json').read_text());scenes=plan['scenes'];variants=plan['variants']
FIELDS=('auc03','auc30','acc','comp','cd','f1')
rows=[]; macro=[]; missing=[];checks=[]
for scene in scenes:
 refpath=OUT/scene/'adaptive_graph/result.json'
 reference=json.loads(refpath.read_text()) if refpath.exists() else None
 for variant,_,execution in variants:
  p=OUT/scene/variant/'result.json'
  if not p.exists() or not (OUT/f'{scene}_{variant}.monitor.json').exists():missing.append(f'{scene}/{variant}');continue
  d=json.loads(p.read_text());monitor=json.loads((OUT/f'{scene}_{variant}.monitor.json').read_text())
  if monitor['returncode'] or monitor['same_gpu_overlap_detected']:raise ValueError('Invalid GPU monitor '+str(p))
  if not d['quality']['complete']:raise ValueError('Incomplete quality '+str(p))
  if reference:
   for key in ('input','quality_protocol','hardware','model_config'):
    assert d[key]==reference[key],(scene,variant,key)
   allowed_changes={'clustering_mode','source_width','source_layout','source_temporal_span'}
   assert all(d['config'][k]==reference['config'][k] for k in reference['config'] if k not in allowed_changes),(scene,variant,'Unplanned config change')
   assert d['timing']['warmup']==1 and d['timing']['repeats']==3 and not d['timing']['capture']
   refmonitor=json.loads((OUT/f'{scene}_adaptive_graph.monitor.json').read_text())
   assert monitor['gpu_uuid']==refmonitor['gpu_uuid'],(scene,variant,'physical GPU mismatch')
  q=d['quality']['values'];assert abs(q['cd']-(q['acc']+q['comp'])/2)<1e-10
  layer_records=json.loads((p.parent/'layers.json').read_text())
  assert len(layer_records)==24 and all(r['query_group_size']==256 and r['fine_clusters']==26009 for r in layer_records)
  t=d['timing'];row=dict(scene=scene,variant=variant,seconds=t['median_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in t['samples']),
    relative_to_adaptive_graph=reference['timing']['median_seconds']/t['median_seconds'] if reference else None,
    dense_speedup=None,dense_speedup_reason='Unchanged Dense control not rerun; historical timings are not paired',
    graph_creations=[s['graph_counters']['graph_cache_creations'] for s in t['samples']],graph_evictions=[s['graph_counters']['graph_cache_evictions'] for s in t['samples']],
    mean_actual_key_density=statistics.mean(r['actual_unique_density'] for r in layer_records),geometry_widths=sorted(set(r['geometry_width'] for r in layer_records)),logical_qk_per_head=sum(r['direct_logical_main_qk_per_head'] for r in layer_records),
    input_sha256=hashlib.sha256(json.dumps(d['input'],sort_keys=True).encode()).hexdigest(),gpu_uuid=monitor['gpu_uuid'],quality_complete=True,**q)
  rows.append(row)
for variant,_,_ in variants:
 subset=[r for r in rows if r['variant']==variant]
 if len(subset)!=3:continue
 m=dict(variant=variant,scenes=3,seconds=statistics.mean(r['seconds'] for r in subset),peak_allocated_gib=max(r['peak_allocated_gib'] for r in subset),**{k:statistics.mean(r[k] for r in subset) for k in FIELDS})
 baseline=[r for r in rows if r['variant']=='adaptive_graph']
 m['relative_to_adaptive_graph']=statistics.mean(r['seconds'] for r in baseline)/m['seconds'] if len(baseline)==3 else None
 m['dense_speedup']=None;m['mean_actual_key_density']=statistics.mean(r['mean_actual_key_density'] for r in subset);macro.append(m)
header=['Method','Full-forward s','Relative to adaptive Graph','Peak GiB','AUC@3 %','AUC@30 %','ACC m','Comp m','CD m','F1 %','Dense speedup']
def table(items):
 lines=['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']
 for r in items:
  values=[r['variant'],f"{r['seconds']:.6f}",f"{r['relative_to_adaptive_graph']:.3f}x" if r.get('relative_to_adaptive_graph') else 'N/A',f"{r['peak_allocated_gib']:.3f}"]
  values += [f"{r[k]*(100 if k in ('auc03','auc30','f1') else 1):.4f}" for k in FIELDS]
  lines.append('| '+' | '.join(values+['N/A'])+' |')
 return lines
lines=['# Fixed-source unanchored clustering: 3 scenes × 128 frames','',f'Completed valid quality runs: {len(rows)}/{len(scenes)*len(variants)}. Missing: {len(missing)}.','',
 'Each row uses one warmup and three full forwards; runtime is the warmed median and includes reset, layout initialization, clustering, routing, history, attention and output heads. Within each scene every row uses the same physical RTX4090, exact frame list, checkpoint, preprocessing and evaluation protocol. No foreign GPU context was detected in included runs.',
 '', 'Equal-scene means below cover heads/chess/stairs only, not all seven 7Scenes scenes. Peak memory is the maximum across included warmed runs. Relative speed is against this batch’s adaptive Graph method, not Dense. Dense speedup is N/A: unchanged Dense was not rerun and historical timing is not paired.','',
 'Quality uses the final timed output of each run, not an average across three independently evaluated outputs. Floating-point atomic reductions and discrete selection are nondeterministic, so small or single-run quality differences are not evidence of a stable improvement. The Graph/eager audit separately measures this limitation.', '', '## Three-scene equal-weight means','']+table(macro)
for scene in scenes:lines+=['',f'## {scene}','']+table([r for r in rows if r['scene']==scene])
if rows:
 d=json.loads((OUT/rows[0]['scene']/rows[0]['variant']/'result.json').read_text())
 lines+=['','## Protocol','',json.dumps(d['quality_protocol'],ensure_ascii=False),
  '', 'SAF3R is not run in this campaign. Historical local SAF3R reference is lazy first-correspondence-head Top-4, not paper DINO feature Top-K. CD=(ACC+Comp)/2, unsquared distances in metres. AUC/F1 are displayed as percentages, stored as fractions. Pose and recon_unposed are both evaluated. No hyperparameter was chosen using test truth.']
# Historical quality-only reference: verify resolved checkpoint + input list/protocol.
historical=Path('/data_NVME1/mmc_lyxiang/3D/SAF3R-query-group-scaling/outputs/dense_saf3r_query128_20261104')
reference_rows=[]
for method in ('dense','saf3r'):
 subset=[]
 for scene in scenes:
  p=historical/scene/method/'result.json'
  if not p.exists():break
  old=json.loads(p.read_text());newp=OUT/scene/'adaptive_graph/result.json'
  if not newp.exists():break
  new=json.loads(newp.read_text())
  for key in ('image_paths','frame_indices','shape'):assert old['input'][key]==new['input'][key],(scene,method,key)
  assert Path(old['input']['checkpoint']).resolve()==Path(new['input']['checkpoint']).resolve()
  assert old['quality_protocol']==new['quality_protocol']
  subset.append(old['quality']['values'])
 if len(subset)==3:reference_rows.append(dict(method=method,**{k:statistics.mean(r[k] for r in subset) for k in FIELDS}))
if reference_rows:
 lines+=['','## Historical local quality reference (20261104 batch; no timing comparison)','', '| Method | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |','|---|---|---|---|---|---|---|']
 for r in reference_rows:lines.append('| '+' | '.join([r['method']]+[f"{r[k]*(100 if k in ('auc03','auc30','f1') else 1):.4f}" for k in FIELDS])+' |')
 dense=next((r for r in reference_rows if r['method']=='dense'),None)
 if dense:
  degradation=[]
  for r in macro:degradation.append(dict(variant=r['variant'],**{k:r[k]-dense[k] for k in FIELDS}))
  lines+=['','Differences below are against matched historical local Dense quality, not published Dense reproduction drift. AUC/F1 differences are percentage points; distances are metres.','', '| Method | ΔAUC@3 pp | ΔAUC@30 pp | ΔACC m | ΔComp m | ΔCD m | ΔF1 pp |','|---|---|---|---|---|---|---|']
  for r in degradation:lines.append('| '+' | '.join([r['variant']]+[f"{r[k]*(100 if k in ('auc03','auc30','f1') else 1):+.4f}" for k in FIELDS])+' |')
else:degradation=[]
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
with (OUT/'three_scene_means.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(macro[0]) if macro else ['variant']);writer.writeheader();writer.writerows(macro)
(OUT/'summary.json').write_text(json.dumps(dict(complete=len(rows)==len(scenes)*len(variants),valid_runs=len(rows),missing=missing,per_scene=rows,three_scene_means=macro,historical_quality=reference_rows),indent=2)+'\n')
with (OUT/'per_scene.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['scene','variant']);writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(valid_runs=len(rows),planned=len(scenes)*len(variants),complete=len(rows)==len(scenes)*len(variants))))
