"""Matched six-metric quality, exhaustive scene audits and disjoint timing."""
import csv,hashlib,json,shutil,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/direct_probability_quality_20261104';OLD=ROOT/'outputs/direct_sparse_quality_20261104';KEYS=['auc03','auc30','acc','comp','cd','f1']
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def csvwrite(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def table(rows,keys):
 def cell(k,v):
  if v is None:return 'N/A'
  return f'{100*v:.4f}%' if k in ['auc03','auc30','f1'] else f'{v:.6f}' if isinstance(v,float) else str(v)
 return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(cell(k,r.get(k)) for k in keys)+' |' for r in rows])
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());rows=[];audits=[];layers=[];missing=[]
 for ds,scenes in manifest['scenes'].items():
  for scene in scenes:
   for method in ['dense','saf3r','v8','direct_legacy']:
    src=OLD/f'quality/{ds}/{scene}/result.json' if method=='direct_legacy' else OLD/f'references/{method}/{ds}/{scene}.json'
    target=OUT/f'references/{method}/{ds}/{scene}.json';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target);d=json.loads(target.read_text())
    rows.append(dict(dataset=ds,scene=scene,method=method,frames=d['frames'],**d['quality']['values'],complete=d['quality']['complete'],reused=True,dense_speedup=1. if method=='dense' else None))
   p=OUT/f'quality/{ds}/{scene}/result.json';mon=OUT/f'monitors/{ds}_{scene}.json'
   if not p.exists() or not mon.exists():missing.append(f'{ds}/{scene}');continue
   d=json.loads(p.read_text());ref=json.loads((OUT/f'references/direct_legacy/{ds}/{scene}.json').read_text());m=json.loads(mon.read_text());l=json.loads((p.parent/'layers.json').read_text())
   normalized=dict(d['config']);normalized['history_evidence_backend']='direct_legacy';normalized['history_cache_format']='packed'
   checks=dict(quality=d['quality']['complete'],images=d['input']['image_paths']==ref['input']['image_paths'],frames=d['input']['frame_indices']==ref['input']['frame_indices'],shape=d['input']['shape']==ref['input']['shape'],checkpoint=Path(d['input']['checkpoint']).resolve()==Path(ref['input']['checkpoint']).resolve(),protocol=d['quality_protocol']==ref['quality_protocol'],other_config=normalized==ref['config'],changed_method=d['config']['history_evidence_backend']=='direct_probability',layers=len(l)==24,cap=all(x['counts_mean']<=x['patch_budget'] for x in l),legacy_qk_removed=all(not x['legacy_observation_repeats_qk'] and x['direct_logical_legacy_observation_qk_per_head']==0 and x['direct_logical_conditional_cache_qk_per_head']==0 for x in l),exclusive=m['returncode']==0 and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']==1)
   audits.append(dict(dataset=ds,scene=scene,checks=checks,passed=all(checks.values())));v=d['quality']['values']
   if d['quality']['complete']:assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-8
   rows.append(dict(dataset=ds,scene=scene,method='probability_history',frames=d['frames'],**v,complete=d['quality']['complete'],reused=False,quality_sample_seconds=d['timing']['median_seconds'],quality_sample_peak_gib=max(x['peak_allocated_gib'] for x in d['timing']['samples']),dense_speedup=None))
   layers.extend(dict(dataset=ds,scene=scene,**{k:v for k,v in x.items() if not isinstance(v,dict)}) for x in l)
 macro=[];changes=[]
 for ds,scenes in manifest['scenes'].items():
  for method in ['dense','saf3r','v8','direct_legacy','probability_history']:
   r=[x for x in rows if x['dataset']==ds and x['method']==method];complete=len(r)==len(scenes) and all(x['complete'] for x in r)
   macro.append(dict(dataset=ds,method=method,scenes=len(r),complete=complete,**{k:statistics.mean(x[k] for x in r) if complete else None for k in KEYS},dense_speedup=1. if method=='dense' else None))
  candidate=next(x for x in macro if x['dataset']==ds and x['method']=='probability_history')
  if candidate['complete']:
   for refmethod in ['dense','v8','direct_legacy']:
    base=next(x for x in macro if x['dataset']==ds and x['method']==refmethod)
    changes.append(dict(dataset=ds,reference='historical matched '+refmethod,**{k+'_relative_change_percent':100*(candidate[k]/base[k]-1) for k in KEYS},**{k+'_percentage_point_change':100*(candidate[k]-base[k]) for k in ['auc03','auc30','f1']}))
 hashes=json.loads((OUT/'source_sha256.json').read_text());unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 timing=json.loads((OUT/'timing/result.json').read_text()) if (OUT/'timing/result.json').exists() else None
 mon=json.loads((OUT/'monitors/timing.json').read_text()) if (OUT/'monitors/timing.json').exists() else {}
 timing_ok=mon.get('returncode')==0 and not mon.get('foreign_gpu_pids') and not mon.get('same_gpu_overlap_detected') and mon.get('max_simultaneous_gpu_applications')==1
 complete=not missing and len(audits)==18 and all(x['passed'] for x in audits) and unchanged and timing_ok
 profile=[]
 if timing and 'profile' in timing:
  prof=timing['profile'];profile=[dict(operation=k,seconds=v,percent_profile=100*v/prof['wall_seconds']) for k,v in prof['exclusive_seconds'].items()]
  profile.append(dict(operation='outside_named_stages',seconds=prof['unattributed_seconds'],percent_profile=100*prof['unattributed_seconds']/prof['wall_seconds']))
 nested=[] if not timing else [dict(operation=k,seconds=v,scope='nested within execution_and_evidence; do not add again') for k,v in timing.get('profile',{}).get('direct_parts',{}).items()]
 diagnostics=dict(max_prior_mass_error=max((x.get('prior_patch_mass_max_error',0) for x in layers),default=None),max_seed_error=max((x.get('seed_normalization_max_error',0) for x in layers),default=None))
 if timing:
  from scripts.quality_metrics import quality_record,quality_markdown
  (OUT/'timing/quality.md').write_text(quality_markdown(quality_record({},'Performance-only complete-forward benchmark; pose/reconstruction evaluated separately per scene')))
 summary=dict(complete=complete,macro=macro,relative_changes=changes,audits=audits,missing=missing,source_unchanged=unchanged,timing=timing,timing_monitor_ok=timing_ok,controls_executed=0,quality_retuning=False,dense_speedup=None,dense_reason=manifest['dense_reason'],default_changed=False,diagnostics=diagnostics)
 save(OUT/'summary.json',summary)
 for name,data in [('per_scene.csv',rows),('macro.csv',macro),('per_layer.csv',layers),('relative_changes.csv',changes),('profile.csv',profile),('profile_nested.csv',nested)]:csvwrite(OUT/name,data)
 text=['# Direct Sparse probability history','',f'Campaign complete: **{complete}**. Opt-in only; default unchanged. No test-truth tuning or unchanged-control reruns.','',
 '## Method','',
 'Removed legacy all-query selected-QK/Grid observations and the old four-probe conditional-cache QK. Main AV is unchanged Direct compensated Sparse. One independent four-probe export uses final sparse LSE; it computes log(mean_h(exp(raw_h - LSE_comp,h))), algebraically the shared-bias-removed actual sparse probability. A zero assignment weight is handled by the raw/LSE limiting score, rather than dividing two floating-point zeros. No per-head edge bank or FP32 raw-logit bank. Four-probe QK still exists; this is not a zero-additional-QK claim.','',
 'Missing-token score is logmeanexp_h(pooled_grid_logit_h - LSE_comp,h), requiring no new Grid QK beyond the existing coarse observer. Selected and coarse missing scores jointly form positive per-Grid conditional distributions via compact log-sum-exp reductions. Signed FP16 log conditionals are cached to avoid FP16 probability underflow. Next layer applies the unchanged M5c beta(1,2) confidence, blending conditionals with uniform, and uses current coarse Grid mass to complete unselected tokens. No hard cap that assigns the entire Grid mass to selected tokens. This intentionally changes the estimator: ratio of head-mean scores, instead of the mean of per-head Grid conditionals; residual exploration uses four-probe means instead of all32 Queries. It is not mathematically equivalent to v8 nor exact Dense probability recovery.','',
 'Retention20%,adaptive-local clusters,four probes,Drop/Rep/History/Exact,16-round budget,state losses,compensation and all geometry protocols remain frozen.','',
 '## Six-metric quality','',
 'All7 7Scenes scenes128frames; all11 ETH3D scenes14–76frames; equal-scene macros. AUC/F1 are percentages here and fractions in raw JSON. ACC/Comp/CD in metres; unsquared CD=(ACC+Comp)/2. Threshold/voxel:7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. Same checkpoint,exact image list,preprocessing,pose and reconstruction protocols audited against preceding Direct run. Dense/SAF3R/v8/Direct-legacy are historical local controls. SAF3R is lazy first-correspondence-head Top4 with ETH3D profile,not paper DINO Top-K.','',table(macro,['dataset','method','scenes','complete',*KEYS]),'',
 '## Complete-forward timing','',
 'heads128,GPU7 RTX5090 D,onewarmup+three warmed full forwards plus SEPARATE profiled forward. Includes initialization inside forward,clustering,routing,history,reordering and prediction heads. CPU-only leases,one CUDA application per physical GPU,held through full process exit.','',
 (f"Warmed median **{timing['median_seconds']:.9f}s**,samples {[x['seconds'] for x in timing['samples']]},max allocated **{max(x['peak_allocated_gib'] for x in timing['samples']):.6f}GiB**." if timing else 'Timing not complete.'),'',
 'End-to-end Dense speedup **N/A**: no unchanged Dense rerun in this campaign; historical Dense timings cannot be used as paired speed comparisons. Dense identity1.00x. Preceding Direct18.999162s is historical from direct_sparse_quality_20261104,not a contemporaneous speed comparison.','',
 'Profile percentages use the separate profile wall. execution_and_evidence includes both AV and four-probe export; nested Direct scope timings separate those operations and must not be added twice. outside_named_stages includes shared base-model work PLUS uninstrumented host/allocator gaps; it is not all unavoidable Dense cost.','',table(profile,['operation','seconds','percent_profile']),'',table(nested,['operation','seconds','scope']),'',
 f"Audits={len(audits)},source unchanged={unchanged},missing={missing},diagnostics={diagnostics}.",'',
 '## Per-scene results','',table([r for r in rows if r['method']=='probability_history'],['dataset','scene','frames','complete',*KEYS]),'',
 'Raw metrics,per-layer records,source snapshot/hashes,monitor logs and diagnostic failures are retained. Numerical tests cover extreme bias,zero assignments,FP16 underflow,arbitrary support order/holes,all/none/partial Grids,tail probes and fresh CUDA Graph replay; six geometric fields N/A for these diagnostics. Numerical correctness does not establish geometric quality.']
 (OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(complete=complete,missing=missing,macro=[x for x in macro if x['method']=='probability_history'],median_seconds=None if timing is None else timing['median_seconds']),indent=2))
if __name__=='__main__':main()
