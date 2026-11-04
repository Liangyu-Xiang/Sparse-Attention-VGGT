"""Full six-metric evaluation and complete-forward accounting for32/256."""
import csv,hashlib,json,shutil,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sampled_history_quality_20261104';REF=ROOT/'outputs/direct_probability_quality_20261104';KEYS=['auc03','auc30','acc','comp','cd','f1']
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def csvwrite(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)
def table(rows,keys):
 def cell(k,v):
  if v is None:return 'N/A'
  return f'{100*v:.4f}%' if k in ['auc03','auc30','f1'] else f'{v:.6f}' if isinstance(v,float) else str(v)
 return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(cell(k,r.get(k)) for k in keys)+' |' for r in rows])
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());rows=[];layers=[];audits=[];missing=[]
 for ds,scenes in manifest['scenes'].items():
  for scene in scenes:
   for method in ['dense','saf3r','v8','direct_legacy','probability4']:
    src=REF/f'quality/{ds}/{scene}/result.json' if method=='probability4' else REF/f'references/{method}/{ds}/{scene}.json'
    dst=OUT/f'references/{method}/{ds}/{scene}.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);d=json.loads(dst.read_text());rows.append(dict(dataset=ds,scene=scene,method=method,frames=d['frames'],**d['quality']['values'],complete=d['quality']['complete'],reused=True,dense_speedup=1. if method=='dense' else None))
   ref=json.loads((OUT/f'references/probability4/{ds}/{scene}.json').read_text())
   for interval in manifest['intervals']:
    p=OUT/f'quality/{interval}/{ds}/{scene}/result.json';mp=OUT/f'monitors/{interval}_{ds}_{scene}.json'
    if not p.exists() or not mp.exists():missing.append(f'{interval}/{ds}/{scene}');continue
    d=json.loads(p.read_text());m=json.loads(mp.read_text());ll=json.loads((p.parent/'layers.json').read_text());normalized=dict(d['config']);normalized.pop('history_query_interval');normalized.update(history_evidence_backend='direct_probability',history_cache_format='plain_log_fp16')
    checks=dict(quality=d['quality']['complete'],images=d['input']['image_paths']==ref['input']['image_paths'],frames=d['input']['frame_indices']==ref['input']['frame_indices'],shape=d['input']['shape']==ref['input']['shape'],checkpoint=Path(d['input']['checkpoint']).resolve()==Path(ref['input']['checkpoint']).resolve(),protocol=d['quality_protocol']==ref['quality_protocol'],other_config=normalized==ref['config'],sampled_interval=d['config']['history_query_interval']==interval,layers=len(ll)==24,cap=all(x['counts_mean']<=x['patch_budget'] for x in ll),no_aux_qk=all(x['extra_known_probe_qk_per_head']==0 and x['evidence_extra_qk_per_head']==0 for x in ll),exclusive=m['returncode']==0 and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']==1)
    audits.append(dict(interval=interval,dataset=ds,scene=scene,checks=checks,passed=all(checks.values())));v=d['quality']['values']
    if d['quality']['complete']:assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-8
    rows.append(dict(dataset=ds,scene=scene,method=f'sampled{interval}',frames=d['frames'],**v,complete=d['quality']['complete'],reused=False,quality_sample_seconds=d['timing']['median_seconds'],quality_peak_gib=max(x['peak_allocated_gib'] for x in d['timing']['samples']),dense_speedup=None))
    layers.extend(dict(interval=interval,dataset=ds,scene=scene,**{k:v for k,v in x.items() if not isinstance(v,dict)}) for x in ll)
 macro=[];changes=[]
 for ds,scenes in manifest['scenes'].items():
  for method in ['dense','saf3r','v8','direct_legacy','probability4','sampled32','sampled256']:
   r=[x for x in rows if x['dataset']==ds and x['method']==method];complete=len(r)==len(scenes) and all(x['complete'] for x in r)
   macro.append(dict(dataset=ds,method=method,scenes=len(r),complete=complete,**{k:statistics.mean(x[k] for x in r) if complete else None for k in KEYS},dense_speedup=1. if method=='dense' else None))
  for method in ['sampled32','sampled256']:
   candidate=next(x for x in macro if x['dataset']==ds and x['method']==method)
   if candidate['complete']:
    for control in ['dense','probability4','sampled32']:
     base=next(x for x in macro if x['dataset']==ds and x['method']==control)
     if not base['complete'] or method==control:continue
     changes.append(dict(dataset=ds,method=method,reference=control,**{k+'_relative_change_percent':100*(candidate[k]/base[k]-1) for k in KEYS},**{k+'_percentage_point_change':100*(candidate[k]-base[k]) for k in ['auc03','auc30','f1']}))
 timings=[];profiles=[];nested=[];timing_objects={};timing_ok=True
 from scripts.quality_metrics import quality_record,quality_markdown
 for interval in manifest['intervals']:
  p=OUT/f'timing/{interval}/result.json';mp=OUT/f'monitors/timing_{interval}.json'
  if not p.exists() or not mp.exists():timing_ok=False;continue
  d=json.loads(p.read_text());m=json.loads(mp.read_text());okay=m['returncode']==0 and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']==1;timing_ok &= okay;timing_objects[str(interval)]=d
  timings.append(dict(interval=interval,median_seconds=d['median_seconds'],peak_allocated_gib=max(x['peak_allocated_gib'] for x in d['samples']),dense_speedup=None,samples=str([x['seconds'] for x in d['samples']])))
  prof=d['profile'];profiles.extend(dict(interval=interval,operation=k,seconds=v,percent_profile=100*v/prof['wall_seconds']) for k,v in prof['exclusive_seconds'].items());profiles.append(dict(interval=interval,operation='outside_named_stages',seconds=prof['unattributed_seconds'],percent_profile=100*prof['unattributed_seconds']/prof['wall_seconds']))
  nested.extend(dict(interval=interval,operation=k,seconds=v,scope='nested in execution_and_evidence;do not add twice') for k,v in prof['direct_parts'].items());(p.parent/'quality.md').write_text(quality_markdown(quality_record({},'Timing-only full-forward benchmark;geometry separately evaluated per scene')))
 hashes=json.loads((OUT/'source_sha256.json').read_text());unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items());complete=len(audits)==36 and not missing and all(x['passed'] for x in audits) and unchanged and timing_ok
 summary=dict(complete=complete,macro=macro,relative_changes=changes,timing=timings,timing_objects=timing_objects,profiles=profiles,nested=nested,audits=audits,missing=missing,source_unchanged=unchanged,timing_ok=timing_ok,controls_executed=0,quality_retuning=False,dense_speedup=None,dense_reason=manifest['dense_reason'],max_prior_mass_error=max((x['prior_patch_mass_max_error'] for x in layers),default=None),max_seed_normalization_error=max((x['seed_normalization_max_error'] for x in layers),default=None))
 save(OUT/'summary.json',summary)
 for name,data in [('per_scene.csv',rows),('macro.csv',macro),('per_layer.csv',layers),('relative_changes.csv',changes),('timing.csv',timings),('profile.csv',profiles),('profile_nested.csv',nested)]:csvwrite(OUT/name,data)
 text=['# One sparse history Probe every32/256 Queries','',f'Campaign complete: **{complete}**;only the two changed methods rerun,no GT tuning or parameter/quality-gate changes.','',
 '## Fixed estimator and unsampled-group rule','',
 'Both methods export inline debiased log sparse scores with final compensated LSE,one Probe at offset4.32 samples every Query Group;256 samples every8Groups. Only sampled Groups receive probability history. Other Groups use their OWN current coarse-uniform prior;their residual summary uses own coarse patch mass and selected-count/GridSize. No neighbor Key IDs or probabilities are copied. Their own previous support remains eligible for the unchanged state selector.','',
 'Sampled Groups:complete missing Keys using pooled-Key logits in the same sparse-LSE scale;store signed FP16 log conditionals and coarse/unobserved statistics;apply M5c beta(1,2) in the next layer. The prior uses coarse Probe0 consistently;four coarse Probes remain for residual candidate exploration. Cache is sampled-only and one-Probe. Retention20%,adaptive-local clusters,Drop/Rep/History/Exact,state loss,16-round budget and mass-transfer compensation stay fixed. The inline log exporter has FP16 auxiliary quantization;this is not an exact old-four-Probe trajectory claim.','',
 '## Complete-forward runtime','',
 'Both intervals use GPU7 RTX5090 D,the identical heads128 image list/checkpoint/preprocessing,one warmup+three full-forward samples,and a separate profiled forward. Includes initialization inside forward,clustering,routing,history,reordering and prediction heads. CPU-only exclusive leases held through full process exit;one CUDA program per physical GPU.','',table(timings,['interval','median_seconds','peak_allocated_gib','dense_speedup','samples']),'',
 'Dense end-to-end speedup N/A:no unchanged Dense rerun. Historical probability4=15.641640s from direct_probability_quality_20261104 is not a paired baseline. Compare32vs256 within this batch;do not label the historical difference a paired speedup.','',
 'Disjoint profile stages,percentages use each separate profile wall rather than the warmed median:','',table(profiles,['interval','operation','seconds','percent_profile']),'',
 'Nested AV and probability-finalizer times are already inside execution_and_evidence:','',table(nested,['interval','operation','seconds','scope']),'',
 'outside_named_stages includes shared original-model work plus uninstrumented host/allocator gaps;it is not all unavoidable Dense cost.','',
 '## Six-metric quality','',
 'All7 7Scenes scenes128frames and all11 ETH3D scenes14–76frames,pose+recon_unposed,equal-scene macros. Same exact frames/checkpoint/shape/protocol audited against historical probability4. AUC/F1 percentages here,fractions in raw JSON. Distances metres,unsquared CD=(ACC+Comp)/2. Threshold/voxel7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. Dense/SAF3R/v8/Direct-legacy/probability4 controls are historical local results. SAF3R uses local lazy first-correspondence-head Top4/ETH3D profile,not paper DINO Top-K.','',table(macro,['dataset','method','scenes','complete',*KEYS]),'',
 '## Per-scene changed methods','',table([x for x in rows if x['method'].startswith('sampled')],['dataset','scene','method','frames','complete',*KEYS]),'',
 f"Source unchanged={unchanged};scene audits={len(audits)};missing={missing};max prior patch-mass error={summary['max_prior_mass_error']};actual stored-log conditional normalization error={summary['max_seed_normalization_error']}.",'',
 'Raw metrics,per-layer rows,source snapshot/hashes and all GPU monitors are preserved. Numerical and two-frame diagnostic runs report all six fields N/A with concrete reasons. The first numerical graph test required bitwise equality of atomic cluster sums;the corrected harness keeps exact prior equality and tolerates FP32 atomic summation rounding(~1e-8),without changing the estimator. Completed quality evaluation does not imply quality equivalence or automatic default promotion.']
 (OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(complete=complete,missing=missing,timing=timings,macro=[x for x in macro if x['method'].startswith('sampled')]),indent=2))
if __name__=='__main__':main()
