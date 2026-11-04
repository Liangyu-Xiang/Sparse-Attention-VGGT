"""Audit changed cluster-count variants against reused sampled32 controls."""
import csv,hashlib,json,shutil,statistics
from pathlib import Path
from scripts.quality_metrics import quality_record,quality_markdown
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cluster_count_quality_20261104';REF=ROOT/'outputs/sampled_history_quality_20261104';KEYS=['auc03','auc30','acc','comp','cd','f1']
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
 manifest=json.loads((OUT/'manifest.json').read_text());rows=[];layers=[];audits=[];missing=[];failed=[]
 for ds,scenes in manifest['scenes'].items():
  for scene in scenes:
   refs={}
   for method in ['dense','saf3r','sampled32_baseline']:
    src=REF/f'quality/32/{ds}/{scene}/result.json' if method=='sampled32_baseline' else REF/f'references/{method}/{ds}/{scene}.json'
    dst=OUT/f'references/{method}/{ds}/{scene}.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);d=json.loads(dst.read_text());refs[method]=d
    rows.append(dict(dataset=ds,scene=scene,method=method,frames=d['frames'],**d['quality']['values'],complete=d['quality']['complete'],reused=True,dense_speedup=1. if method=='dense' else None))
   ref=refs['sampled32_baseline']
   for fraction in manifest['cluster_percentages']:
    p=OUT/f'quality/{fraction}/{ds}/{scene}/result.json';mp=OUT/f'monitors/{fraction}_{ds}_{scene}.json'
    if not p.exists():
     missing.append(f'{fraction}/{ds}/{scene}');failure=p.parent/'failure.json'
     q=json.loads(failure.read_text())['quality'] if failure.exists() else quality_record({},'Run missing or interrupted; see campaign logs')
     rows.append(dict(dataset=ds,scene=scene,method=f'clusters{fraction}',frames=ref['frames'],**q['values'],complete=False,reused=False,dense_speedup=None));continue
    d=json.loads(p.read_text());m=json.loads(mp.read_text());ll=json.loads((p.parent/'layers.json').read_text());normalized=dict(d['config']);normalized['cluster_fraction']=ref['config']['cluster_fraction']
    checks=dict(quality=d['quality']['complete'],images=d['input']['image_paths']==ref['input']['image_paths'],frames=d['input']['frame_indices']==ref['input']['frame_indices'],shape=d['input']['shape']==ref['input']['shape'],checkpoint=Path(d['input']['checkpoint']).resolve()==Path(ref['input']['checkpoint']).resolve(),protocol=d['quality_protocol']==ref['quality_protocol'],other_config=normalized==ref['config'],cluster_fraction=d['config']['cluster_fraction']==fraction/100,history32=d['config']['history_query_interval']==32,layers=len(ll)==24,cap=all(x['counts_mean']<=x['patch_budget'] for x in ll),no_aux_qk=all(x['extra_known_probe_qk_per_head']==0 and x['evidence_extra_qk_per_head']==0 for x in ll),exclusive=m['returncode']==0 and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']==1)
    audits.append(dict(cluster_percent=fraction,dataset=ds,scene=scene,checks=checks,passed=all(checks.values())));v=d['quality']['values']
    if d['quality']['complete']:assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-8
    rows.append(dict(dataset=ds,scene=scene,method=f'clusters{fraction}',frames=d['frames'],**v,complete=d['quality']['complete'],reused=False,quality_sample_seconds=d['timing']['median_seconds'],quality_peak_gib=max(x['peak_allocated_gib'] for x in d['timing']['samples']),dense_speedup=None))
    layers.extend(dict(cluster_percent=fraction,dataset=ds,scene=scene,**{k:v for k,v in x.items() if not isinstance(v,dict)}) for x in ll)
 macro=[];changes=[]
 for ds,scenes in manifest['scenes'].items():
  for method in ['dense','saf3r','sampled32_baseline','clusters15','clusters10','clusters5']:
   r=[x for x in rows if x['dataset']==ds and x['method']==method];complete=len(r)==len(scenes) and all(x['complete'] for x in r)
   macro.append(dict(dataset=ds,method=method,scenes=len(r),complete=complete,**{k:statistics.mean(x[k] for x in r) if complete else None for k in KEYS},dense_speedup=1. if method=='dense' else None))
  for method in ['clusters15','clusters10','clusters5']:
   c=next(x for x in macro if x['dataset']==ds and x['method']==method)
   for control in ['dense','sampled32_baseline']:
    b=next(x for x in macro if x['dataset']==ds and x['method']==control)
    if c['complete'] and b['complete']:changes.append(dict(dataset=ds,method=method,reference=control,**{k+'_relative_change_percent':100*(c[k]/b[k]-1) for k in KEYS},**{k+'_percentage_point_change':100*(c[k]-b[k]) for k in ['auc03','auc30','f1']}))
 timings=[];profiles=[];nested=[];timing_objects={};timing_ok=True
 for fraction in manifest['cluster_percentages']:
  p=OUT/f'timing/{fraction}/result.json';mp=OUT/f'monitors/timing_{fraction}.json'
  if not p.exists() or not mp.exists():timing_ok=False;continue
  d=json.loads(p.read_text());m=json.loads(mp.read_text());timing_ok &= m['returncode']==0 and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']==1;timing_objects[str(fraction)]=d
  timings.append(dict(cluster_percent=fraction,median_seconds=d['median_seconds'],peak_allocated_gib=max(x['peak_allocated_gib'] for x in d['samples']),dense_speedup=None,samples=str([x['seconds'] for x in d['samples']])))
  prof=d['profile'];profiles.extend(dict(cluster_percent=fraction,operation=k,seconds=v,percent_profile=100*v/prof['wall_seconds']) for k,v in prof['exclusive_seconds'].items());profiles.append(dict(cluster_percent=fraction,operation='outside_named_stages',seconds=prof['unattributed_seconds'],percent_profile=100*prof['unattributed_seconds']/prof['wall_seconds']))
  nested.extend(dict(cluster_percent=fraction,operation=k,seconds=v,scope='nested in execution_and_evidence;do not add twice') for k,v in prof['direct_parts'].items());(p.parent/'quality.md').write_text(quality_markdown(quality_record({},'Timing-only benchmark;pose and reconstruction separately evaluated per scene')))
 hashes=json.loads((OUT/'source_sha256.json').read_text());unchanged=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items());complete=len(audits)==54 and not missing and all(x['passed'] for x in audits) and unchanged and timing_ok
 summary=dict(complete=complete,macro=macro,relative_changes=changes,timing=timings,timing_objects=timing_objects,profiles=profiles,nested=nested,audits=audits,missing=missing,source_unchanged=unchanged,timing_ok=timing_ok,controls_executed=0,quality_retuning=False,dense_speedup=None,dense_reason=manifest['dense_reason'])
 save(OUT/'summary.json',summary)
 for name,data in [('per_scene.csv',rows),('macro.csv',macro),('per_layer.csv',layers),('relative_changes.csv',changes),('timing.csv',timings),('profile.csv',profiles),('profile_nested.csv',nested)]:csvwrite(OUT/name,data)
 text=['# Cluster-count sweep:32 Query history,20% Attention budget','',f'Campaign complete: **{complete}**. Only15%/10%/5% cluster variants rerun;54 quality runs. No test-truth tuning or changed quality gates.','',
 '## Method and scope','',
 'Base c85543d. Same selected-Key inline probability history every32Queries,coarse calibration and missing-token completion. Only cluster_fraction is varied;20% Attention retention budget,states,loss,16-round allocation and mass transfer fixed. The old guard F>=budget is removed. First layer uses one real representative for every cluster,then deterministic uniformly spaced non-representative real members to fill the SAME budget. Extra members have weight1,subtracted from their representative weights;cluster mass is conserved and IDs are unique. This initialization adaptation is necessary because old initialization duplicates representatives for F<budget. Legacy F>=budget path produces identical seed IDs/masses. Initial-layer state1 diagnostics count representative clusters;extra members are explicitly reported separately,not claimed to be allocator state choices.','',
 'Larger clusters can increase geometry width and trigger existing width>64 fallback instead of fused state/reused graph kernels. The initial width256 history kernel exceeded shared memory,and one width>256 scene failed its guard. Retry preserves identical launches for width<=128,caps history tiles at8192 elements for wider clusters,and extends exact geometry/drop bins to512;no cluster membership or state math changes. Successful first-attempt runs with width<=128 and clean exclusive monitors are reused;all failed/overlapped runs are retried. Initial failed runs and source snapshot remain in cluster_count_quality_attempt1_20261104. Synthetic widths32–512 preserve exact owners and losses within FP32 reduction tolerance. Actual per-layer densities and geometry widths are in per_layer.csv. Retention is an upper budget,not an exact guarantee for the discrete allocator.','',
 '## Warmed full-forward runtime','',
 'All3 variants timed on physicalGPU2 RTX5090D with identical heads128 frames/checkpoint/preprocessing,one warmup+three full forwards+separate profile. Includes initialization inside forward,clustering,routing,history,reordering and prediction heads. Initial batch used all idle GPUs0–7;after other sessions occupied0/3/4/7,the retry batch used idle1/2/5/6,one CUDA program per physical GPU,CPU-only exclusive leases and overlap monitoring.','',table(timings,['cluster_percent','median_seconds','peak_allocated_gib','dense_speedup','samples']),'',
 'Unchanged sampled32 baseline approximately20% clusters:15.222148s historical warmed median from sampled_history_quality_20261104;not rerun and not a paired speed comparison. Dense speedup=N/A because unchanged Dense not rerun under current user scope. Historical SAF3R timings are also not paired. Dense identity=1.00x.','',
 'Profile percentages use separate profile wall,not warmed median. Outside named stages includes original shared work and unclassified dispatch gaps,not wholly unavoidable Dense cost.','',table(profiles,['cluster_percent','operation','seconds','percent_profile']),'','Nested AV/probability export already included in execution_and_evidence:','',table(nested,['cluster_percent','operation','seconds','scope']),'',
 '## Six-metric quality','',
 'All7 7Scenes scenes128frames and11 ETH3D scenes14–76frames,pose+recon_unposed,equal-scene macro averages. Exact image paths/indices,checkpoint,shape,protocol audited against reused sampled32. AUC/F1 percentages in tables,fractions in JSON;distances metres,unsquared CD=(ACC+Comp)/2. Threshold/downsample7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. Dense/SAF3R/sampled32 controls are historical local measurements. SAF3R local lazy first-correspondence-head Top4/ETH3D profile is not paper DINO feature Top-K or Top128. No published-reference scores are mixed into local tables.','',table(macro,['dataset','method','scenes','complete',*KEYS]),'',
 '## Per-scene metrics','',table(rows,['dataset','scene','method','frames','complete',*KEYS]),'',
 f'Source unchanged={unchanged};audits={len(audits)};missing={missing}. Raw results,layer records,source snapshot/hashes and GPU monitors preserved. Numerical/smoke/timing/failed runs carry all six fields as unavailable with concrete reasons;they are not complete quality evaluations.']
 (OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(complete=complete,missing=missing,timing=timings,macro=macro),indent=2))
if __name__=='__main__':main()
