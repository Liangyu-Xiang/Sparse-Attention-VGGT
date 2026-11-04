"""Archive partial/rejected candidates and report frozen final P0 six-metric quality."""
import csv,hashlib,json,shutil,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/v8_p0_quality_v3_20261104';REF=Path('/tmp/sparse-vggt-beta-gridmass-20261104/outputs/beta_gridmass_20261104/references');KEYS=['auc03','auc30','acc','comp','cd','f1']
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def table(rows,keys):
 def cell(k,v):
  if v is None:return 'N/A'
  return f'{v*100:.4f}%' if k in ['auc03','auc30','f1'] else f'{v:.6f}' if isinstance(v,float) else str(v)
 return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(cell(k,r.get(k)) for k in keys)+' |' for r in rows])
def csvwrite(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());rows=[];audits=[];layers=[];references=[];missing=[]
 for ds,scenes in manifest['scenes'].items():
  for scene in scenes:
   for method in ['dense','saf3r','v8']:
    path=OUT/f'references/{method}/{ds}/{scene}.json'
    if not path.exists():path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REF/f'{method}/{ds}/{scene}.json',path)
    d=json.loads(path.read_text());rows.append(dict(dataset=ds,scene=scene,method=method,frames=d['frames'],**d['quality']['values'],complete=d['quality']['complete'],reused=True,dense_speedup=1. if method=='dense' else None))
    references.append(dict(path=str(path.relative_to(OUT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
   p=OUT/f'quality/{ds}/{scene}/result.json';mon=OUT/f'monitors/{ds}_{scene}.json'
   if not p.exists() or not mon.exists():missing.append(f'{ds}/{scene}');continue
   d=json.loads(p.read_text());ref=json.loads((OUT/f'references/v8/{ds}/{scene}.json').read_text());m=json.loads(mon.read_text());l=json.loads((p.parent/'layers.json').read_text());baseconfig={k:v for k,v in d['config'].items() if k!='p0_loss_pack'}
   checks=dict(quality=d['quality']['complete'],images=d['input']['image_paths']==ref['input']['image_paths'],frames=d['input']['frame_indices']==ref['input']['frame_indices'],shape=d['input']['shape']==ref['input']['shape'],checkpoint=Path(d['input']['checkpoint']).resolve()==Path(ref['input']['checkpoint']).resolve(),protocol=d['quality_protocol']==ref['quality_protocol'],config=baseconfig==ref['config'],layer_count=len(l)==24,cap=all(x['counts_mean']<=x['patch_budget'] for x in l),exclusive=not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['max_simultaneous_gpu_applications']==1 and m['returncode']==0)
   audits.append(dict(dataset=ds,scene=scene,checks=checks,passed=all(checks.values())))
   v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-8
   rows.append(dict(dataset=ds,scene=scene,method='p0_v2',frames=d['frames'],**v,complete=d['quality']['complete'],reused=False,quality_sample_seconds=d['timing']['median_seconds'],quality_sample_peak_gib=max(x['peak_allocated_gib'] for x in d['timing']['samples']),dense_speedup=None))
   layers.extend(dict(dataset=ds,scene=scene,**{k:v for k,v in x.items() if not isinstance(v,dict)}) for x in l)
 macro=[];changes=[]
 for ds,scenes in manifest['scenes'].items():
  for method in ['dense','saf3r','v8','p0_v2']:
   r=[x for x in rows if x['dataset']==ds and x['method']==method];complete=len(r)==len(scenes) and all(x['complete'] for x in r)
   macro.append(dict(dataset=ds,method=method,scenes=len(r),complete=complete,**{k:statistics.mean(x[k] for x in r) if complete else None for k in KEYS},dense_speedup=1. if method=='dense' else None))
  candidate=next(x for x in macro if x['dataset']==ds and x['method']=='p0_v2');base=next(x for x in macro if x['dataset']==ds and x['method']=='v8')
  if candidate['complete']:changes.append(dict(dataset=ds,**{k+'_relative_change_percent':100*(candidate[k]/base[k]-1) for k in KEYS}))
 hashes=json.loads((OUT/'source_sha256.json').read_text());unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
 complete=not missing and len(audits)==18 and all(x['passed'] for x in audits) and unchanged
 bench=ROOT/'outputs/dense_p0_v2_comparison_20261104';timing={}
 if (bench/'completion.json').exists():
  ds={m:json.loads((bench/m/'result.json').read_text()) for m in ['dense','p0']};timing=dict(dense_seconds=ds['dense']['median_seconds'],p0_v2_seconds=ds['p0']['median_seconds'],dense_speedup=ds['dense']['median_seconds']/ds['p0']['median_seconds'],scope='Separate paired heads128 benchmark;not per-scene quality timings',p0_samples=ds['p0']['samples'])
 summary=dict(complete=complete,macro=macro,relative_changes=changes,audits=audits,missing=missing,source_unchanged=unchanged,paired_timing=timing,quality_retuning=False,controls_quality_rerun=False,diagnostics=dict(global_attention_density={ds:statistics.mean(x['actual_unique_density'] for x in layers if x['dataset']==ds) for ds in manifest['scenes'] if any(x['dataset']==ds for x in layers)}),acceptance='Rejected: corrected P0 also fails ALL-window routing equivalence at L1 and does not improve full-forward speed; default unchanged')
 save(OUT/'summary.json',summary);save(OUT/'references/index.json',references)
 csvwrite(OUT/'per_scene.csv',rows);csvwrite(OUT/'macro.csv',macro);csvwrite(OUT/'per_layer.csv',layers)
 text=['# Corrected v8 P0 execution prototype','',f'Evaluation complete: **{complete}**. Default unchanged. No test-truth tuning. Only corrected P0 geometry is rerun; Dense/SAF3R/v8 quality is reused. Dense timing rerun is explicitly authorized by the later user request to compare original forward/core.','',
 'All7 7Scenes scenes128frames and all11 ETH3D scenes14–76frames; exact frame list, shape, checkpoint, preprocessing and pose+recon_unposed protocol audited. Equal-scene macro, AUC/F1 percentages below, JSON fractions; unsquared distances metres, CD=(ACC+Comp)/2. Threshold/voxel7Scenes0.05/0.0078125m,ETH3D0.25/0.0390625m. SAF3R local lazy first-correspondence-head Top4/ETH3D profile,not paper DINO Top-K.','',table(macro,['dataset','method','scenes','complete',*KEYS]),'',
 'Independent matched heads128 timing: '+json.dumps(timing)+'. Each quality scene has one warmup and one measured output, recorded separately and not presented as warmed3-sample timing. Dataset timing speedups N/A; complete-forward paired speedup is only defined for heads128.','',
 'The first prototype changed cluster tile size while sharing geometry across2 Query Groups; small loss rounding differences produced a routing mismatch at L4. Its quality queue was stopped without discarding completed scene scores; it is not an accepted equivalent implementation. The corrected kernel preserves original cluster tile/reduction layout while reusing query-independent geometry. Captured-vs-original-eager routing matches Keys/padding/state counts in the first64-group window of all23 changed layers. Maximum weight relativeL2 5.802890612471856e-8; maximum absolute6.103515625e-5 on large weights, within the unchanged rtol1e-5 gate. L0 seed unchanged. Original v8 Graph-vs-eager passed the same check.','',
 'The stronger ALL-window routing oracle FAILED at L1 beyond the sampled window. The corrected prototype is NOT accepted as globally equivalent; small floating-point loss differences can change discrete budget decisions. Failure and first-mismatch investigation are preserved. The complete quality evaluation measures a rejected candidate, not an equivalence success.', '',
 'Direct packing uses per-cluster counts/prefix and direct member writes, then sorts ONLY the retained Key list to preserve original ascending token order. It eliminates full-Patch flags/prefix, but sorting can offset the saving. This initial execution prototype does not implement Cluster–Grid sufficient statistics, does not change beta or probability definition, and does not claim a successful speed optimization.','',
 f'Frozen-source unchanged: {unchanged}. Completed audits: {len(audits)}. Missing scenes: {missing}. Full source archive, hashes, failed diagnostics, numerical checks, six-metric per-scene records and GPU exclusivity monitors are preserved.','',
 'Six-metric relative changes versus historical matched v8 (percent): '+json.dumps(changes)+'. Differences in floating-point execution can propagate through later discrete decisions; no quality gate was relaxed and no result was selected by repeated successful quality runs.']
 (OUT/'report.md').write_text('\n'.join(text)+'\n');print(json.dumps(dict(complete=complete,macro=[x for x in macro if x['method']=='p0_v2'],timing=timing,missing=missing),indent=2))
if __name__=='__main__':main()
