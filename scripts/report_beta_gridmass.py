"""Frozen changed-method results; unchanged controls are historical only."""
import csv,hashlib,json,statistics,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/beta_gridmass_20261104'
HISTORY=Path('/mnt/rfs/users/xiangliangyu/projects/Sparse-Attention-VGGT/outputs')
METRICS=['auc03','auc30','acc','comp','cd','f1']

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)),lineterminator='\n');writer.writeheader();writer.writerows(rows)
def table(rows,keys):
    def cell(key,val):
        if val is None:return 'N/A'
        if isinstance(val,float):return f'{val*100:.4f}%' if key in ['auc03','auc30','f1'] else f'{val:.6f}'
        return str(val)
    return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(cell(k,row.get(k)) for k in keys)+' |' for row in rows])

def main():
    manifest=json.loads((OUT/'manifest.json').read_text());effective_config=dict(attention_block_keys=64,**manifest['config']);refs=OUT/'references';refs.mkdir(exist_ok=True)
    index=refs/'index.json'
    if not index.exists():
        entries=[]
        for old in csv.DictReader((HISTORY/'grid_local_final_20261102/per_scene.csv').open()):
            if old['method'] in ['dense','saf3r','grid_local']:
                entries.append(dict(method=old['method'],dataset=old['dataset'],scene=old['scene'],original=old['source']))
        for ds,scenes in manifest['scenes'].items():
            for scene in scenes:entries.append(dict(method='v8',dataset=ds,scene=scene,original=str(HISTORY/f'efficiency_quality_v8_20261103/{ds}/{scene}/efficiency/result.json')))
        for row in entries:
            p=refs/row['method']/row['dataset']/(row['scene']+'.json');p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['original'],p)
            row['path']=str(p.relative_to(OUT));row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        save(index,entries)
    entries=json.loads(index.read_text());rows=[];audit=[];layers=[];missing=[];reference={}
    for entry in entries:
        p=OUT/entry['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
        raw=json.loads(p.read_text());reference[entry['method'],entry['dataset'],entry['scene']]=raw
        rows.append(dict(method=entry['method'],dataset=entry['dataset'],scene=entry['scene'],frames=raw['frames'],**raw['quality']['values'],complete=raw['quality']['complete'],reused=True,seconds=raw['timing']['median_seconds'],timing_batch=entry['original'],peak_gib=max(x['peak_allocated_gib'] for x in raw['timing']['samples']),dense_speedup=1. if entry['method']=='dense' else None))
    for ds,scenes in manifest['scenes'].items():
        for scene in scenes:
            folder=OUT/'quality'/ds/scene
            if not (folder/'result.json').exists() or not (OUT/f'monitors/{ds}_{scene}.json').exists():missing.append(f'{ds}/{scene}');continue
            raw=json.loads((folder/'result.json').read_text());ref=reference['dense',ds,scene];layer=json.loads((folder/'layers.json').read_text());mon=json.loads((OUT/f'monitors/{ds}_{scene}.json').read_text())
            checks=dict(quality=raw['quality']['complete'],images=raw['input']['image_paths']==ref['input']['image_paths'],frames=raw['input']['frame_indices']==ref['input']['frame_indices'],shape=raw['input']['shape']==ref['input']['shape'],checkpoint=Path(raw['input']['checkpoint']).resolve()==Path(ref['input']['checkpoint']).resolve(),protocol=raw['quality_protocol']==ref['quality_protocol'],config=raw['config']==effective_config,layers=len(layer)==24,cap=all(x['counts_mean']<=x['patch_budget'] for x in layer),monitor=not mon['auxiliary_cuda_context'] and not mon['foreign_gpu_pids'] and not mon['same_gpu_overlap_detected'] and mon['max_simultaneous_gpu_applications']==1 and mon['returncode']==0)
            audit.append(dict(dataset=ds,scene=scene,checks=checks,passed=all(checks.values())))
            values=raw['quality']['values'];assert abs(values['cd']-(values['acc']+values['comp'])/2)<1e-8
            rows.append(dict(method='beta_gridmass',dataset=ds,scene=scene,frames=raw['frames'],**values,complete=raw['quality']['complete'],reused=False,seconds=raw['timing']['median_seconds'],timing_batch='beta_gridmass_20261104_quality_one_sample',peak_gib=max(x['peak_allocated_gib'] for x in raw['timing']['samples']),dense_speedup=None))
            for r in layer:layers.append(dict(dataset=ds,scene=scene,**{k:v for k,v in r.items() if not isinstance(v,dict)}))
    macros=[];changes=[]
    for ds,scenes in manifest['scenes'].items():
        for method in ['dense','saf3r','grid_local','v8','beta_gridmass']:
            group=[r for r in rows if r['dataset']==ds and r['method']==method];complete=len(group)==len(scenes) and all(r['complete'] for r in group)
            macros.append(dict(dataset=ds,method=method,scenes=len(group),complete=complete,**{k:statistics.mean(r[k] for r in group) if complete else None for k in METRICS},mean_quality_sample_seconds=statistics.mean(r['seconds'] for r in group) if complete else None,peak_gib=max((r['peak_gib'] for r in group),default=None),dense_speedup=1. if method=='dense' else None))
    for ds in manifest['scenes']:
        candidate=next(x for x in macros if x['dataset']==ds and x['method']=='beta_gridmass')
        if not candidate['complete']:continue
        for method in ['dense','v8','grid_local']:
            ref=next(x for x in macros if x['dataset']==ds and x['method']==method)
            changes.append(dict(dataset=ds,reference=method,**{k+'_relative_percent':100*(candidate[k]/ref[k]-1) for k in METRICS},**{k+'_pp':100*(candidate[k]-ref[k]) for k in ['auc03','auc30','f1']}))
    timing=json.loads((OUT/'timing/result.json').read_text()) if (OUT/'timing/result.json').exists() else None
    detailed=json.loads((OUT/'detailed_profile/profile.json').read_text()) if (OUT/'detailed_profile/profile.json').exists() else None
    monitors=[]
    for label in ['timing','detailed_profile']:
        p=OUT/f'monitors/{label}.json'
        if p.exists():
            m=json.loads(p.read_text());monitors.append(dict(label=label,passed=not m['auxiliary_cuda_context'] and not m['foreign_gpu_pids'] and not m['same_gpu_overlap_detected'] and m['returncode']==0 and m['max_simultaneous_gpu_applications']==1))
        else:missing.append(label+'_monitor')
    hashes=json.loads((OUT/'source_sha256.json').read_text());unchanged=all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    complete=not missing and len(audit)==18 and all(x['passed'] for x in audit) and all(x['passed'] for x in monitors) and unchanged and timing is not None and detailed is not None
    diagnostics=dict(max_prior_patch_mass_error=max((r['prior_patch_mass_max_error'] for r in layers),default=None),
        max_quantized_sparse_cache_mass_excess=max((r['seed_normalization_max_error'] for r in layers),default=None),
        global_attention_density={ds:statistics.mean(r['actual_unique_density'] for r in layers if r['dataset']==ds) for ds in manifest['scenes'] if any(r['dataset']==ds for r in layers)})
    summary=dict(complete=complete,effective_config=effective_config,quality_acceptance='not_accepted_severe_regression',macro=macros,relative_changes=changes,timing=timing,detailed_profile=detailed,audit=audit,monitors=monitors,source_unchanged=unchanged,missing=missing,diagnostics=diagnostics,controls_executed=0,dense_speedup=None,dense_reason=manifest['dense_reason'])
    save(OUT/'summary.json',summary)
    for name,data in [('macro.csv',macros),('per_scene.csv',rows),('relative_changes.csv',changes),('per_layer.csv',layers)]:csvwrite(OUT/name,data)
    profile=[]
    if detailed:
        profile=[dict(operation=k,seconds=v['exclusive_seconds'],percent_full=100*v['exclusive_seconds']/detailed['wall_seconds']) for k,v in detailed['groups'].items()]
        profile.append(dict(operation='outside_global',seconds=detailed['wall_seconds']-detailed['global_attention_seconds'],percent_full=100*(detailed['wall_seconds']-detailed['global_attention_seconds'])/detailed['wall_seconds']))
        assert abs(sum(x['seconds'] for x in profile)-detailed['wall_seconds'])<1e-7
        csvwrite(OUT/'profile.csv',profile)
    text=['# Beta + current Grid mass: changed-method experiment','',
      f'Evaluation complete: **{complete}**. **Not accepted: severe geometry quality regression.** This models actual compensated sparse Softmax P as a cross-layer prior; it is not an equivalent v8 optimization. Config and source were frozen before quality results; no parameter/gate/budget was tuned against test truth.','',
      'Only beta_gridmass ran. Dense/SAF3R/grid_local/v8 rows are reused complete historical controls; raw results and original batch paths are preserved in references/.7Scenes all7 scenes128frames; ETH3D all11 scenes14–76frames. Same exact inputs/checkpoint/preprocessing and pose/recon_unposed protocol audited. AUC/F1 percentages below, distances unsquared metres, CD=(ACC+Comp)/2; JSON/CSV use fractions. Threshold/voxel:7Scenes0.05m/0.0078125m;ETH3D0.25m/0.0390625m. SAF3R is lazy first-correspondence-head Top4/ETH3D profile, not paper DINO Top-K.','',table(macros,['dataset','method','scenes','complete',*METRICS]),'',
      '## Runtime','',
      'Fixed heads128 on RTX5090 D GPU7, one warmup+three complete-forward samples, then separate profile. Includes initialization within forward, clustering/routing/history/repair/reordering/heads. Excludes checkpoint/image loading and GT quality. GPU4–7 have CPU-only monitors, exactly one CUDA application per GPU until full process exit. Dense speedup N/A: no unchanged Dense rerun. Historical v8 heads12819.981s belongs to a different batch, not a paired speed comparison.','',
      (f"Changed-method warmed median **{timing['median_seconds']:.9f}s**, samples {[x['seconds'] for x in timing['samples']]}, maximum sample peak **{max(x['peak_allocated_gib'] for x in timing['samples']):.6f}GiB**." if timing else 'Timing incomplete.'),'',
      'Independent detailed event profile (parents are exclusive, not added twice); this profile wall is not the warmed benchmark median.','',table(profile,['operation','seconds','percent_full']),'',
      'The former Grid-normalizer kernel is absent. This first implementation still exports large raw probe logits, and adds actual-P head reduction/cache, Grid-mass accumulation and current-layer compensation. Removing one kernel is not a guarantee of the same total saving. Differences in selected supports and other stages must also be considered.','',
      '## Validation and limits','',
      'Numerical export, formal AV, bias/Special/tail/padding, FP16-u14 storage and per-Grid compensation were checked on FP32/BF16. Numerical Grid-mass conservation passes, but this does not establish geometric quality. Model layer records preserve actual density, underfill, prior patch-mass error and cache rounding excess. Actual sparse P includes compensation bias and normalizes over selected keys, unlike old v8 conditionals. This semantic difference and across-layer distribution prediction are plausible error sources; this experiment does not uniquely isolate their contributions.','',
      'Actual recorded diagnostics: '+json.dumps(diagnostics)+'. seed_normalization_max_error in this new method records excess of the quantized actual-sparse patch mass over1; it is not a geometric score and is distinct from the current Dense prior mass-conservation error.','',
      'The first numerical diagnostic used an inappropriate unquantized FP32 absolute tolerance for a FP16 cache; v2 checks the quantized reference. Both diagnostics and the failed v1 trace are preserved. This did not change the model or geometric quality gates. No successful geometry run was repeated to select a score.','',
      f'Source unchanged: {unchanged}; audited scene runs: {len(audit)}; missing: {missing}. Source archive and per-run source hashes, numerical diagnostics, monitors, raw six-metric quality JSON/Markdown, layer tables and historical controls are published. Timing-only diagnostics have all six quality fields unavailable with concrete reasons.','',
      'The existing v8 quality/time controls are historical, and neither quality equivalence nor a paired Dense speedup is claimed. Only the changed method was rerun.']
    (OUT/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(complete=complete,missing=missing,macro=[x for x in macros if x['method']=='beta_gridmass'],median_seconds=None if timing is None else timing['median_seconds']),indent=2))

if __name__=='__main__':main()
