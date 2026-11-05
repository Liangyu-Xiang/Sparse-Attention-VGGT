"""Audit matched inputs/protocols and report the frozen ETH3D repair campaign."""
import argparse,csv,hashlib,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['auc03','auc30','acc','comp','cd','f1']

def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def table(rows):
    out=['| Dataset / scope | Method | Scenes | AUC@3 | AUC@30 | ACC (m) | Comp (m) | CD (m) | F1 |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        v=r['values'];values=[f'{100*v[k]:.3f}%' if k in ('auc03','auc30','f1') else f'{v[k]:.6f}' for k in FIELDS]
        out.append('| '+' | '.join([r['dataset'],r['method'],str(r['scenes'])]+values)+' |')
    return '\n'.join(out)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'outputs/eth3d_repair_20261105');a=p.parse_args()
    out=a.output;dest=out/'reports';dest.mkdir(exist_ok=True)
    plan=read(out/'frozen_plan.json');old=ROOT/'outputs/integrated_quality_speed_20261105/final_runs'
    scene_rows=[];macro=[];audits=[];runtimes=[]
    for ds,scenes in plan['datasets'].items():
        grouped={'before_main':[],'anchor_concat':[]}
        for scene in scenes:
            before=read(old/ds/scene/'main/result.json');folder=out/'final_runs'/ds/scene/'anchor_concat';after=read(folder/'result.json')
            assert before['quality']['complete'] and after['quality']['complete'],(ds,scene,'incomplete')
            checks={k:before['input'][k]==after['input'][k] for k in ('image_paths','frame_indices','shape','checkpoint_sha256')}
            checks['quality_protocol']=before['quality_protocol']==after['quality_protocol']
            checks['frozen_config']=hashlib.sha256(Path(plan['methods']['anchor_concat']['config']).read_bytes()).hexdigest()==plan['methods']['anchor_concat']['config_sha256']
            checks['frozen_model_source']=all(after['source_sha256'].get(k)==v for k,v in plan['model_source_sha256'].items())
            checks['unchanged_evaluation_source']=all(before['source_sha256'][k]==v for k,v in after['source_sha256'].items() if k.startswith('evaluation/'))
            monitor=read(out/'final_runs'/f'{ds}_{scene}_anchor_concat.monitor.json')
            checks['gpu_overlap_free']=monitor['returncode']==0 and not monitor['same_gpu_overlap_detected']
            assert all(checks.values()),(ds,scene,checks)
            audits.append(dict(dataset=ds,scene=scene,checks=checks))
            for name,d in [('before_main',before),('anchor_concat',after)]:
                v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-9
                grouped[name].append(v)
                scene_rows.append(dict(dataset=ds,scene=scene,method=name,frames=d['frames'],**v,speedup_vs_dense=None,speedup_reason='No authorized paired warmed Dense timing; ETH3D speed not requested'))
            layers=read(folder/'layers.json')
            density=statistics.mean(l['actual_unique_density'] for l in layers)
            assert len(layers)==24
            assert all(l['protected_patch_keys']==after['input']['shape'][-2]//14*(after['input']['shape'][-1]//14) for l in layers)
            runtimes.append(dict(dataset=ds,scene=scene,frames=after['frames'],median_seconds=after['timing']['median_seconds'],warmup=after['timing']['warmup'],repeats=after['timing']['repeats'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in after['timing']['samples']),sparsity=1-density,hardware=after['hardware'],speedup_vs_dense=None,speedup_reason='Unchanged controls reused; no paired warmed Dense timing',scope='Complete model forward, all clustering/routing/history/reset/reordering included; validation synchronization retained'))
        for name,values in grouped.items():macro.append(dict(dataset=ds,method=name,scenes=len(values),values={k:statistics.mean(v[k] for v in values) for k in FIELDS}))
    diagnostics=[]
    for directory in sorted(out.iterdir()):
        if not directory.is_dir() or not (directory/'reference_repeat.json').exists():continue
        repeat=read(directory/'reference_repeat.json');assert all(v['equal'] for v in repeat.values()),directory
        for q in sorted(directory.glob('*/result.json')):
            d=read(q);diagnostics.append(dict(batch=directory.name,variant=d['label'],quality=d['quality'],deviation_from_dense=d.get('prediction_deviation'),instrumented_seconds=d['instrumented_seconds'],peak_allocated_gib=d['peak_allocated_gib']))
    save(dest/'summary.json',dict(macros=macro,per_scene=scene_rows,runtimes=runtimes,audits=audits,diagnostics=diagnostics,units='AUC and F1 fractions in JSON; metres for ACC/Comp/CD',complete=True))
    for name,rows in [('per_scene',scene_rows),('runtime',runtimes)]:
        with (dest/f'{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# ETH3D regression repair','',
          'Frozen before quality evaluation. Same checkpoint, exact RGB frames, preprocessing, pose, alignment/fusion and reconstruction settings as before_main. Equal-scene macro averages; all 11 ETH3D and all seven 7Scenes scenes required. No 7Scenes test-truth parameter tuning.','',
          'Before_main is the unchanged, complete previous batch; it was not rerun. This is a local measured comparison, not a paper reproduction.','',table(macro),'',
          'CD=(ACC+Comp)/2 with unsquared distances. AUC/F1 in this report are percentages; raw JSON uses fractions. ETH3D threshold/downsample: 0.25 m / 0.0390625 m; 7Scenes: 0.05 m / 0.0078125 m. Pose + recon_unposed complete for each run. This method does not use SAF3R Top-K or its head profile.','',
          '## Diagnosis and intervention','',
          'Current Dense quality was reproduced on pipes and playground to check baseline/environment drift; see controls/. Independent Dense numerical trajectories were repeated exactly for every diagnostic batch. These numerical diagnostics have all six geometry metrics marked N/A and are not quality evaluations.','',
          'The former calibration compared Dense attention on already corrupted sparse Q/K/V, which misses accumulated feature error. Dense Camera Query correction therefore gives a misleadingly small local error. Smaller Query groups and extra clustering iterations were not consistent cures.','',
          'The repair preserves every patch Key in the first coordinate-anchor frame inside the existing total Key budget, removes those patches from clustering, and clusters the remaining rectangular frame domain. It also replaces head-mean Value features with the concatenation of separate heads, preventing cross-head cancellation in the selection geometry. Other algorithm parameters and layer retention caps remain frozen. First-frame Dense Query correction was tested and rejected, not enabled in the final method.','',
          'The anchor ablation improves independent Dense fidelity; this supports a causal contribution, not a claim that all approximation error is eliminated. Complete six-metric results above determine the actual quality outcome.','',
          '## Runtime scope','',
          '7Scenes uses one full-forward warmup followed by three complete timed forwards per scene. End-to-end speedup relative to Dense and the SAF3R +10% target are N/A: unchanged controls were not rerun for timing. Previous batch timings are historical, not paired comparisons. ETH3D uses no warmup and one forward solely for quality; those cold/instrumented times are not speed benchmarks.','']
    seven=[r for r in runtimes if r['dataset']=='7scenes']
    text+=['7Scenes equal-scene mean of warmed per-scene medians: %.6f s; maximum timed allocated memory: %.6f GiB; mean actual global-attention sparsity: %.6f%%.'%(statistics.mean(r['median_seconds'] for r in seven),max(r['peak_allocated_gib'] for r in seven),100*statistics.mean(r['sparsity'] for r in seven)),'',
           '## Per-scene quality','']
    text.append(table([dict(dataset=r['dataset']+'/'+r['scene'],method=r['method'],scenes=1,values={k:r[k] for k in FIELDS}) for r in scene_rows]))
    (dest/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(macros=macro,quality_runs=len(scene_rows)//2,diagnostic_runs=len(diagnostics),audits_passed=len(audits)),indent=2))

if __name__=='__main__':main()
