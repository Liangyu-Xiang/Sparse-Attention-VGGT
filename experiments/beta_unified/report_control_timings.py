"""Paired full-forward timing, with timing-only vs reused quality kept distinct."""
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/paired_fullforward_7scenes128_20261109'
SCENES=['chess','fire','heads','office','pumpkin','redkitchen','stairs']
METHODS=['dense','quick','candidate'];KEYS=['auc03','auc30','acc','comp','cd','f1']

def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def csvfile(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    assert (OUT/'complete.json').exists(),'Do not report partial seven-scene macro'
    results={};rows=[];audit=[];raw=[]
    for scene in SCENES:
        group={m:json.loads((OUT/'runs'/scene/m/'result.json').read_text()) for m in METHODS}
        base=group['dense'];gpu=base['hardware']['physical_GPU']
        for method,d in group.items():
            assert d['input']==base['input'] and d['processed_image_sha256']==base['processed_image_sha256']
            assert d['hardware']==base['hardware'] and d['timing']['warmup']==1 and d['timing']['repeats']==3
            assert len(d['timing']['samples'])==3 and not d['quality']['complete']
            assert all(d['quality']['values'][k] is None for k in KEYS)
            monitor=json.loads((OUT/'monitors'/f'{scene}_{method}.json').read_text())
            assert not monitor['same_gpu_overlap_detected'] and monitor['returncode']==0
            seconds=d['timing']['median_seconds'];speedup=base['timing']['median_seconds']/seconds
            d['speedup_vs_dense']=speedup;d['speedup_reason']=None
            save(OUT/'runs'/scene/method/'result.json',d)
            rows.append(dict(method=method,scene=scene,physical_GPU=gpu,seconds=seconds,
                speedup_vs_Dense=speedup,warm_peak_GiB=max(x['peak_allocated_GiB'] for x in d['timing']['samples']),
                cold_peak_GiB=max(x['peak_allocated_GiB'] for x in d['timing']['cold_samples']),
                warm_min_seconds=min(x['seconds'] for x in d['timing']['samples']),
                warm_max_seconds=max(x['seconds'] for x in d['timing']['samples'])))
            results[(scene,method)]=d;raw.append(d)
        audit.append(dict(scene=scene,physical_GPU=gpu,all_inputs_identical=True,all_pixels_identical=True,
            same_timing_protocol=True,no_same_GPU_overlap=True,processed_image_sha256=base['processed_image_sha256']))
    macro=[];dense_mean=statistics.mean(results[(s,'dense')]['timing']['median_seconds'] for s in SCENES)
    for method in METHODS:
        rr=[r for r in rows if r['method']==method];seconds=statistics.mean(r['seconds'] for r in rr)
        macro.append(dict(method=method,scene_count=7,frames_per_scene=128,mean_scene_median_seconds=seconds,
            speedup_vs_Dense=dense_mean/seconds,mean_scene_speedup_vs_Dense=statistics.mean(r['speedup_vs_Dense'] for r in rr),
            max_warm_peak_GiB=max(r['warm_peak_GiB'] for r in rr),max_cold_peak_GiB=max(r['cold_peak_GiB'] for r in rr)))
    by={r['method']:r for r in macro}
    comparison=dict(candidate_seconds=by['candidate']['mean_scene_median_seconds'],quick_seconds=by['quick']['mean_scene_median_seconds'],
        candidate_over_quick_seconds_ratio=by['candidate']['mean_scene_median_seconds']/by['quick']['mean_scene_median_seconds'],
        candidate_speed_gate_macro_pass=by['candidate']['mean_scene_median_seconds']<=by['quick']['mean_scene_median_seconds'],
        candidate_speed_gate_every_scene_pass=all(results[(s,'candidate')]['timing']['median_seconds']<=results[(s,'quick')]['timing']['median_seconds'] for s in SCENES),
        quality_gate='Unchanged; no quality measured in this timing batch. Previous candidate failed strict six-metric quality gates.',
        macro_speedup_definition='Ratio of equal-scene mean warmed full-forward medians; per-scene ratios also reported separately',
        no_parameter_changes=True)
    # Six fields remain unavailable for every newly measured timing-only method.
    timing_quality=[dict(method=m,scene='macro7',**{k:None for k in KEYS},reason='Timing-only; no new pose/recon_unposed metrics; this is not a complete quality evaluation') for m in METHODS]
    reused=[];provenance=[];quality_per=[]
    for method in ['quick','candidate']:
        qq=[]
        for scene in SCENES:
            p=Path(results[(scene,method)]['reference_result']);d=json.loads(p.read_text());assert d['quality']['complete']
            qq.append(d['quality']['values']);quality_per.append(dict(method=method,scene=scene,**d['quality']['values'],provenance='Reused prior quality; not measured in timing batch'))
            provenance.append(dict(method=method,scene=scene,source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        reused.append(dict(method=method,scene='macro7',**{k:statistics.mean(q[k] for q in qq) for k in KEYS},reason='Reused complete prior quality; unchanged checkpoint/inputs/method'))
    reused.append(dict(method='dense',scene='macro7',**{k:None for k in KEYS},reason='No complete matched six-metric Dense quality reused; this request authorizes timing only'))
    csvfile(OUT/'timing_per_scene.csv',rows);csvfile(OUT/'timing_macro.csv',macro)
    csvfile(OUT/'timing_only_quality.csv',timing_quality);csvfile(OUT/'reused_quality_macro.csv',reused);csvfile(OUT/'reused_quality_per_scene.csv',quality_per)
    save(OUT/'raw_results.json',raw);save(OUT/'matched_audit.json',audit);save(OUT/'comparison.json',comparison)
    save(OUT/'quality_provenance.json',provenance)
    failed=[]
    for p in sorted((OUT/'failed_attempts').glob('*/*/quality.json')):
        d=json.loads(p.read_text());reason=d.get('reason','See retained failure.json')
        failed.append(dict(run=str(p.parent.relative_to(OUT)),**{k:None for k in KEYS},reason=reason,
                           speedup_vs_Dense=None,speedup_reason='Failed before timed forward; no valid timing'))
        (p.parent/'quality.txt').write_text('AUC@3=N/A; AUC@30=N/A; ACC=N/A; Comp=N/A; CD=N/A; F1=N/A\n'+reason+'\n')
    if failed:csvfile(OUT/'failed_attempts_quality.csv',failed)
    def fmt(r):return f"{r['method']} | {r['mean_scene_median_seconds']:.6f} | {r['speedup_vs_Dense']:.3f}x | {r['max_warm_peak_GiB']:.3f}"
    lines=['Paired same-batch7Scenes128 full-forward timing,all7 scenes,RTX5090D.',
        'method | mean per-scene warmed median seconds | Dense-relative speedup | max warm allocatedGiB',*[fmt(r) for r in macro],
        '', 'Scope:1cold+3warm synchronized forward calls;fresh process per method;eachscene samephysicalGPU;method order rotates byscene. Includes per-forward initialization/reset,clustering,routing,history,reordering,all camera/depth/point prediction heads. Model/checkpoint/image loading and CPU quality excluded. Track head has no querypoints,as in prior quality runs.',
        'Same FP32 model/prediction heads andBF16backbone autocast,TF32disabled,CuDNNbenchmarkenabled for all methods. OriginalDense unpatched;quick frozen oldcode and originalheadchunk8;candidate frozen previously evaluatedmodel withheadchunk32. Per-method optimization difference is retained,not silently applied to controls.',
        'Dense-relative macro speedup=equal-scene mean(Dense warmed median)/equal-scene mean(method warmed median);Dense=1.00x. Individualscene ratios are in timing_per_scene.csv;mean-of-ratios reported separately.',
        str(comparison), '', 'New timing-only six-metric table:',
        'method | AUC@3 | AUC@30 | ACC | Comp | CD | F1',
        *[m+' | N/A | N/A | N/A | N/A | N/A | N/A : explicitly timing only' for m in METHODS],
        '', 'Reused quality: AUC/F1 shown as percentages;ACC/Comp/CD metres;CD=(ACC+Comp)/2 unsquared:',
        *[r['method']+' | '+' | '.join('N/A' if r[k] is None else f"{100*r[k]:.2f}%" if k in ['auc03','auc30','f1'] else f"{r[k]:.5f}" for k in KEYS)+' : '+r['reason'] for r in reused],
        'Quality protocol reference only,not executed:unposed reconstruction threshold0.05m,downsample/voxel0.0078125m,TSDFtrunc0.04m;SAF3RTopK/profileN/A for these methods. Prior fullqualitysourcefiles+hashes recorded separately. No quality improvement or Dense degradation can be inferred from timing alone.',
        '', 'Per scene:',*[str(r) for r in rows],
        'Existing SAF3R/FastVGGT timings stay separate historical batches;no new paired speed claim for them. No paper comparison or H100 timing used.']
    (OUT/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:5]));print(comparison)

if __name__=='__main__':main()
