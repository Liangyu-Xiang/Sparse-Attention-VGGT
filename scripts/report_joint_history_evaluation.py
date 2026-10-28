"""Seven-scene quality plus fresh same-UUID Dense/new-method timing report."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/joint_history_06c5e0c'
BASE=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128'
SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
FIELDS=('auc03','auc30','acc','comp','cd','f1')

def table(d):
    def fmt(x):return f'{x:.5f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
        ['| '+' | '.join(fmt(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)])

def main():
    runs=[];timings=[];layers=[]
    method_roots={'Dense quality (historical)':BASE,'SAF3R quality (historical)':BASE,'History80 (historical)':BASE,
        'Old reuse+residual (historical)':BASE,'57bdb78':ROOT/'outputs/global_attention/unified_refactor_57bdb78/quality128',
        'Physical H/Rep baseline':ROOT/'outputs/physical_overlap_accounting/quality128','Joint History (new)':OUT/'quality128'}
    subdirs={'Dense quality (historical)':'dense','SAF3R quality (historical)':'saf3r','History80 (historical)':'history80','Old reuse+residual (historical)':'reuse_explore'}
    for scene in SCENES:
        reference=json.loads((BASE/scene/'dense/result.json').read_text())
        for method,root in method_roots.items():
            f=root/scene/subdirs[method]/'result.json' if method in subdirs else root/scene/'result.json'
            d=json.loads(f.read_text());assert d['quality']['complete']
            assert d['input']==reference['input'] and d['quality_protocol']==reference['quality_protocol']
            v=d['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
            runs.append(dict(scene=scene,method=method,values=v,recorded_seconds=d['timing']['median_seconds'],
                peak_allocated_gib=max(x['peak_allocated_gib'] for x in d['timing']['samples'])))
        folder=OUT/'quality128'/scene
        pair=json.loads((folder/'paired_timing.json').read_text());dense=json.loads((OUT/'paired_dense128'/scene/'diagnostic.json').read_text());new=json.loads((folder/'result.json').read_text())
        assert dense['input']==new['input'] and dense['hardware']==new['hardware']
        timings.append(dict(scene=scene,device_uuid=pair['device_uuid'],dense_seconds=pair['dense_seconds'],
            method_seconds=pair['method_seconds'],speedup=pair['speedup'],uncontended=pair['uncontended'],
            dense_peak_allocated_gib=max(x['peak_allocated_gib'] for x in dense['timing']['samples']),
            method_peak_allocated_gib=max(x['peak_allocated_gib'] for x in new['timing']['samples']),
            dense_peak_reserved_gib=max(x['peak_reserved_gib'] for x in dense['timing']['samples']),
            method_peak_reserved_gib=max(x['peak_reserved_gib'] for x in new['timing']['samples'])))
        data=json.loads((folder/'layers.json').read_text());assert len(data)==24
        for l in data:
            layers.append(dict(scene=scene,layer=l['layer'],actual_density=l['actual_unique_density'],
                utilization=l['physical_budget_utilization_mean'],optimizer_gap=l['optimizer_physical_gap_mean'],
                history_fraction=l['history_reuse_ratio_mean'],history_count=l['reuse_edges_mean'],
                population_rep_clusters=l['population_representative_clusters_mean'],
                additive_rep_keys=l['representatives_mean'],exact=l['exact_edges_mean'],
                history_promoted_representatives=l.get('history_promoted_representatives_mean',0),
                history_mib=l['stored_history_bytes']/2**20,unused_budget=l['unused_budget_mean']))
    macro=[]
    for method in method_roots:
        r=[x for x in runs if x['method']==method];assert len(r)==7
        macro.append(dict(method=method,values={k:sum(x['values'][k] for x in r)/7 for k in FIELDS},
            mean_recorded_scene_seconds=sum(x['recorded_seconds'] for x in r)/7))
    dense_time=sum(x['dense_seconds'] for x in timings)/7;new_time=sum(x['method_seconds'] for x in timings)/7
    valid=all(x['uncontended'] for x in timings)
    speed=dense_time/new_time if valid else None
    df=pd.DataFrame(layers);ld=df.groupby('layer').mean(numeric_only=True).reset_index()
    df.to_csv(OUT/'layerwise_per_scene.csv',index=False);ld.to_csv(OUT/'layerwise.csv',index=False)
    qrows=[dict(method=r['method'],**{k:r['values'][k]*(100 if k in ('auc03','auc30','f1') else 1) for k in FIELDS}) for r in macro]
    per_scene=[dict(scene=r['scene'],method=r['method'],**{k:r['values'][k]*(100 if k in ('auc03','auc30','f1') else 1) for k in FIELDS}) for r in runs]
    prior=macro[-2]['values'];latest=macro[-1]['values'];delta={k:(latest[k]-prior[k])*(100 if k in ('auc03','auc30','f1') else 1) for k in FIELDS}
    lines=['# Joint History-conditioned optimization:7Scenes128 evaluation','',
        'Evaluated upstream06c5e0c760b34b34c3f1a715a99417dd0406fd76. Current default joint_history_exploration=true;20% retention; all24 global layers; local/frame unchanged. No production modifications or7Scenes test-truth tuning.',
        'The fixed relation-candidate-slot / adaptive-fresh-QK proposal remains a DOCUMENTED PLAN, not the implementation tested here.',
        'Original checkpoint and exact input lists/preprocessing/pose+recon_unposed protocols matched. AUC/F1 percent, distances metres, CD=(ACC+Comp)/2 unsquared; threshold.05m,downsample.0078125m.',
        '', '## Seven-scene equal macro quality',table(pd.DataFrame(qrows)),
        'Dense quality is reused from frozen matched results. Fresh Dense runs below are TIMING-ONLY, have six explicit N/A quality fields and must not be called new quality evaluations.',
        'SAF3R is frozen local lazy first-correspondence-head per-frame Top4/ETH3D profile, not paper DINO-feature TopK reproduction.',
        '', '## Fresh paired runtime and peak memory',table(pd.DataFrame(timings)),
        f'Mean scene warmed medians:Dense={dense_time:.5f}s,new={new_time:.5f}s; macro end-to-end speedup={speed if speed is not None else "N/A: at least one pair has foreign GPU contention"}. Dense reference=1.00x. Ratio is mean Dense scene medians / mean method scene medians, not mean per-scene ratios.',
        'Each pair uses the same UUID,input,checkpoint and timing protocol:1warmup/3whole forwards,capture=False. Includes all initialization,Value updates,selection,history,reordering. No per-layer sync. Foreign GPU processes are monitored; contaminated pairs are markedN/A. Cold samples remain in forward.json.',
        '', '## Historical latency context (NOT fresh speed comparison)',table(pd.DataFrame([dict(method=r['method'],mean_recorded_seconds=r['mean_recorded_scene_seconds']) for r in macro])),
        '', '## Per-scene six quality metrics',table(pd.DataFrame(per_scene)),
        '', '## Layer behavior',table(ld),
        'history_fraction is physical History / patch support; merged population representation is reflected by population_rep_clusters, not only additive_rep_keys.',
        '', '## Interpretation',
        'New minus previous physical H/Rep baseline:'+json.dumps(delta)+'; AUC/F1 percentage points, distances metres.',
        'This is a measured tradeoff, not a claim of universal improvement. Inspect AUC@3,AUC@30 and reconstruction together. No planned fresh-probe efficiency benefit may be attributed to this current implementation.',
        'All quality gates, speed and memory must be considered separately; no thresholds or method settings were altered after observing quality.']
    (OUT/'comparison.md').write_text('\n\n'.join(lines)+'\n')
    (OUT/'comparison.json').write_text(json.dumps(dict(per_scene=runs,macro=macro,paired_timings=timings,
        mean_dense_seconds=dense_time,mean_method_seconds=new_time,speedup_vs_dense=speed,
        all_pairs_uncontended=valid,new_minus_physical_baseline=delta,all_seven_complete=True),indent=2)+'\n')

if __name__=='__main__':main()
