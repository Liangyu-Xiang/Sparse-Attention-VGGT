"""Validate and summarize three frozen, complete 7Scenes branch revisions."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/global_attention'
SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
VERSIONS=(('3afe65a','unified_refactor_20261025'),('46bd65b','unified_refactor_46bd65b'),
          ('57bdb78','unified_refactor_57bdb78'))
METRICS=('auc03','auc30','acc','comp','cd','f1')

def cells(v):
    return [f'{v[k]*100:.4f}' if k in ('auc03','auc30','f1') else f'{v[k]:.5f}' for k in METRICS]

def run():
    records=[];aggregates=[];baselines={}
    for revision,directory in VERSIONS:
        folder=BASE/directory
        assert json.loads((folder/'completion.json').read_text())['all_seven_complete']
        dense={}
        comparisons=json.loads((folder/'comparison.json').read_text())
        for r in comparisons['per_run']:
            if r['method']=='dense':dense[r['scene']]=r
        for scene in SCENES:
            path=folder/'quality128'/scene/'result.json'
            d=json.loads(path.read_text());q=d['quality']
            assert q['complete'] and all(q['values'][k] is not None for k in METRICS)
            v=q['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
            assert d['quality_protocol']==dense[scene]['quality_protocol']
            ref=BASE/'reuse_explore_20261021/quality128'/scene/'dense/result.json'
            reference=json.loads(ref.read_text())
            assert d['input']==reference['input'] and d['frames']==128
            assert d['timing']['warmup']==1 and d['timing']['repeats']==3
            assert d['config']['retention']==.2
            assert reference['quality']['complete']
            layer_rows=json.loads((path.parent/'layers.json').read_text())
            assert len(layer_rows)==24
            records.append(dict(revision=revision,scene=scene,values=v,
                forward_seconds=d['timing']['median_seconds'],
                peak_allocated_gib=max(s['peak_allocated_gib'] for s in d['timing']['samples']),
                actual_unique_density=mean(l['actual_unique_density'] for l in layer_rows),
                speedup_vs_dense=None,speedup_unavailable='No contemporaneous paired Dense timing',
                source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        rows=[r for r in records if r['revision']==revision]
        aggregates.append(dict(revision=revision,complete='7/7',values={k:mean(r['values'][k] for r in rows) for k in METRICS},
            mean_scene_median_seconds=mean(r['forward_seconds'] for r in rows),
            max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
            actual_unique_density=mean(r['actual_unique_density'] for r in rows),speedup_vs_dense=None))
        if not baselines:baselines={r['method']:r for r in comparisons['macro'] if r['method']!='unified'}
    lines=['# Unified Reuse / Exploration: three completed revision evaluations','',
        'All three revisions completed all seven 7Scenes scenes, 128 frames per scene, exact matched inputs/checkpoint/protocol.',
        'All24 global layers sparse; frame/local attention unchanged. Total edge retention cap20%, including special Keys.',
        'One warmup, three measured whole forwards including initialization, clustering, routing, history and evidence.',
        'Quality: AUC/F1 percentages, reconstruction distances metres; CD=(ACC+Comp)/2 unsquared.',
        'Unposed reconstruction threshold .05m; downsampling .0078125m. No test-truth parameter tuning.',
        '', '## Changes and provenance','',
        '- `3afe65a`: first unified allocator, cluster-local History candidates and per-cluster cap.',
        '- `46bd65b`: direct query-group History relation pool, no per-cluster cap, compact bank and merged representative deduplication.',
        '- `57bdb78`: additionally scales History gains by previous explicit-relation coverage in the canonical grid.',
        '- Every revision includes the disclosed FP32 sparse mapping / four-effective-probe padded Tensor Core compatibility overlay; see each evaluation_overlay.patch and source snapshots.',
        '- Dataset/model files are not uploaded. Archived results and code snapshots are immutable experimental evidence, not a replacement for current method source.',
        '', '## Equal-scene macro quality','',
        '| Method/revision | Complete | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels={'dense':'VGGT Dense (historical)','saf3r':'SAF3R lazy Top4 (historical)',
            'history80':'History80 + population (historical)','reuse_explore':'Original reuse + residual (historical)'}
    for method,r in baselines.items():lines.append('| '+' | '.join([labels[method],'7/7']+cells(r['quality']['values']))+' |')
    for r in aggregates:lines.append('| '+' | '.join([r['revision'],r['complete']]+cells(r['values']))+' |')
    lines+=['','## Timing, memory and actual edge retention','',
        'Rows are recorded in separate evaluation phases, not contemporaneous matched speed tests. No newly measured Dense speedup is available; historical ratios must not be presented as new speedups.',
        'Timing includes selector costs; final edge density is not total computed-QK density. Peak is allocated memory, not reserved.',
        '', '| Revision | Mean scene warmed median s | Max peak allocated GiB | Mean retained edges % | Paired Dense speedup |',
        '|---|---:|---:|---:|---|']
    for r in aggregates:lines.append(f"| {r['revision']} | {r['mean_scene_median_seconds']:.3f} | {r['max_peak_allocated_gib']:.3f} | {r['actual_unique_density']*100:.4f} | N/A: no contemporary pair |")
    lines+=['','## Per-scene quality for every evaluated revision','',
        '| Revision | Scene | AUC@3 % | AUC@30 % | ACC m | Comp m | CD m | F1 % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append('| '+' | '.join([r['revision'],r['scene']]+cells(r['values']))+' |')
    lines+=['','## Interpretation and limitations','',
        'Neither update consistently improves all quality metrics. Latest macro F1 remains below matched Dense and both earlier revisions. Do not infer a component-level causal conclusion without controlled ablations.',
        'SAF3R baseline uses local lazy first-correspondence-head per-frame Top4 with the ETH3D profile, NOT paper DINO feature TopK reproduction. Local measured results are not published-reference scores.',
        'These are seven-scene128-frame evaluations, not300-frame runs or evaluation of every dataset frame.',
        '', '## Full frozen reports','']
    for revision,directory in VERSIONS:
        lines.append(f'- [{revision}: per-scene baseline comparison and provenance](../../outputs/global_attention/{directory}/comparison.md)')
    (ROOT/'docs/experiments/unified_reuse_exploration_revision_results.md').write_text('\n'.join(lines)+'\n')
    (BASE/'unified_revision_comparison.json').write_text(json.dumps(dict(per_scene=records,macro=aggregates,
        historical_baselines=baselines,all_seven_required=True,quality_units='AUC/F1 fractions; distances metres'),indent=2)+'\n')
    print('Validated21 complete scene-runs; generated revision Markdown and raw JSON')

if __name__=='__main__':run()
