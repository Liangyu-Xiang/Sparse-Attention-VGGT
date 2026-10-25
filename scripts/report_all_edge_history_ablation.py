"""Handoff stages3–5 from frozen raw outputs, no tuning or method changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/all_edge_history_no_coverage'
BASE=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128'
PREV=ROOT/'outputs/global_attention/unified_refactor_57bdb78/quality128'
SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
FIELDS=('auc03','auc30','acc','comp','cd','f1')

def table(d):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.5f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])

def quality_row(label,values):
    return dict(method=label,**{k:(v*100 if k in ('auc03','auc30','f1') else v) for k,v in values.items()})

def main():
    runs=[];layers=[];records=[]
    methods=('dense','saf3r','reuse_explore','previous_unified','all_edge_no_coverage')
    labels=('Dense (historical)','SAF3R lazy Top4 (historical)','Old reuse+residual (historical)','57bdb78 (previous)','All-edge/no-coverage (new)')
    for scene in SCENES:
        ref=json.loads((BASE/scene/'dense/result.json').read_text())
        for method in methods:
            p=PREV/scene/'result.json' if method=='previous_unified' else OUT/'quality128'/scene/'result.json' if method=='all_edge_no_coverage' else BASE/scene/method/'result.json'
            r=json.loads(p.read_text());assert r['quality']['complete']
            assert r['input']==ref['input'] and r['quality_protocol']==ref['quality_protocol']
            q=r['quality']['values'];assert abs(q['cd']-(q['acc']+q['comp'])/2)<1e-10
            records.append(dict(scene=scene,method=method,values=q,forward_seconds=r['timing']['median_seconds'],
                cold_seconds=r['timing']['cold_samples'][0]['seconds'],
                peak_allocated_gib=max(s['peak_allocated_gib'] for s in r['timing']['samples']),
                peak_reserved_gib=max(s['peak_reserved_gib'] for s in r['timing']['samples']),
                speedup_vs_dense=r['quality'].get('speedup_vs_vggt') if method in ('dense','saf3r','reuse_explore') else None,
                speedup_note='Historical phase only' if method in ('dense','saf3r','reuse_explore') else 'N/A: no contemporaneous paired Dense timing'))
        new=json.loads((OUT/'quality128'/scene/'result.json').read_text());runs.append(new)
        lr=json.loads((OUT/'quality128'/scene/'layers.json').read_text());assert len(lr)==24
        for l in lr:
            layers.append(dict(scene=scene,layer=l['layer'],history_candidates=l['history_candidates_mean'],
                selected_history=l['reuse_edges_mean'],history_fraction_patch=l['history_reuse_ratio_mean'],
                history_fraction_all=l['reuse_edges_mean']/(l['counts_mean']+l['special_keys']),
                filter_keep=l['history_filter_keep_ratio_mean'],rep_overlap=l['history_rep_candidate_overlap_mean'],
                unused_budget=l['unused_budget_mean'],rep=l['representatives_mean'],exact=l['exact_edges_mean'],
                stored_history_mib=l['stored_history_bytes']/2**20,actual_unique_density=l['actual_unique_density']))
    data=pd.DataFrame(layers);macro=data.groupby('layer').mean(numeric_only=True).reset_index()
    data.to_csv(OUT/'quality128/layerwise_per_scene.csv',index=False);macro.to_csv(OUT/'layerwise.csv',index=False)
    stage=[]
    for label,a,b in [('L0-L8',0,8),('L9-L16',9,16),('L17-L23',17,23)]:
        vals=macro[macro.layer.between(a,b)].drop(columns='layer').mean().to_dict();stage.append(dict(stage=label,**vals))
    pd.DataFrame(stage).to_csv(OUT/'stage_averages.csv',index=False)
    (OUT/'layerwise.json').write_text(json.dumps(dict(per_scene=layers,macro=macro.to_dict('records'),stages=stage),indent=2)+'\n')
    quality=[];timing=[];macros=[]
    for method,label in zip(methods,labels):
        rows=[r for r in records if r['method']==method]
        v={k:float(np.mean([r['values'][k] for r in rows])) for k in FIELDS}
        quality.append(quality_row(label,v))
        t=dict(method=label,mean_scene_median_s=float(np.mean([r['forward_seconds'] for r in rows])),
            mean_cold_s=float(np.mean([r['cold_seconds'] for r in rows])),
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),
            speedup='N/A: no fresh pair' if method in ('previous_unified','all_edge_no_coverage') else 'historical only')
        timing.append(t);macros.append(dict(method=method,values=v,timing=t))
    header=['# All-edge History / no-coverage: 7Scenes128 comparison','',
        'Seven complete scenes per method; equal-scene macros. AUC/F1 percent; ACC/Comp/CD metres. CD=(ACC+Comp)/2 unsquared.',
        'Original checkpoint and exact128-frame lists/protocols checked. Threshold .05m, downsample .0078125m.1warmup/3complete measured forwards; capture disabled.',
        'All24 global layers use20% total edge retention; no local/frame attention changes. No test-truth tuning.',
        'SAF3R: local lazy first-correspondence-head per-frame Top4/ETH3D profile, not paper DINO-feature TopK reproduction.',
        'Old timings are frozen historical measurements, not contemporaneous speed comparisons. New Dense speedup is N/A; initialization/clustering/routing/history included in full latency.',
        '', '## Macro quality',table(pd.DataFrame(quality)),'','## Timing and memory',table(pd.DataFrame(timing)),
        '', '## Per-scene six metrics',table(pd.DataFrame([dict(scene=r['scene'],**quality_row(r['method'],r['values'])) for r in records])),
        '', '## All24-layer behavior (seven-scene equal averages)',
        'Fractions below are0–1, not percentages. history_fraction_all includes special keys; history_fraction_patch is the recorded per-query-group patch-only fraction. Do not confuse candidates with accepted reuse.',
        table(macro),'','## Stage averages',table(pd.DataFrame(stage)),
        'The independent allocator can count the same physical History/Rep candidate twice before packing. rep_overlap and unused_budget are reported, NOT corrected in this ablation.']
    (OUT/'comparison.md').write_text('\n\n'.join(header)+'\n')
    (OUT/'comparison.json').write_text(json.dumps(dict(per_scene=records,macro=macros,layer_stages=stage,all_seven_complete=True),indent=2)+'\n')
    pipe=pd.concat([pd.read_csv(OUT/'pipeline'/s/'pipeline.csv').assign(scene=s) for s in ('chess','fire','redkitchen')],ignore_index=True)
    pipe.to_csv(OUT/'pipeline/all_scenes.csv',index=False)
    a=['# All-executed-edge History + no-coverage causal analysis','',
        'Evaluated upstream1a7bb41d955da284eded195a2e0b11faf49c2ec3 with disclosed evaluation-only compatibility. Stage0 logs, Stage1 sanity, seven complete quality runs and all raw outputs are archived.',
        'Only upstream all-edge eligibility + removal of coverage are under test. These two coupled upstream edits are not separated factorially; their individual effects cannot be identified here.',
        '', '## A. Did bootstrap deadlock disappear?',
        'Stage1 verifies candidate density, key count, recall and mass match previous_full at every L9–L16 on all three clips. Stage0 full-layer tests verify History is written from all executed patch support afterL0, available startingL1. This establishes semantic eligibility, not the usefulness of every candidate.']
    pts=[]
    for scene in ('chess','fire','redkitchen','macro'):
        df=pipe if scene=='macro' else pipe[pipe.scene==scene]
        df=df.groupby(['layer','support']).mean(numeric_only=True).reset_index()
        rows=[]
        for l in range(9,17):
            d=df[df.layer==l].set_index('support');h=d.loc['history_selected'];c=d.loc['history_candidates']
            rows.append(dict(layer=l,previous_density=d.loc['previous_full','candidate_density'],candidate_density=c.candidate_density,
                selected_density=h.candidate_density,recall=h.recall,precision=h.precision,enrichment=h.enrichment,
                mass=h.attention_mass_coverage,filter_keep=h.candidate_keys/max(1e-30,c.candidate_keys)))
        pd.DataFrame(rows).to_csv(OUT/'pipeline'/f'{scene}_filter_table.csv',index=False)
        a+=['',f'### {scene}: Stage1 filtering (fractions0–1; enrichment is a multiplier)',table(pd.DataFrame(rows))]
        if scene=='macro':pts=rows
    a+=['','## B. Does implicit filtering work?']
    p10=next(r for r in pts if r['layer']==10);p15=next(r for r in pts if r['layer']==15)
    a+=[f"L10→L15 selected precision {p10['precision']:.5f}→{p15['precision']:.5f}; enrichment {p10['enrichment']:.5f}→{p15['enrichment']:.5f}; mass {p10['mass']:.5f}→{p15['mass']:.5f}. This reports endpoints, not an assumption of monotonic progression. Inspect all layers/scenes above.",
        'Precision is selected support membership in current patch Top20; enrichment divides by actual integer-rounded target density. Recall alone cannot establish a good filter. These two-frame targets are on the sparse trajectory, not identical to a Dense trajectory.',
        '', '## C. How much History is actually selected?',table(macro[macro.layer.between(9,16)]),
        'All-layer and stage-average tables are in comparison.md and layerwise.csv. Candidate bank size is not selected History fraction. Inclusion/exclusion of mandatory special keys is labeled explicitly.',
        '', '## D. Does quality improve?',table(pd.DataFrame(quality))]
    prev=macros[-2]['values'];new=macros[-1]['values']
    dq={k:(new[k]-prev[k])*(100 if k in ('auc03','auc30','f1') else 1) for k in FIELDS}
    a+=['New minus57bdb78: '+json.dumps(dq)+'. AUC/F1 differences are percentage points; distance differences metres.',
        'Per-scene differences for the requested reconstruction-sensitive scenes:']
    dr=[]
    for scene in ('redkitchen','fire','pumpkin'):
        old=next(r for r in records if r['scene']==scene and r['method']=='previous_unified')['values']
        now=next(r for r in records if r['scene']==scene and r['method']=='all_edge_no_coverage')['values']
        dr.append(dict(scene=scene,**{k:(now[k]-old[k])*(100 if k in ('auc03','auc30','f1') else 1) for k in FIELDS}))
    a+=[table(pd.DataFrame(dr)), '', '## E. What is the efficiency cost?',table(pd.DataFrame(timing)),table(pd.DataFrame(stage)),
        'All-edge History banks retain a full previous20% patch support per query group; stored_history_mib is recorded explicitly. The unchanged selector still evaluates every Fine exploration item. Higher reuse is not claimed to reduce exploration work.',
        'History/Rep overlap and unused budgets are confounds of independent cost accounting; this experiment deliberately does NOT fix them. Timings across historical phases do not establish a fresh paired speedup.',
        '', '## Evidence-based next action (not implemented)']
    if new['f1']>prev['f1'] and new['auc03']>=prev['auc03']:
        recommendation='Retain corrected all-edge History semantics; next isolate whether selected History can reduce unresolved exploration work. Do not change Rep/Exact simultaneously.'
    elif stage[1]['history_fraction_all']<.01:
        recommendation='Candidate bank is healthy but accepted History remains small: next isolate History score/shared-budget calibration, not candidate generation.'
    elif p15['enrichment']<=1 or new['f1']<prev['f1']:
        recommendation='Do not increase History blindly. Its bank is healthy but quality/filtering does not establish success; next isolate filtering strength or insufficient coarse evidence with one controlled ablation, keeping Rep/Exact fixed.'
    else:
        recommendation='Keep History fixed for the next diagnostic and isolate Rep/Exact approximation; do not mix it into this completed History ablation.'
    a+=[recommendation,'No additional production change was implemented. All quality metrics use the same frozen checkpoint/inputs; no thresholds were tuned using7Scenes truth.']
    (OUT/'analysis.md').write_text('\n\n'.join(a)+'\n')

if __name__=='__main__':main()
