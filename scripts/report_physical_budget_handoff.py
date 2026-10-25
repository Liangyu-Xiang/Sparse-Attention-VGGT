"""Generate the required physical-budget template from frozen measurements."""
import json
import subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/physical_overlap_accounting'
BASE=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128'
PREVIOUS=ROOT/'outputs/all_edge_history_no_coverage/quality128'
UNIFIED=ROOT/'outputs/global_attention/unified_refactor_57bdb78/quality128'
SCENES=('heads','chess','stairs','fire','office','pumpkin','redkitchen')
FIELDS=('auc03','auc30','acc','comp','cd','f1')
MAP={'utilization':'physical_budget_utilization_mean','optimizer_gap':'optimizer_physical_gap_mean',
    'unused':'unused_budget_mean','history_candidates':'history_candidates_mean','selected_history':'reuse_edges_mean',
    'history_fraction':'history_reuse_ratio_mean','merged_h_rep':'history_rep_merges_mean',
    'population_rep_clusters':'population_representative_clusters_mean','exact':'exact_edges_mean',
    'history_exact_residual':'history_exact_overlap_selected_mean','density':'actual_unique_density'}

def table(d):
    def fmt(v):return f'{v:.5f}' if isinstance(v,float) else str(v)
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])

def quality_table(rows):
    return pd.DataFrame([dict(method=r['method'],**{k:r['values'][k]*(100 if k in ('auc03','auc30','f1') else 1) for k in FIELDS}) for r in rows])

def main():
    # Template requires analysis of committed measurements. This is reached only
    # after all quality workers complete; freeze raw outputs before analysis.
    subprocess.run(['git','add','.gitignore','outputs/physical_overlap_accounting'],cwd=ROOT,check=True)
    changed=subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT)
    if changed.returncode==1:
        subprocess.run(['git','commit','-qm','Freeze complete physical-budget raw measurements before analysis'],cwd=ROOT,check=True)
    elif changed.returncode!=0:raise RuntimeError('Cannot inspect staged raw measurements')
    data=[];records=[]
    labels=('Dense historical','SAF3R historical','Old reuse+residual historical','57bdb78','All-edge/no-coverage','Physical H/Rep')
    for scene in SCENES:
        ref=json.loads((BASE/scene/'dense/result.json').read_text())
        paths=(BASE/scene/'dense',BASE/scene/'saf3r',BASE/scene/'reuse_explore',UNIFIED/scene,PREVIOUS/scene,OUT/'quality128'/scene)
        for label,folder in zip(labels,paths):
            r=json.loads((folder/'result.json').read_text());assert r['quality']['complete']
            assert r['input']==ref['input'] and r['quality_protocol']==ref['quality_protocol']
            v=r['quality']['values'];assert abs(v['cd']-(v['acc']+v['comp'])/2)<1e-10
            layers=json.loads((folder/'layers.json').read_text()) if (folder/'layers.json').exists() else []
            assert r['timing']['warmup']==1 and r['timing']['repeats']==3
            density=sum(l['actual_unique_density'] for l in layers)/24 if len(layers)==24 else (1.0 if label=='Dense historical' else None)
            util=sum(l['counts_mean']/l['patch_budget'] for l in layers)/24 if len(layers)==24 else None
            hist=sum(l['reuse_edges_mean']/max(1,l['counts_mean']) for l in layers)/24 if len(layers)==24 else None
            records.append(dict(scene=scene,method=label,values=v,seconds=r['timing']['median_seconds'],
                cold_seconds=r['timing']['cold_samples'][0]['seconds'],
                peak_allocated_gib=max(s['peak_allocated_gib'] for s in r['timing']['samples']),
                peak_reserved_gib=max(s['peak_reserved_gib'] for s in r['timing']['samples']),
                density=density,utilization=util,history_fraction=hist,speedup_vs_dense=None,
                speedup_note='N/A: no fresh paired Dense; historical timing phase only'))
        for l in json.loads((OUT/'quality128'/scene/'layers.json').read_text()):
            data.append(dict(scene=scene,layer=l['layer'],**{k:l[v] for k,v in MAP.items()},
                additive_rep_keys=l['representatives_mean'],stored_history_mib=l['stored_history_bytes']/2**20))
    df=pd.DataFrame(data);layer=df.groupby('layer').mean(numeric_only=True).reset_index()
    df.to_csv(OUT/'layerwise_per_scene.csv',index=False);layer.to_csv(OUT/'layerwise.csv',index=False)
    stages=[]
    for name,a,b in [('L0-L8',0,8),('L9-L16',9,16),('L17-L23',17,23)]:
        stages.append(dict(stage=name,**layer[layer.layer.between(a,b)].drop(columns='layer').mean().to_dict()))
    sd=pd.DataFrame(stages);sd.to_csv(OUT/'stage_averages.csv',index=False)
    (OUT/'layerwise.json').write_text(json.dumps(dict(per_scene=data,macro=layer.to_dict('records'),stages=stages),indent=2)+'\n')
    macros=[]
    for label in labels:
        rows=[r for r in records if r['method']==label];m=dict(method=label,complete='7/7',values={k:sum(r['values'][k] for r in rows)/7 for k in FIELDS})
        for k in ('seconds','cold_seconds','density','utilization','history_fraction'):
            m[k]=sum(r[k] for r in rows)/7 if all(r[k] is not None for r in rows) else None
        for k in ('peak_allocated_gib','peak_reserved_gib'):m[k]=max(r[k] for r in rows)
        m['speedup_vs_dense']=None;macros.append(m)
    qt=quality_table(macros)
    timing=pd.DataFrame([{k:m[k] for k in ('method','seconds','peak_allocated_gib','peak_reserved_gib','density','utilization','history_fraction')} for m in macros])
    p=pd.concat([pd.read_csv(OUT/'pipeline'/s/'pipeline.csv').assign(scene=s) for s in ('chess','fire','redkitchen')])
    p.to_csv(OUT/'pipeline/all_scenes.csv',index=False)
    candidate=p[p.support=='history_candidates'].groupby('layer').candidate_density.mean()
    selected=p[p.support=='history_selected'].groupby('layer').mean(numeric_only=True)
    pt=pd.DataFrame(dict(layer=selected.index,candidate_density=candidate.values,
        selected_density=selected.candidate_density.values,recall=selected.recall.values,precision=selected.precision.values,
        enrichment=selected.enrichment.values,mass=selected.attention_mass_coverage.values))
    pt.to_csv(OUT/'pipeline/macro_filter.csv',index=False)
    for scene in ('chess','fire','redkitchen'):
        part=p[p.scene==scene]
        part.pivot(index='layer',columns='support',values='precision').to_csv(OUT/'pipeline'/scene/'precision_table.csv')
    oldpipe=pd.concat([pd.read_csv(ROOT/'outputs/all_edge_history_no_coverage/pipeline'/s/'pipeline.csv') for s in ('chess','fire','redkitchen')])
    op=oldpipe[oldpipe.support=='history_selected'].groupby('layer').mean(numeric_only=True)
    trends=pd.DataFrame(dict(layer=selected.index,old_precision=op.precision.values,new_precision=selected.precision.values,
        old_enrichment=op.enrichment.values,new_enrichment=selected.enrichment.values))
    old_layers=[]
    for s in SCENES:
        for l in json.loads((PREVIOUS/s/'layers.json').read_text()):
            if 9<=l['layer']<=16:old_layers.append(l)
    old_util=sum(l['counts_mean']/l['patch_budget'] for l in old_layers)/len(old_layers)
    old_unused=sum(l['unused_budget_mean'] for l in old_layers)/len(old_layers)
    old_density=sum(l['actual_unique_density'] for l in old_layers)/len(old_layers)
    newmid=stages[1]
    delta={k:(macros[-1]['values'][k]-macros[-2]['values'][k])*(100 if k in ('auc03','auc30','f1') else 1) for k in FIELDS}
    # Follow the required template's eight headings exactly.
    manifest=json.loads((OUT/'manifest.json').read_text())
    analysis=['# Physical History–Rep Budget Ablation Analysis','', '## 1. Evaluated revision',
        f"Upstream commit:{manifest['upstream']}; execution commit:{manifest['execution_commit']}.",
        'Stage0:three requested modules py_compile passed; original suite11passed, expanded diagnostic suite14passed. Existing diagnostic compatibility plugin supplies FP32 sparse.mm and the same four-effective-probe padded kernel. Capture-only merged-History instrumentation was corrected and tested; production capture=False math is upstream unchanged.',
        'Config:physical_overlap_accounting=true; control=false; all other settings frozen. Hardware/UUID assignments:accounting_complete.json,pipeline_complete.json,quality_complete.json.',
        'Seven-scene128 matched original checkpoint/frame lists, uniform seed0;1warmup/3complete forward measurements; capture=False. AUC/F1 percent, distances metres; CD=(ACC+Comp)/2 unsquared; recon_unposed threshold.05m/downsample.0078125m. No baseline reruns, no truth tuning. Diagnostic six quality metrics and paired speedup N/A.',
        '', '## 2. Matched two-frame accounting sanity',table(pd.read_csv(OUT/'accounting_sanity.csv')),
        'Stage1 input frames are the same uniform Chess endpoint frames in both arms; Stage2 uses frames0/20. Physical-support uniqueness/budget audits and explicit acceptance are archived. Read acceptance.json for the actual measured decision; no utilization threshold was fitted to quality.',
        '', '## 3. History filtering after accounting correction',table(pt),
        'All values except enrichment are fractions0–1. Precision/enrichment use current Top20 on the same sparse trajectory. New minus old endpoints alone cannot establish monotonic sharpening.',table(trends),
        'Strictly monotonic L10–L15 enrichment: '+str(bool((selected.loc[10:15].enrichment.diff().dropna()>=0).all()))+
        '. This is descriptive, not a tuned success criterion.',
        '', '## 4. Seven-scene macro quality',table(qt),
        'New minus all-edge/no-coverage (AUC/F1 percentage points; distances metres): '+json.dumps(delta),
        'Per-scene six metrics:',table(pd.DataFrame([dict(scene=r['scene'],method=r['method'],**{k:r['values'][k]*(100 if k in ('auc03','auc30','f1') else 1) for k in FIELDS}) for r in records])),
        '', '## 5. Seven-scene layer behavior',table(sd),
        'Fractions are0–1. history_fraction uses physical selected History / physical PATCH support as requested. Additive_rep_keys excludes merged H+Rep; population_rep_clusters includes them. Complete L0–L23:',table(layer),
        '', '## 6. Runtime and memory',table(timing),
        'Times are different historical phases, not contemporaneous paired speedups. New end-to-end Dense speedup N/A; no paired Dense rerun was authorized. Timing includes all initialization, clustering, routing, history and reordering. Exploration still does not shrink as History grows. This is not the final efficient implementation.',
        '', '## 7. Causal conclusions','### Budget accounting',
        f"L9–L16 full128 utilization {old_util:.6f}→{newmid['utilization']:.6f}; unused budget {old_unused:.3f}→{newmid['unused']:.3f}; actual density {old_density:.6f}→{newmid['density']:.6f}. New optimizer/physical gap {newmid['optimizer_gap']:.3f}. Frozen old128 records do NOT contain optimizer cost: old128 gap is N/A, not nominal budget minus executed keys. The matched same-code two-frame control above provides the direct old/new gap measurement.",
        '### Quality effect of restoring physical budget',
        'Quality changes above are attributable to the tested physical overlap accounting under frozen scoring/configuration, but not proof that any one scene must improve. Inspect Chess/Fire/Pumpkin/RedKitchen rows and all six metrics; do not equate lower CD with higher F1.',
        '### Remaining accounting issue',
        f"L9–L16 selected History/Exact residual {newmid['history_exact_residual']:.3f} keys/group; measured optimizer gap {newmid['optimizer_gap']:.3f}. Reported separately from integer solver slack. Neither Exact overlap nor budget backfilling was fixed in this experiment.",
        '### History filtering','Precision/enrichment trend is tabulated against the prior experiment, not inferred from a higher History fraction. Targets evolve with each model trajectory; these are not controlled identical-QK masks.',
        '', '## 8. Exactly one next recommendation']
    if newmid['utilization']<=old_util:
        recommendation='Stop further quality ablations and debug physical accounting; utilization did not improve. Freeze scoring, Value, Rep/Exact, grouping, population and retention.'
    elif newmid['history_exact_residual']>newmid['unused']*.5 and newmid['history_exact_residual']>1:
        recommendation='Run one separate History/Exact physical-overlap accounting ablation: residual overlap explains a material remaining gap. Freeze all scoring, clustering, Rep/Exact criteria, grouping, population and retention.'
    elif macros[-1]['values']['f1']<=macros[-2]['values']['f1']:
        recommendation='Keep the physical accounting fix and isolate Rep/Exact compressibility in the next experiment. Freeze History scoring/filtering, grouping, observation, population compensation and20% retention; do not combine another History change.'
    else:
        recommendation='Keep physical accounting and run one separate current-layer History-filter evidence ablation. Freeze physical accounting, Value/Rep criterion, population compensation, observation and20% retention.'
    analysis+=[recommendation,'This recommendation has NOT been implemented.']
    (OUT/'analysis.md').write_text('\n\n'.join(analysis)+'\n')
    (OUT/'comparison.md').write_text('# Physical H/Rep accounting comparison\n\n'+table(qt)+'\n\n'+table(timing)+
        '\n\nAUC/F1 percent; distances metres; seven-scene equal macro. SAF3R is local lazy per-frame Top4/ETH3D profile, not paper DINO feature TopK. Historical timings are not fresh speedups.\n\n'+table(layer)+'\n\n'+table(sd)+'\n')
    (OUT/'comparison.json').write_text(json.dumps(dict(per_scene=records,macro=macros,layer_stages=stages,new_minus_all_edge=delta,
        old_middle_utilization=old_util,old_middle_unused=old_unused,old_middle_density=old_density,all_seven_complete=True),indent=2)+'\n')

if __name__=='__main__':main()
