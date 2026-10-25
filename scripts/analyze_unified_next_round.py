"""Render handoff template sections from frozen Stage1 CSVs; no model/tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/unified_next_round'
SCENES=('chess','fire','redkitchen')

def table(df,percent=False):
    def cell(x):return f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x)
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    rows+=['| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(rows)

def combined(category,name):
    return pd.concat([pd.read_csv(OUT/category/s/name).assign(scene=s) for s in SCENES],ignore_index=True)

def run():
    completion=json.loads((OUT/'execution_complete.json').read_text())
    assert completion['all_success'],'Do not summarize incomplete runs as successful'
    ceiling=combined('reuse_ceiling','transitions.csv')
    pipe=combined('history_pipeline','pipeline.csv')
    gran=combined('query_granularity','granularity.csv')
    clusters=combined('compressibility','clusters.csv')
    ranking=json.loads((OUT/'compressibility/summary.json').read_text())
    scale=pd.read_csv(OUT/'history_scaling/scaling.csv')
    lines=['# Unified Next-Round Analysis','',
        'Upstream:0856426c0537195255946b356f027416d349be5b. Diagnostic corrections are recorded in git; production files unchanged.',
        'Stage0: original tests9passed/1BF16 sparse failure; diagnostic compatibility10passed; expanded suite12passed. Six requested scripts passed py_compile.',
        'All five Stage1 categories completed. Chess/Fire/RedKitchen, seq-01, frames0/20, original checkpoint and preprocessing. No Stage2 or model tuning.',
        'All values below are diagnostic, not pose/reconstruction quality evaluations: AUC@3,AUC@30,ACC,Comp,CD,F1 and paired Dense speedup are N/A (not evaluated). Each run has quality.json/quality.md.',
        'Recall and mass tables use percentages. Tables aggregate three scenes equally. Dense reuse holds head index fixed; it is not a best cross-layer head-matching oracle. Two-frame results do not establish128-frame accuracy.',
        'Dense targets/candidates are patch-only20% (integer rounding); sparse20% total budget includes special keys, so its patch-only candidate density is about19.5–19.6%. Pipeline diagnostic targets are full patch QK on the same sparse trajectory, not the original Dense network trajectory.',
        '', '## 1. Dense adjacent-layer reuse ceiling','']
    avg=ceiling.groupby(['source_layer','target_layer','candidate_retention']).mean(numeric_only=True).reset_index()
    rows=[]
    for (source,target),d in avg.groupby(['source_layer','target_layer']):
        r=dict(Pair=f'L{int(source)}→L{int(target)}')
        for ratio in (.2,.4,.5,.8):r[f'Prev Top{int(ratio*100)} recall %']=float(d.loc[np.isclose(d.candidate_retention,ratio),'recall'].iloc[0])*100
        r['Prev Top20 mass %']=float(d.loc[np.isclose(d.candidate_retention,.2),'attention_mass_coverage'].iloc[0])*100;rows.append(r)
    lines+=[table(pd.DataFrame(rows)),'']
    middle=ceiling[ceiling.target_layer.between(11,15)].groupby('candidate_retention')[['recall','attention_mass_coverage','key_saliency_control_recall','key_saliency_control_mass','same_prefix_jaccard']].mean()*100
    lines+=['Middle subset is exactly L10→11 through L14→15. Candidate retention is a fraction of ALL patch keys, not a fraction of the final20% mask.',table(middle.reset_index()),
        'Even high recall from80% candidates cannot be described as80% useful reuse inside a20% budget. Same-size20% overlap is the meaningful equal-budget observation.','']
    lines+=['## 2. Current History pipeline','']
    pa=pipe.groupby(['layer','support'])[['recall','attention_mass_coverage','candidate_density']].mean()
    pr=pa.recall.unstack()*100;pr=pr[['previous_full','history_candidates','history_selected','final_support']]
    lines+=[table(pr.reset_index()),'','Candidate density (%) for the SAME four supports:',table((pa.candidate_density.unstack()[pr.columns]*100).reset_index()),
        'Current diagnostic patch-attention mass coverage (%):',table((pa.attention_mass_coverage.unstack()[pr.columns]*100).reset_index()),'']
    losses=pd.DataFrame({'layer':pr.index,'bank_loss_pp':pr.previous_full-pr.history_candidates,'selection_loss_pp':pr.history_candidates-pr.history_selected})
    lines+=['Loss decomposition (percentage points):',table(losses),'',
        'H1/H2/H3 may coexist and vary by layer. Compare both absolute and relative drops; a candidate pool of only a fraction of1% cannot have high full Top20 recall. Sparse-trajectory targets and Dense-trajectory targets must not be subtracted as if identical.','']
    lines+=['## 3. Query-group granularity','']
    ga=gran.groupby(['source_layer','target_layer','group_size'])[['recall','attention_mass_coverage']].mean()
    for field,title in [('recall','Top20 recall (%)'),('attention_mass_coverage','Current attention mass coverage (%)')]:
        t=ga[field].unstack()[[32,16,8,4,1]]*100
        lines+=[title,table(t.reset_index()),'']
    deltas=[]
    for scene,d in gran.groupby('scene'):
        g=d.groupby('group_size')[['recall','attention_mass_coverage']].mean()
        deltas.append(dict(scene=scene,recall_group32=100*g.loc[32,'recall'],recall_per_query=100*g.loc[1,'recall'],
            recall_gain_pp=100*(g.loc[1,'recall']-g.loc[32,'recall']),mass_gain_pp=100*(g.loc[1,'attention_mass_coverage']-g.loc[32,'attention_mass_coverage'])))
    lines+=[table(pd.DataFrame(deltas)), 'All groups use20% candidates, and head sharing remains fixed. Per-query here still means HEAD-SHARED support, not per-head Top20. Smaller groups alone cannot remove head-sharing loss.','']
    lines+=['## 4. Value-cluster compressibility','', '### Metric ranking','']
    metric=pd.DataFrame.from_dict(ranking['metrics'],orient='index').reset_index().rename(columns={'index':'signal'})
    lines+=[table(metric),'',f"Best measured Precision@20% signal: **{ranking['best_by_precision']}**. This is a diagnostic ranking, not a validated deployment threshold.",
        'Raw value_rel_l2/key_rel_l2 are ratios of summed squared L2 energies (no square root), despite their names. Spearman ranks average ties. Reference error is importance × mass-preserving relative Value error; it does NOT measure the affinity/mass error itself.','']
    summary=[];examples=[]
    for scene,d in clusters.groupby('scene'):
        low=float(d.value_cosine_dist.quantile(.25));high=float(d.rep_affinity_error.quantile(.75))
        bad=d[(d.value_cosine_dist<=low)&(d.rep_affinity_error>=high)]
        summary.append(dict(scene=scene,pairs=len(d),low_value_q25=low,high_affinity_q75=high,bad_pairs=len(bad),fraction_all=len(bad)/len(d),fraction_low_value=len(bad)/max(1,int((d.value_cosine_dist<=low).sum()))))
        examples.append(bad.nlargest(5,'rep_affinity_error'))
    lines+=['Descriptive low/high quadrants: bottom25% Value cosine distortion AND top25% affinity error, per scene. These are analysis quantiles, NOT tuned production thresholds.',table(pd.DataFrame(summary)),'']
    ex=pd.concat(examples);ex.to_csv(OUT/'compressibility/low_value_high_affinity_examples.csv',index=False)
    lines+=[table(ex[['scene','layer','query_group','cluster','cluster_size','value_cosine_dist','rep_affinity_error','mass_preserving_value_error','selected_state']]),'']
    dist=clusters.groupby('selected_state')[['value_cosine_dist','key_cosine_dist','qk_logit_std','rep_affinity_error','mass_preserving_value_error']].median()
    dist['pairs']=clusters.groupby('selected_state').size()
    lines+=['Current Rep / exact-like state median metrics:',table(dist.reset_index()),
        'The rep/exact_like labels are support-derived inferred states, not randomized treatments. Correlations and state differences do not establish that switching the criterion will improve AUC/F1.','']
    charts=[('value_cosine_dist','rep_affinity_error','value_vs_affinity'),('value_cosine_dist','mass_preserving_value_error','value_vs_contribution'),
            ('qk_logit_std','mass_preserving_value_error','logit_std_vs_contribution'),('key_cosine_dist','mass_preserving_value_error','key_vs_contribution')]
    for x,y,name in charts:
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(clusters[x],clusters[y],s=3,alpha=.15);ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'compressibility'/f'{name}.png',dpi=160);plt.close(fig)
        lines.append(f'![{name}](compressibility/{name}.png)')
    lines+=['','## 5. History reuse scaling','',table(scale[['reuse_ratio','history_items','fine_items','exploration_ms_window','solver_ms_window','total_ms_window','projected_all_windows_ms','peak_allocated_mib']]),
        'The x-axis is synthetic History CANDIDATE ITEMS / budget, not measured accepted History reuse. Fine exploration table is fixed. Projected time covers ONLY exploration-state construction + joint objective/dual solver; not pooling, packing, evidence or full model.',
        'The structural requirement more reuse → less exploration work → cheaper mask construction is **not implemented**: all Fine exploration items are constructed for every History pool size. Timing measures the cost of this structure, not a deployed speedup.','']
    lines+=['## 6. Causal conclusions','', '### History','',
        '**Primary case H1, with a bootstrap limitation consistent with H3; H2 is secondary in these clips.** Previous-full recall is14.875–20.127%, whereas History candidates collapse to0% inL9–L12 and only0.442% atL16. L16 loses14.433pp at candidate construction versus0.097pp at selection. Relative selection loss atL16 is not zero, but it is not the dominant absolute loss.',
        'The candidate pool density is0–0.252% of patch keys, versus about19.5% for the previous full support. Tiny History ratios are therefore not simply evidence that adjacent-layer Dense relations do not exist.',
        'Code evidence: L0 labels all representative keys non-genuine; subsequent History stores only `real` edges (retained History or Exact expansion). A high-weight representative QK is still excluded. Early Rep-only layers have no genuine seed, hence no History candidate and zero explicit-relation coverage. Later Exact expansion slowly introduces candidates. This explains the observed near-empty pipeline without changing the production scoring formula.',
        'H3 qualification: previous-full support is itself only near its roughly20% density, not a high-quality oracle support. Candidate starvation is established; improving the candidate pool alone is not proven sufficient for end-to-end accuracy. H4 is not supported because high final-support recall was not observed.',
        'Dense reuse is real but modest: middle Top20→Top20 recall35.557% versus25.512% saliency control and20% random expectation. Full L8–L17 Top20 recall averages34.549%; Top50 gives66.806%, Top80 gives88.957%. Thus50–80% RECALL can be reached only with wider candidates for many pairs;50–80% accepted reuse inside the final20% budget is NOT established.',
        'Query grouping is another measurable limitation: group32→per-query improves recall8.134/8.440/8.660pp on Chess/Fire/RedKitchen and mass9.544/10.052/10.110pp. This is a Dense offline control; it is not a causal measurement on the sparse trajectory and cannot be attributed solely to the History scorer.','',
        '### Rep / Exact','',
        'Value similarity alone does not guarantee representative QK affinity accuracy. The descriptive low-Value/high-affinity quadrant contains6005/76032 sampled pairs (7.90% overall), and29.5–34.3% of each scene\'s low-Value quartile. Low here is relative to the sampled distribution (cosine thresholds0.094–0.105), not an absolute near-identity guarantee.',
        'Selected states are75761 Rep,267 exact-like,4 other:99.64% Rep among these sampled important pairs. Rep median mass-preserving relative contribution error is0.448. This supports concern about aggressive representative use, but does not prove that reallocating the fixed budget to Exact will improve quality.',
        'Exact-like medians are confounded by144 singleton clusters out of267; do not claim their zero median errors demonstrate a randomized Exact advantage. These samples are top64 clusters per group and layer, not all clusters or independent scene samples.',
        'Value squared-L2 has the best measured rank correlation0.433 and Precision@20%=30.78%. Among the requested Attention-side candidates, affinity error has best precision29.12%, Key squared-L2 has best rank correlation0.342; QK std precision28.05%. No Attention-side signal wins consistently over Value for this MASS-PRESERVING reference.',
        'The reference deliberately assigns true cluster mass and therefore removes affinity/mass error. This experiment cannot establish the best predictor of total Attention-output error or conclude that population correction is wrong. Keep population compensation out of scope. The old prototype advantage hypothesis H-D remains UNRESOLVED: no old-prototype matched causal ablation was run.',
        'Efficiency H-E is supported for the benchmarked objective/solver: Fine items remain26009 at every History size. Exploration stays0.130–0.140ms/window, total2.185ms atzero History versus2.380ms at80% History candidates; projected subtotal1.137s→1.239s (~8.9% increase), peak13.10→27.71MiB. The implementation does not yet reduce unresolved exploration work with reuse. This does not measure actual accepted reuse or whole mask construction.','',
        '## 7. What NOT to change yet','',
        '- No production History scoring, Value clustering, Rep/Exact logic, population compensation or query-group size changed.',
        '- Do not infer improved pose/reconstruction from a diagnostic support metric.',
        '- Do not use7Scenes test AUC/F1 or these descriptive quantiles to tune a production threshold.',
        '- Do not call the two-frame fixed-head overlap a128-frame geometric-relation theorem.',
        '', '## 8. Next end-to-end experiment','',
        '**One History module change only: History evidence/candidate eligibility.** Permit the real representative Key whose QK was actually executed to supply a single-edge History candidate, using its existing population-divided single-edge salience, rather than restricting eligibility to retained History/Exact members. Keep the current gain formula, budget solver, total20% retention, Value grouping, population correction and group32 fixed. Coverage should consume the same newly eligible single-edge evidence; do not independently retune its formula. This tests H1 and the observed seed starvation, not a guarantee of improved accuracy.',
        '**Rep/Exact production change: none recommended yet.** Current cheap Key/QK signals did not beat Value under the prescribed reference. If a separate tiny-clip diagnostic ablation is approved, use the documented affinity-error oracle to test affinity-related failure independently, with the same clusters/budget and explicit total-output error measurement. It is not an accelerator and should not be combined with the History change in the first causal test.',
        'No Stage2 has been executed. Any eventual deployment criterion/threshold must be selected on independent calibration, frozen, and then evaluated with all six quality metrics on7Scenes128.']
    (OUT/'analysis.md').write_text('\n\n'.join(lines)+'\n')
    for category,name in [('reuse_ceiling','transitions.csv'),('history_pipeline','pipeline.csv'),('query_granularity','granularity.csv')]:
        for s in SCENES:
            d=pd.read_csv(OUT/category/s/name)
            if category=='history_pipeline':
                for field in ('recall','candidate_density','attention_mass_coverage'):
                    d.pivot(index='layer',columns='support',values=field).to_csv(OUT/category/s/f'{field}_table.csv')
    (OUT/'analysis_statistics.json').write_text(json.dumps(dict(dense_middle=middle.reset_index().to_dict('records'),pipeline_losses=losses.to_dict('records'),
        query_group_changes=deltas,low_value_high_affinity=summary),indent=2)+'\n')

if __name__=='__main__':run()
