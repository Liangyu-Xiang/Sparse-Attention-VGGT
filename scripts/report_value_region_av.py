"""Paired Dense / old fused / whole-Value-region attention comparison."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())


def report(root):
    results={mode:read(root/mode/'chess/result.json') for mode in ('dense','fused','region')}
    dense=results['dense'];old=results['fused'];new=results['region']
    assert all(r['input']==dense['input'] and r['hardware']==dense['hardware'] for r in results.values())
    assert all(r['frames']==128 and r['dataset']=='7scenes' for r in results.values())
    assert all(r['protocol']['warmup']>=1 and r['protocol']['repeats']>=3 and not r['protocol']['capture'] for r in results.values())
    assert old['config']['attention_ratios']==new['config']['attention_ratios']
    assert old['config']['retentions']==new['config']['retentions']
    for layer in new['layers']:
        assert layer['terminal_member_qk_per_head']==0
        assert layer['actual_attention_min']>=layer['attention_edges_target']
        if layer['attention_edges_target']:
            assert layer['overshoot_max']<layer['max_fine_population']
    profiles={}
    for mode,result in results.items():
        totals=defaultdict(float)
        for row in result['profile']:totals[row['stage']]+=row['seconds']
        profiles[mode]=dict(totals)
    rows=[]
    for mode,result in results.items():
        rows.append(dict(mode=mode,seconds=result['median_seconds'],peak_gib=result['median_peak_gib'],
            speedup=dense['median_seconds']/result['median_seconds'],pose=result['metrics']['pose'],
            auc30_delta_pp=100*(result['metrics']['pose']['auc30']-dense['metrics']['pose']['auc30'])))
    summary=dict(scope='7Scenes Chess seq-01, 128 uniform frames, pose only',rows=rows,profile=profiles,
        latency_reduction_vs_old=1-new['median_seconds']/old['median_seconds'],
        actual_density_min=min(r['actual_density'] for r in new['layers']),
        actual_density_max=max(r['actual_density'] for r in new['layers']),
        max_overshoot_edges=max(r['overshoot_max'] for r in new['layers']),
        terminal_qk_per_head=0,acceptance=dict(speed_2x=dense['median_seconds']/new['median_seconds']>=2,
            memory_not_above_dense=new['median_peak_gib']<=dense['median_peak_gib'],
            chess_auc30_loss_at_most_1pp=100*(dense['metrics']['pose']['auc30']-new['metrics']['pose']['auc30'])<=1,
            full_seven_scene_three_task_assessment='not_run'))
    lines=['# Value-based whole-region Attention: Chess128 comparison','',
        'Same128 input frames, checkpoint,392×518 resolution, BF16, one warmup and three measured forwards. '
        'No primary per-layer hooks/synchronizations. Includes all hierarchy/selection/packing/history costs. '
        'Local/frame attention unchanged; all24 global layers use the manual mixed schedule.', '',
        '| Method | Full seconds ↓ | Speedup ↑ | Peak allocated GiB ↓ | AUC5 % ↑ | AUC15 % ↑ | AUC30 % ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        p=row['pose']
        lines.append(f"| {row['mode']} | {row['seconds']:.4f} | {row['speedup']:.3f}× | {row['peak_gib']:.4f} | "
            f"{100*p['auc05']:.4f} | {100*p['auc15']:.4f} | {100*p['auc30']:.4f} |")
    lines+=['',f"New vs old latency reduction: {100*summary['latency_reduction_vs_old']:.2f}%.",
        f"Actual unique edge density: {100*summary['actual_density_min']:.6f}%–{100*summary['actual_density_max']:.6f}%. "
        f"Largest per-Query-group budget overshoot: {summary['max_overshoot_edges']} edges; every layer satisfies overshoot < maximum fine cluster population.",
        '', '## Independent profile (seconds, not the primary timing)', '',
        '| Stage | Old fused | Whole regions |','|---|---:|---:|']
    for label,a,b in [('Global modules','global_module','global_module'),('Value','value','value'),
        ('Hierarchy / pooling','pool','value_region_hierarchy'),
        ('Selection incl. terminal / expansion','selection_including_terminal','selection_including_expansion'),
        ('Terminal real-Key QK (included in old selection)','terminal','terminal'),
        ('Parent scoring / ranking','region_parent','region_parent'),
        ('Crossing-parent child refinement','region_child_boundary','region_child_boundary'),
        ('Crossing-child fine refinement','region_fine_boundary','region_fine_boundary'),
        ('Whole-region expansion','region_expand','region_expand'),
        ('Final unified attention','execution','execution'),('History','history','history')]:
        lines.append(f"| {label} | {profiles['fused'].get(a,0):.4f} | {profiles['region'].get(b,0):.4f} |")
    lines+=['', 'Profile rows have containment relationships; do NOT sum selection and its sub-stages.', '',
        '## Per-layer edge budgets and selected complete regions','',
        '| Layer | Target A keys | Value keys | Unique edge % | Max overshoot | Whole parent regions | Whole child regions | Whole fine regions |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in new['layers']:
        lines.append(f"| {row['layer']} | {row['attention_edges_target']} | {row['value_edges']} | "
            f"{100*row['actual_density']:.6f} | {row['overshoot_max']} | {row.get('parent_regions',0)} | "
            f"{row.get('child_regions',0)} | {row.get('fine_regions',0)} |")
    lines+=['', '## Scope / limitations','',
        'The old fused selector remains available as mode=fused. mode=region uses the same Value maintenance '
        'and same manual20% target schedule, but selects complete Value-based regions, with explicitly allowed fine-cluster overshoot. '
        'These are different masks, not a bitwise-equivalent kernel optimization.', '',
        'The first parent boundary alone is refined to its children; the first child boundary alone is refined to fine clusters. '
        'Other chosen parents/children are final complete regions. Fine clusters are never cut. Parent scoring remains Query-group specific. '
        'A Value representative/history Key excludes only that Key, not the whole partially covered cluster.', '',
        'No terminal real-Key QK, no full selected logits, and no Token×Token mask. Expansion uses a bounded '
        'window-of-Query-groups × patch-count uint8 history lookup and compact support indices; this workspace '
        'and its clearing/packing are included in time and memory.', '',
        'History remains a same-pass tile-mass approximation. All heads retain their own Q/K/V and unified softmax. '
        'This single-scene pose experiment is not a full seven-scene quality acceptance.', '',
        'Acceptance checks: '+json.dumps(summary['acceptance']), '']
    (root/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:15]))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    report(p.parse_args().root)
