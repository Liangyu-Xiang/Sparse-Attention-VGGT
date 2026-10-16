"""Summarize all-layer paired runs without hiding actual adaptive density."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def report(root):
    runs={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))}
    if not runs:
        raise ValueError('No complete scene runs')
    methods=list(next(iter(runs.values()))['methods'])
    if any(set(r['methods'])!=set(methods) for r in runs.values()):
        raise ValueError('Scene methods differ; no unpaired aggregation allowed')
    aggregate={}
    head_rows=[]
    for method in methods:
        rr=[r['methods'][method] for r in runs.values()]
        aggregate[method]=dict(scenes=len(rr),seconds=float(np.mean([r['median_seconds'] for r in rr])),
            global_ms=1000*float(np.mean([r['median_global_seconds'] for r in rr])),
            peak_gib=float(np.mean([r['median_peak_gib'] for r in rr])),
            **{k:100*float(np.mean([r['pose'][k] for r in rr])) for k in ['auc30','auc15','auc05','auc03']})
        if 'layers' not in rr[0]:
            continue
        layers=[l for r in rr for l in r['layers']]
        aggregate[method].update(actual_density=float(np.mean([l['actual_density'] for l in layers])),
            fallback_fraction=float(np.mean([d['dense_fallback'] for l in layers for d in l['decisions']])),
            attention_active_fraction=float(np.mean([d['a_blocks']>0 for l in layers for d in l['decisions']])),
            audit_pass_fraction=float(np.mean([d['audit_passed'] for l in layers for d in l['decisions']])),
            chosen_budget_counts=dict(Counter(f"{d['rho']:.2f}" for l in layers for d in l['decisions'])))
        for scene,run in runs.items():
            for l in run['methods'][method]['layers']:
                for h,d in enumerate(l['decisions']):
                    head_rows.append(dict(scene=scene,method=method,layer_zero_based=l['layer_zero_based'],head=h,
                        **d,actual_density=l['actual_density'][h],C=l['concentration'][h],S=l['specificity'][h],
                        previous_head=l['head_matching'][h],history_enrichment=l['history_enrichment'][h],
                        reused_blocks=l['reused_blocks'][h],explored_blocks=l['explored_blocks'][h],
                        promoted_value_blocks=l['promoted_value_blocks'][h],history_blocks=l['history_blocks_per_head'][h]))
    layer_rows=[]
    for method in methods:
        if 'actual_density' not in aggregate[method]:
            continue
        for layer in range(24):
            rr=[r for r in head_rows if r['method']==method and r['layer_zero_based']==layer]
            layer_rows.append(dict(method=method,layer_zero_based=layer,**{k:float(np.mean([r[k] for r in rr]))
                for k in ['actual_density','dense_fallback','a_blocks','alpha','C','S','audit_nmse','audit_p90',
                          'audit_passed','reused_blocks','explored_blocks','promoted_value_blocks','history_blocks']}))
    write_csv(root/'per_scene_layer_head.csv',head_rows)
    write_csv(root/'per_layer.csv',layer_rows)
    (root/'report_aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    lines=['# Paired 7Scenes128: all 24 global layers','',
        f'Completed scenes ({len(runs)}): '+', '.join(runs)+'.',
        'One warmup, three measured repeats per method/scene; tables use per-scene timing medians, then scene macro means.',
        'Check result.json input metadata for exact repeats, input indices, checkpoint configuration and settings.',
        'Same checkpoint/frames/GPU per scene, BF16 autocast. Frame attention remains dense.',
        'All initialization, Value clustering, routing, online probe QK/output evaluation, selection, history and fallback costs are INCLUDED.',
        'SAF3R is the full profiled method, correspondence topk=128 per frame, not a global 128-key mask.',
        'Fixed10 has block-rounding; adaptive begins at 10% but may expand to 100%. Density includes mandatory special keys.',
        'AUC is a pose metric, not depth/reconstruction accuracy. Seven selected sequences are not an entire-dataset generalization claim.','',
        '| Method | Global24 ms | Whole model s | AUC30 % | AUC5 % | Actual edges % | Dense heads % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,r in aggregate.items():
        density=f"{100*r['actual_density']:.3f}" if 'actual_density' in r else ('100.000' if method=='dense' else 'profile-dependent')
        fallback=f"{100*r['fallback_fraction']:.3f}" if 'fallback_fraction' in r else '—'
        lines.append(f"| {method} | {r['global_ms']:.3f} | {r['seconds']:.6f} | {r['auc30']:.3f} | {r['auc05']:.3f} | {density} | {fallback} |")
    lines+=['','## Speed comparisons','',
        'Ratios below divide macro-mean times; >1 means the candidate is faster.','',
        '| Candidate | Global vs Dense | Whole vs Dense | Global vs SAF3R | Whole vs SAF3R |',
        '|---|---:|---:|---:|---:|']
    for method in ['fixed10','adaptive']:
        if method not in aggregate or not all(m in aggregate for m in ['dense','saf3r']):
            continue
        r=aggregate[method];dense=aggregate['dense'];saf=aggregate['saf3r']
        lines.append(f"| {method} | {dense['global_ms']/r['global_ms']:.3f}x | {dense['seconds']/r['seconds']:.3f}x | {saf['global_ms']/r['global_ms']:.3f}x | {saf['seconds']/r['seconds']:.3f}x |")
    lines+=['','## Per scene','',
        '| Scene | Method | Global24 ms | Whole s | AUC30 % | AUC5 % |',
        '|---|---|---:|---:|---:|---:|']
    for scene,run in runs.items():
        for method,r in run['methods'].items():
            lines.append(f"| {scene} | {method} | {1000*r['median_global_seconds']:.3f} | {r['median_seconds']:.6f} | {100*r['pose']['auc30']:.3f} | {100*r['pose']['auc05']:.3f} |")
    lines+=['','## Interpretation / safeguards','',
        '- C*S is not used as an edge fraction. Attention-active counts and actual mixtures are saved per head.',
        '- Budget-selection probes measure current sparse-prefix inputs; they do not establish closeness to the original Dense model at every later layer.',
        '- Passing sampled output-error thresholds does not certify all queries or final poses.',
        '- Dense fallback is exact at that head current input, not an assertion of a sparse speedup.',
        '- Prior legacy AUC30 40.393% is an archived result; new comparisons against Dense/SAF3R use this matched rerun.',
        '- Shared Value hierarchy is retained, but masks/history/budgets are now head-specific and query-block-specific.',
        '- Block pooling/quantization remains an approximation. Replacing the old scalar path does not prove the new pipeline beats optimized SAF3R.',
        '- Full masks are not stored by default; per-scene/layer/head decisions, match indices, densities and history lifecycle counters are retained.',
        '- Source/configuration and separate operator checks are documented in docs/experiments/adaptive_block_value.md.','']
    (root/'report.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    report(parser.parse_args().root)
