"""Paired report for representative Value / full-QK token-TopK schedule."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def report(root,reference=None):
    runs={p.parent.name:json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))}
    if len(runs)!=7:
        raise ValueError(f'Expected seven completed scenes, got {len(runs)}')
    methods=list(next(iter(runs.values()))['methods'])
    if any(set(r['methods'])!=set(methods) for r in runs.values()):
        raise ValueError('Cannot aggregate mismatched methods')
    layer_rows=[dict(scene=scene,**layer) for scene,r in runs.items()
                for layer in r['methods']['hybrid']['layers']]
    with (root/'hybrid_layers.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(layer_rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(layer_rows)
    aggregate={}
    for method in methods:
        rr=[r['methods'][method] for r in runs.values()]
        aggregate[method]=dict(scenes=len(rr),seconds=float(np.mean([r['median_seconds'] for r in rr])),
            global_seconds=float(np.mean([r['median_global_seconds'] for r in rr])),
            peak_gib=float(np.mean([r['median_peak_gib'] for r in rr])),
            **{k:100*float(np.mean([r['pose'][k] for r in rr])) for k in ['auc30','auc15','auc05','auc03']})
        if method=='hybrid':
            aggregate[method].update({key:float(np.mean([r['median_'+key] for r in rr]))
                for key in ['value_seconds','attention_topk_seconds','dense_prefix_seconds']})
            aggregate[method]['mean_connection_density']=float(np.mean([r['mean_connection_density'] for r in rr]))
    cold=all(r['input']['warmup']==0 and r['input']['repeats']==1 for r in runs.values())
    lines=['# 7Scenes128: representative Value / exact QK Top-K hybrid','',
        ('SINGLE COLD FORWARD PER SCENE: timings INCLUDE first-use compilation, not warmed formal speed measurements. Primary objective is seven-scene accuracy validation.' if cold else 'Warmed paired timing protocol; see metadata for repeats.'),
        'Zero-based global layers: 0–2 Dense, 3–9 Value, 10–16 full-QK Top-K, 17–23 Value. Frame attention unchanged.',
        'Value keeps real representative K/V, not Key blocks. One shared head-mean-V clustering, GPU lightweight L2 initialization and deterministic centroid-nearest member updates.',
        'Shared hierarchy is maintained in 10–16 as well, so layer 17 inherits layer 16, not stale layer 9.',
        'All specials mandatory and COUNT in fixed 10% total budget. TopK selects the remaining patch slots per head/query.',
        'All initialization, bridge-layer Value updates, full FP32 QK, exact patch TopK, renormalized probabilities and gathered AV INCLUDED in timings.',
        'No adaptive budget, online audit or extra selected-edge QK. Query chunks bound memory; masks remain token-wise per-query.',
        'SAF3R is the full profile, correspondence topk=128 per frame, not 128 global keys.',
        'Pose AUC only; no depth/reconstruction claim. Each scene is one designated sequence, 128 uniformly sampled frames.',
        'Within-scene medians over measured repeats, then seven-scene macro means. Check per-scene input metadata for warmups/repeats and source hashes.','',
        '| Method | Global24 s | Whole s | AUC30 % | AUC5 % | Peak GiB |',
        '|---|---:|---:|---:|---:|---:|']
    for method,r in aggregate.items():
        lines.append(f"| {method} | {r['global_seconds']:.4f} | {r['seconds']:.4f} | {r['auc30']:.4f} | {r['auc05']:.4f} | {r['peak_gib']:.3f} |")
    h=aggregate['hybrid']
    lines+=['','## Hybrid components (24-layer totals)','',
        f"- Dense prefix 0–2, INCLUDING L2 seed: {h['dense_prefix_seconds']:.4f}s.",
        f"- 14 representative-Value layers 3–9/17–23: {h['value_seconds']:.4f}s.",
        f"- 7 full-QK Top-K layers 10–16, INCLUDING their Value updates: {h['attention_topk_seconds']:.4f}s.",
        f"- Mean kept edges over all 24 layers: {100*h['mean_connection_density']:.5f}%; sparse layers target 10%, prefix layers 100%.",
        '- Kept-edge density is NOT computed-QK density: TopK layers still score all Keys.',
        '- Nominal QK+AV FLOPs: 10% in Value layers, 55% in TopK layers; overall 34.375% before clustering, sorting, memory access and projection costs.',
        '- FP32 QK/TopK and gathered AV can be much slower than tensor-core Flash Dense; nominal FLOPs do not establish latency speedup.','',
        '| Reference | Whole-model speedup (reference/hybrid) | Global speedup | Hybrid AUC30 difference (pp) |',
        '|---|---:|---:|---:|']
    for method in ['dense','saf3r']:
        if method in aggregate:
            r=aggregate[method]
            lines.append(f"| {method} | {r['seconds']/h['seconds']:.4f}x | {r['global_seconds']/h['global_seconds']:.4f}x | {h['auc30']-r['auc30']:+.4f} |")
    lines+=['','## Per scene','',
        '| Scene | Method | Whole s | Global24 s | AUC30 % | AUC5 % |',
        '|---|---|---:|---:|---:|---:|']
    for scene,run in runs.items():
        for method,r in run['methods'].items():
            lines.append(f"| {scene} | {method} | {r['median_seconds']:.4f} | {r['median_global_seconds']:.4f} | {100*r['pose']['auc30']:.4f} | {100*r['pose']['auc05']:.4f} |")
    if reference is not None:
        lines+=['','## Archived matched-input accuracy references','',
            'Reference poses are from the earlier seven-scene v4 Dense/SAF3R run. All frame paths, shapes and checkpoint paths must match.',
            'Only accuracy is compared here; archived runtimes are NOT mixed into a new paired speed claim.','',
            '| Scene | Dense AUC30 % | SAF3R AUC30 % | Hybrid AUC30 % |',
            '|---|---:|---:|---:|']
        refs={'dense':[],'saf3r':[]}
        for scene,run in runs.items():
            old=json.loads((reference/scene/'result.json').read_text())
            for key in ['image_paths','image_shape','checkpoint']:
                if run['input'][key]!=old['input'][key]:raise ValueError(f'Reference input drift: {scene}/{key}')
            for name in refs:refs[name].append(100*old['methods'][name]['pose']['auc30'])
            lines.append(f"| {scene} | {refs['dense'][-1]:.4f} | {refs['saf3r'][-1]:.4f} | {100*run['methods']['hybrid']['pose']['auc30']:.4f} |")
        aggregate['archived_accuracy_references']={name:dict(auc30=float(np.mean(v)),
            hybrid_gap_pp=h['auc30']-float(np.mean(v))) for name,v in refs.items()}
        lines.append(f"| Macro mean | {np.mean(refs['dense']):.4f} | {np.mean(refs['saf3r']):.4f} | {h['auc30']:.4f} |")
    lines+=['','This is NOT the old global-PAM seed + q=8 exact-medoid experiment. It restores the representative-token attention form while using the explicitly documented GPU lightweight hierarchy.',
        'Earlier full-layer block/adaptive results and L3–L9-only results are different policies and cannot substitute for these cumulative runs.','']
    (root/'report.md').write_text('\n'.join(lines))
    (root/'hybrid_aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--reference',type=Path)
    args=p.parse_args();report(args.root,args.reference)
