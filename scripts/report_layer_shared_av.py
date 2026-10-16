"""Report complete runs, actual A/V profiles, and archived accuracy references."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np

SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenes, layers, aggregates = [], [], {}
    for name, path in [('reference', args.reference), ('screened_tau08_v1', args.tuned),
                       ('tau09_noncanonical_v1', args.selected), ('first_corrected_tau09_v3', args.first_corrected),
                       ('selected_joint_v3', args.final)]:
        results = []
        for scene in SCENES:
            source = path/scene/'result.json'
            if not source.exists():
                raise ValueError(f'Incomplete experiment: {source}')
            r = json.loads(source.read_text())
            if len(r['layers']) != 24 or r['config']['sensing_only']:
                raise ValueError('Not a complete sparse-policy validation')
            if not r['measured_pose_identical']:
                raise ValueError('Predictions changed across measured repeats')
            results.append(r)
            row = dict(method=name, scene=scene, auc30=100*r['pose']['auc30'],
                auc05=100*r['pose']['auc05'], seconds=r['median_seconds'],
                global_seconds=r['median_global_seconds'], peak_gib=r['median_peak_gib'])
            for label, start, end in [('early', 3, 10), ('middle', 10, 17), ('late', 17, 24)]:
                row[label+'_attention'] = float(np.mean([l['attention_keys']/(l['attention_keys']+l['value_keys']) for l in r['layers'][start:end]]))
            scenes.append(row)
            for l in r['layers']:
                layers.append(dict(method=name, scene=scene, **{k:v for k,v in l.items() if k != 'stage_seconds'},
                    **{k+'_seconds':v for k,v in l['stage_seconds'].items()}))
        hashes = {json.dumps(r['source_sha256'], sort_keys=True) for r in results}
        if len(hashes) != 1: raise ValueError('Source differs across scenes in one arm')
        subset = [r for r in scenes if r['method'] == name]
        aggregates[name] = {k: float(np.mean([r[k] for r in subset])) for k in subset[0] if k not in ('method', 'scene')}
        aggregates[name]['pure_value_layer_fraction'] = float(np.mean([l['attention_keys'] == 0 for r in results for l in r['layers']]))
        aggregates[name]['pure_attention_layer_fraction'] = float(np.mean([l['value_keys'] == 0 for r in results for l in r['layers']]))
        aggregates[name]['actual_density'] = float(np.mean([l['actual_density'] for r in results for l in r['layers']]))
        aggregates[name]['last_forward_stage_seconds'] = {k: float(np.mean([sum(l['stage_seconds'][k] for l in r['layers']) for r in results])) for k in results[0]['layers'][0]['stage_seconds']}
        heldout = [r for r in subset if r['scene'] != 'chess']
        aggregates[name]['heldout_six'] = {k: float(np.mean([r[k] for r in heldout])) for k in heldout[0] if k not in ('method', 'scene')}
    for name, rows in [('per_scene', scenes), ('per_layer', layers)]:
        with (args.output_dir/(name+'.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader(); writer.writerows(rows)
    (args.output_dir/'aggregate.json').write_text(json.dumps(aggregates, indent=2)+'\n')
    text = ['# 7Scenes128 revised A/V experiment', '',
        'Zero-based layers. Fixed final density including mandatory special keys; local/frame attention unchanged.',
        'One warmup + three measured forwards per scene. Macro means of per-scene medians.', '',
        '| Method | AUC30 (%) | AUC5 (%) | Whole (s) | Global (s) | A 3–9 (%) | A 10–16 (%) | A 17–23 (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method, r in aggregates.items():
        text.append(f"| {method} | {r['auc30']:.3f} | {r['auc05']:.3f} | {r['seconds']:.3f} | {r['global_seconds']:.3f} | {100*r['early_attention']:.2f} | {100*r['middle_attention']:.2f} | {100*r['late_attention']:.2f} |")
    text += ['', 'Reference: multiplier1, tau.9, eta1%. Screened v1: multiplier32, tau.8, eta.25%. Selected v3: multiplier32, tau.95, eta.25%, query chunk1024, canonical coarse-QK tiles128 and token-level history capacity.',
        'Initial screening used Chess offline sensing without pose labels. Final selection used the correct-policy joint Chess scale/threshold sweep; other six scenes were not used to select those parameters.',
        'Offline Dense-trajectory calibration ratios are NOT actual sparse-execution ratios; the table uses actual retained A slots.', '',
        '| Layer | Reference raw alpha (%) | Final raw alpha (%) | Final A slots (%) | Value-only scenes / 7 |',
        '|---|---:|---:|---:|---:|']
    for layer in range(24):
        rr = [r for r in layers if r['method'] == 'reference' and r['layer'] == layer]
        tt = [r for r in layers if r['method'] == 'selected_joint_v3' and r['layer'] == layer]
        text.append(f"| {layer} | {100*np.mean([r['alpha_raw'] for r in rr]):.3f} | {100*np.mean([r['alpha_raw'] for r in tt]):.2f} | {100*np.mean([r['attention_keys']/(r['attention_keys']+r['value_keys']) for r in tt]):.2f} | {sum(r['attention_keys']==0 for r in tt)} |")
    text += ['', 'Chess was used for calibration, so the seven-scene macro mean includes one calibration scene. Independent held-out six-scene results:', '',
        '| Method | Held-out AUC30 (%) | Held-out AUC5 (%) |', '|---|---:|---:|']
    for name, r in aggregates.items():
        h = r['heldout_six']
        text.append(f"| {name} | {h['auc30']:.3f} | {h['auc05']:.3f} |")
    archived = {}
    for name in ['dense', 'saf3r']:
        rr = []
        for scene in SCENES:
            path = Path('outputs/adaptive_block_value/7scenes_n128_audit02_v4')/scene
            row = json.loads((path/f'{name}_result.json').read_text())
            old_input = json.loads((path/'result.json').read_text())['input']
            new_input = json.loads((args.reference/scene/'result.json').read_text())['input']
            if any(old_input[k] != new_input[k] for k in ('image_paths', 'image_shape', 'checkpoint')):
                raise ValueError('Archived accuracy has mismatched inputs')
            rr.append(dict(scene=scene, auc30=100*row['pose']['auc30']))
        archived[name] = dict(auc30=float(np.mean([r['auc30'] for r in rr])),
            heldout_six_auc30=float(np.mean([r['auc30'] for r in rr if r['scene'] != 'chess'])))
    (args.output_dir/'archived_accuracy.json').write_text(json.dumps(archived, indent=2)+'\n')
    text += ['', f"Archived accuracy references (same input paths/shape/checkpoint checked, NOT paired timings): {json.dumps(archived)}", '',
        'Earlier fixed hybrid AUC30 83.817% is also an archived accuracy-only reference.',
        'Earlier fixed hybrid kept global layers 0–2 Dense and used full fine Dense QK TopK on layers10–16; it is NOT the present all24 fixed-10% method.',
        'Do not infer speedup or safety from a good branch profile alone. See aggregate.json for stage times and pure-mode fractions.',
        'Stage profiles are from each scene\'s last measured forward, not medians of per-stage repeats; do not sum them and compare against median whole/global latency.', '']
    paired_path = Path('outputs/layer_shared_av/paired_fresh_topk_chess_n128/chess/result.json')
    paired = json.loads(paired_path.read_text())
    chosen = json.loads((args.final/'chess/result.json').read_text())
    if any(paired['input'][k] != chosen['input'][k] for k in ('image_paths', 'image_shape', 'checkpoint')):
        raise ValueError('Fresh baseline input mismatch')
    runner = Path('scripts/benchmark_adaptive_block_value.py')
    if paired['input']['source_sha256'][str(runner)] != hashlib.sha256(runner.read_bytes()).hexdigest():
        raise ValueError('Fresh baseline runner changed; verify per-forward Top-K reset')
    comparison = {name:dict(auc30=100*r['pose']['auc30'], seconds=r['median_seconds'],
        global_seconds=r['median_global_seconds']) for name,r in paired['methods'].items()}
    comparison['selected_joint_v3'] = dict(auc30=100*chosen['pose']['auc30'],
        seconds=chosen['median_seconds'], global_seconds=chosen['median_global_seconds'])
    (args.output_dir/'fresh_chess_comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    text += ['', '## Fresh Chess128 full-cost comparison', '',
        'Same inputs/checkpoint and 1 warmup + 3 repeats. A/V is a separate run, not randomized interleaving. SAF3R rebuilds lazy Top-K inside every timed forward; original profile/topk128-per-frame is not a uniform10% budget.', '',
        '| Method | AUC30 (%) | Whole (s) | Global attention (s) |', '|---|---:|---:|---:|']
    for name,r in comparison.items():
        text.append(f"| {name} | {r['auc30']:.3f} | {r['seconds']:.3f} | {r['global_seconds']:.3f} |")
    proof = json.loads(Path('outputs/layer_shared_av/chunk_invariance_v3.json').read_text())
    algorithm_sha = hashlib.sha256(Path('saf3r/patch/layer_shared_av.py').read_bytes()).hexdigest()
    final_results = [json.loads((args.final/scene/'result.json').read_text()) for scene in SCENES]
    if not proof['equal'] or proof['algorithm_sha256'] != algorithm_sha:
        raise ValueError('Canonical arithmetic proof is stale or failed')
    if any(r['source_sha256']['saf3r/patch/layer_shared_av.py'] != algorithm_sha for r in final_results):
        raise ValueError('Final scene source does not match current policy')
    selected = json.loads(Path('outputs/layer_shared_av/joint_v3_chess_n128/selection.json').read_text())['selected']
    calibration = np.load(Path(selected['source']).parent/'poses.npz')['pose_enc']
    measured = np.load(args.final/'chess/poses.npz')['pose_enc']
    same_pose = bool(np.array_equal(calibration, measured))
    if not same_pose: raise ValueError('Chess calibration prediction did not reproduce')
    audit = dict(algorithm_sha256=algorithm_sha, final_scenes=SCENES,
        all_final_sources_match=True, repeated_pose_identical=all(r['measured_pose_identical'] for r in final_results),
        calibration_pose_reproduced=same_pose,
        chunk_proof=dict(equal=True, frames=128, layers=list(range(7)), dominance=.9,
            note='Scoped operator/trunk audit, not a complete24-layer pose check at selected tau.95'),
        final_config=selected)
    (args.output_dir/'validation_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    final = aggregates['selected_joint_v3']
    fire = next(r for r in scenes if r['method']=='selected_joint_v3' and r['scene']=='fire')
    text += ['', '## Interpretation', '',
        f"Regional A/V contrast was obtained, but the middle is only {100*final['middle_attention']:.2f}% Attention on the seven-scene mean, not100% and not Attention-majority in every middle layer or held-out scene.",
        f"Selected seven-scene AUC30 {final['auc30']:.3f}%, independent six-scene {final['heldout_six']['auc30']:.3f}%, Fire {fire['auc30']:.3f}%. This is an unacceptable accuracy loss, despite successful Chess calibration.",
        f"No layer entered Attention-only. Value-only accounts for {100*final['pure_value_layer_fraction']:.2f}% of scene-layers; the rest are mixed and incur full per-query routing/indexed-attention/validation costs.",
        'Current implementation is an opt-in experiment, NOT a usable acceleration replacement. Ratio fitting is not evidence of retained model quality or runtime speedup. All rejected arms are retained separately.', '']
    (args.output_dir/'report.md').write_text('\n'.join(text))
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=Path('outputs/layer_shared_av/reference_n128'))
    p.add_argument('--tuned', type=Path, default=Path('outputs/layer_shared_av/tuned_n128'))
    p.add_argument('--selected', type=Path, default=Path('outputs/layer_shared_av/selected_tau09_n128'))
    p.add_argument('--first-corrected', type=Path, default=Path('outputs/layer_shared_av/final_token_history_v3_n128'))
    p.add_argument('--final', type=Path, default=Path('outputs/layer_shared_av/selected_joint_v3_n128'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/layer_shared_av/report'))
    main(p.parse_args())
