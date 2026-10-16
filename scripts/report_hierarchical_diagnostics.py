"""Build the four requested diagnostic tables; never mix with formal timings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def mean(values):
    vv = [v for v in values if v is not None]
    return float(np.mean(vv)) if vv else None


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main(root):
    runs = []
    for p in sorted((root/'full128').glob('*/summary.json')):
        status = json.loads((p.parent/'status.json').read_text())
        if status['state'] != 'complete':
            continue
        runs.append((p.parent, json.loads(p.read_text())))
    if not runs:
        raise ValueError('No completed full128 diagnostics')
    timings, composition, quality, accuracy, head_rows = [], [], [], [], []
    for layer in range(24):
        rows = [r for _, run in runs for r in run['layers'] if r['layer_zero_based'] == layer]
        stage_rows = []
        for directory, run in runs:
            with (directory/'runtime_breakdown.csv').open() as f:
                stage_rows.extend([r for r in csv.DictReader(f) if int(r['layer_zero_based']) == layer])
        keys = sorted(set(r['stage'] for r in stage_rows))
        for stage in keys:
            rr = [r for r in stage_rows if r['stage'] == stage]
            timings.append(dict(layer_zero_based=layer, stage=stage,
                milliseconds=mean([float(r['milliseconds']) for r in rr]),
                percent_of_pipeline=mean([float(r['percent_of_pipeline']) for r in rr]), scenes=len(rr)))
        qn = [run['metadata']['frames']*(run['metadata']['image_shape'][-2]//14* (run['metadata']['image_shape'][-1]//14)+5) for _, run in runs]
        total_patch = [n*(r['attention_budget']+r['value_budget']) for n, r in zip(qn, rows)]
        ratios = lambda key: mean([r[key]/denom for r, denom in zip(rows, total_patch)])
        composition.append(dict(layer_zero_based=layer,
            attention_budget=mean([r['attention_budget'] for r in rows]),
            value_budget=mean([r['value_budget'] for r in rows]),
            attention_fraction=mean([r['attention_budget']/(r['attention_budget']+r['value_budget']) for r in rows]),
            reuse_fraction=ratios('reused'), explore_fraction=ratios('explored'),
            value_only_fraction=ratios('value_only'), av_fraction=ratios('promoted_value')))
        quality.append(dict(layer_zero_based=layer,
            history_survival_rate=mean([r['history_survival_rate'] for r in rows]),
            exploration_precision=mean([r['exploration_precision'] for r in rows]),
            incoming_history=mean([r['incoming'] for r in rows]),
            reused=mean([r['reused'] for r in rows]), validated=mean([r['validated'] for r in rows]),
            rejected_attention_candidates=mean([r['rejected_attention_candidates'] for r in rows])))
        accuracy.append(dict(layer_zero_based=layer,
            output_nmse_mean=mean([r['output_nmse_mean'] for r in rows]),
            output_nmse_median_of_scene_head_medians=mean([r['output_nmse_median'] for r in rows]),
            output_nmse_p90_of_scene_heads=mean([r['output_nmse_p90'] for r in rows]),
            final_model_auc05=mean([run['final_pose']['auc05'] for _, run in runs]),
            final_model_auc30=mean([run['final_pose']['auc30'] for _, run in runs])))
    for directory, run in runs:
        for row in run['per_head']:
            head_rows.append({'scene': directory.name, **row})
    write_csv(root/'runtime_breakdown.csv', timings)
    write_csv(root/'mask_composition.csv', composition)
    write_csv(root/'history_quality.csv', quality)
    write_csv(root/'accuracy_breakdown.csv', accuracy)
    write_csv(root/'accuracy_per_scene_layer_head.csv', head_rows)
    raw_path = root/'chess_frames_0_20/chess/summary.json'
    oracle = json.loads(raw_path.read_text()) if raw_path.exists() else None
    lines = ['# Hierarchical Value diagnostics', '',
        f'{len(runs)} full 128-frame scene diagnostics; each is separate from formal timing.',
        'Formal AUC/runtime: ../report.md. Timings here synchronize GPU before/after',
        'each stage; instrumentation, dense reference, CPU exports and raw tensor I/O',
        'are NOT part of the formal benchmark. Fused QK/softmax/AV is a single measured',
        'stage; no invented standalone QK/AV measurements are reported.', '',
        'Selection remains layer-shared. Head dimension appears only in diagnostic',
        'output errors and dense oracles. Dense outputs use same current Q/K/V, not',
        'features from a different dense model trajectory. NMSE is squared error-energy',
        'ratio; relative L2 is saved separately. AUC is a FINAL MODEL metric, not a',
        'per-layer AUC. Rates with no samples are N/A.', '',
        '## Runtime Breakdown', '', '| Layer | Pipeline ms | Value ms | Coarse sensing ms | Mask/budget ms | Fused sparse ms | Validation ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for layer in range(24):
        r = {x['stage']: x['milliseconds'] for x in timings if x['layer_zero_based'] == layer}
        pipeline = mean([x['pipeline_wall_ms'] for _, run in runs for x in run['layers'] if x['layer_zero_based'] == layer])
        val = sum(r.get(k, 0) for k in ['mean_value', 'value_initialization', 'hierarchy_mapping', 'fine_reassignment', 'representative_update', 'coarse_grouping'])
        coarse = sum(r.get(k, 0) for k in ['pooled_key', 'coarse_qk', 'coarse_softmax', 'coarse_statistics'])
        mask = sum(r.get(k, 0) for k in ['budget_allocation', 'history_lookup_filter_dedup', 'exploration', 'value_fill', 'mask_assemble'])
        lines.append(f'| {layer} | {pipeline:.2f} | {val:.2f} | {coarse:.2f} | {mask:.2f} | {r.get("sparse_qk_softmax_av_fused", 0):.2f} | {r.get("validation", 0):.2f} |')
    lines += ['', '## Mask Composition', '', '| Layer | K_A | K_V | Reuse % | Explore % | V-only % | AV % |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in composition:
        lines.append(f"| {r['layer_zero_based']} | {r['attention_budget']:.2f} | {r['value_budget']:.2f} | "+' | '.join(f'{100*r[k]:.5f}' for k in ['reuse_fraction','explore_fraction','value_only_fraction','av_fraction'])+' |')
    lines += ['', '## History Quality (full128)', '', '| Layer | History survival | Exploration precision | Incoming history edges | Validated edges |', '|---|---:|---:|---:|---:|']
    fmt = lambda x: 'N/A' if x is None else f'{x:.6f}'
    for r in quality:
        lines.append(f"| {r['layer_zero_based']} | {fmt(r['history_survival_rate'])} | {fmt(r['exploration_precision'])} | {r['incoming_history']:.0f} | {r['validated']:.0f} |")
    lines += ['', '## Accuracy Breakdown (same-input operator error, full128)', '', '| Layer | Mean head NMSE | Mean head-median NMSE | Mean head-p90 NMSE |', '|---|---:|---:|---:|']
    for r in accuracy:
        lines.append(f"| {r['layer_zero_based']} | {r['output_nmse_mean']:.6f} | {r['output_nmse_median_of_scene_head_medians']:.6f} | {r['output_nmse_p90_of_scene_heads']:.6f} |")
    if oracle:
        lines += ['', '## Small raw diagnostic: Chess seq-01 frames 0,20', '',
            'Raw .pt files include coarse attention, group budgets, fine/coarse labels,',
            'representatives/sizes, previous/validated indices, final shared supports,',
            'selection origin and post-validation provenance, selected FP32-oracle logits',
            '(stored FP16), dense Top-K indices/mass/recall, dense/sparse outputs/errors.',
            'Layer 0 skips coarse attention by design (stored as null, not fabricated).',
            'Dense Top-K oracle uses FP32 logits; dense output reference follows BF16',
            'SDPA precision. These small-input oracle values are NOT 128-frame estimates.', '',
            '| Layer | Dense Top-K recall | Retained dense mass | Coarse top10% recall | Coarse top20% recall | Coarse top30% recall |',
            '|---|---:|---:|---:|---:|---:|']
        for r in oracle['layers']:
            lines.append(f"| {r['layer_zero_based']} | "+' | '.join(fmt(r.get(k)) for k in ['dense_topk_recall','retained_dense_mass','coarse_group_recall_top10pct','coarse_group_recall_top20pct','coarse_group_recall_top30pct'])+' |')
    lines += ['', '## Reuse and limitations', '',
        'Raw records permit offline inspection/reconstruction of selection and changes',
        'to C/S-to-budget mapping without re-extracting those same captured features.',
        'They do NOT provide exact logits for keys outside the saved support (except',
        'saved dense Top-K indices), and cannot predict new end-to-end model AUC after',
        'changing an earlier mask without rerunning the model.',
        'Code 0=V, 1=A-reuse, 2=A-explore, 3=V promoted to AV, 4=mandatory special.',
        'selection_origin preserves the pre-validation cause; provenance marks V->AV.', '']
    (root/'report.md').write_text('\n'.join(lines))
    (root/'manifest.json').write_text(json.dumps(dict(scenes=[p.name for p, _ in runs],
        full128_frames=128, layers_zero_based=list(range(24)),
        raw_diagnostic='chess_frames_0_20/chess', reports=['runtime_breakdown.csv','mask_composition.csv','history_quality.csv','accuracy_breakdown.csv','accuracy_per_scene_layer_head.csv']), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
