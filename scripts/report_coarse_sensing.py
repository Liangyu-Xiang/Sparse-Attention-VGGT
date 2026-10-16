"""Report controlled coarse-sensing and same-input operator re-evaluation."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def report(root):
    runs = []
    for p in sorted(root.glob('*/summary.json')):
        status = json.loads((p.parent/'status.json').read_text())
        if status.get('state') != 'complete':
            continue
        runs.append((p.parent.name, json.loads(p.read_text())))
    if not runs:
        raise ValueError('No completed evaluations')
    sensing = [{'scene': scene, **r} for scene, run in runs for r in run['sensing']]
    operators = [{'scene': scene, **r} for scene, run in runs for r in run['operators']]
    sizes = sorted(set(r['coarse_size'] for r in sensing))
    modes = ['legacy_mean_qk', 'per_head']
    sensing_aggregate = []
    metric_keys = ['concentration','specificity','cs', 'hypothetical_cs_patch_budget']
    metric_keys += [f'{metric}_top{percentage}pct' for percentage in [10,20,30]
                   for metric in ['dense_patch_mass_fraction','dense_total_mass','dense_patch_topk_recall','token_coverage','mass_lift']]
    for size, mode, bias in itertools.product(sizes, modes, [True, False]):
        rr = [r for r in sensing if r['coarse_size'] == size and r['mode'] == mode and r['population_bias'] == bias]
        sensing_aggregate.append(dict(coarse_size=size, mode=mode, population_bias=bias,
            **{k: float(np.mean([r[k] for r in rr])) for k in metric_keys}))
    operator_aggregate, layer_rows = [], []
    for rho, selector in itertools.product(sorted(set(r['retention_target'] for r in operators)), ['shared_value','uniform','dense_topk_oracle']):
        rr = [r for r in operators if r['retention_target'] == rho and r['selector'] == selector]
        operator_aggregate.append(dict(retention=rho, selector=selector,
            mean_nmse=float(np.mean([r['output_nmse'] for r in rr])),
            median_nmse=float(np.median([r['output_nmse'] for r in rr])),
            p90_head_scene_layer_nmse=float(np.quantile([r['output_nmse'] for r in rr], .9)),
            retained_dense_mass=float(np.mean([r['retained_dense_mass'] for r in rr]))))
    for layer, rho, selector in itertools.product(range(24), [.1,.15,.2], ['shared_value','uniform','dense_topk_oracle']):
        rr = [r for r in operators if r['layer_zero_based'] == layer and r['retention_target'] == rho and r['selector'] == selector]
        if rr:
            layer_rows.append(dict(layer_zero_based=layer, retention=rho, selector=selector,
                mean_nmse=float(np.mean([r['output_nmse'] for r in rr])),
                p90_head_scene_nmse=float(np.quantile([r['output_nmse'] for r in rr], .9)),
                retained_dense_mass=float(np.mean([r['retained_dense_mass'] for r in rr]))))
    write_csv(root/'sensing_aggregate.csv', sensing_aggregate)
    write_csv(root/'operator_aggregate.csv', operator_aggregate)
    write_csv(root/'operator_by_layer.csv', layer_rows)
    write_csv(root/'sensing_per_scene_layer_head.csv', sensing)
    write_csv(root/'operator_per_scene_layer_head.csv', operators)
    lines = ['# Controlled coarse sensing and operator re-evaluation', '',
        f'Scenes: {", ".join(scene for scene, _ in runs)}.',
        f'Input frames: {runs[0][1]["metadata"]["frames"]}; query selection: {runs[0][1]["metadata"]["query_samples"]}.',
        'All 24 global layers are probed on the SAME unmodified dense-prefix QKV.',
        'The dense model forward and predictions are not changed by these probes.',
        'No new sparse end-to-end AUC or production speedup is claimed.', '',
        'Value fine partitions remain shared at 15% for sensing. For each layer,',
        'the 2x2 sensing comparisons share an identical fine/coarse partition.',
        'Coarse-size sweeps regroup the same fixed fine partition, without changing',
        'the source features or feeding the new grouping into subsequent layers.',
        'Layer-0 sensing is a counterfactual diagnostic; its actual Value-only bootstrap',
        'is not changed. Uniform initialization and all mandatory special keys remain.', '',
        'Reliability is measured on relevance probabilities WITHOUT a population prior',
        'in the corrected default helper. Biased probabilities are explicit ablations.',
        'CS is recorded as an uncalibrated statistic, not silently rescaled into a new',
        'budget rule. Q_h and K_h are paired before any optional averaging.', '',
        '## Matched 2x2 ablation (coarse size 128)', '',
        '| Sensing | Size bias | C | S | CS | Top20% true patch mass | Top20% patch Top-K recall | Token coverage | Mass lift |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sensing_aggregate:
        if r['coarse_size'] != 128:
            continue
        lines.append(f"| {r['mode']} | {r['population_bias']} | {r['concentration']:.6f} | {r['specificity']:.6f} | {r['cs']:.6f} | {100*r['dense_patch_mass_fraction_top20pct']:.3f}% | {100*r['dense_patch_topk_recall_top20pct']:.3f}% | {100*r['token_coverage_top20pct']:.3f}% | {r['mass_lift_top20pct']:.3f} |")
    lines += ['', '## Coarse-size sweep (matched-head relevance, no size prior)', '',
        '| Coarse size | C | S | Top10% true patch mass | Top20% true patch mass | Top20% Top-K recall | Token coverage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in sensing_aggregate:
        if r['mode'] != 'per_head' or r['population_bias']:
            continue
        lines.append(f"| {r['coarse_size']} | {r['concentration']:.6f} | {r['specificity']:.6f} | {100*r['dense_patch_mass_fraction_top10pct']:.3f}% | {100*r['dense_patch_mass_fraction_top20pct']:.3f}% | {100*r['dense_patch_topk_recall_top20pct']:.3f}% | {100*r['token_coverage_top20pct']:.3f}% |")
    lines += ['', '## Isolated operator budget sweep', '',
        'These selectors are DIAGNOSTIC CONTROLS, not a proposed deployable final method.',
        'A common shared 20%-fine partition is used across 10/15/20% budgets so that',
        'all budgets can select enough UNIQUE representatives. This differs from the',
        '15%-fine sensing partition and must not be conflated with the old cumulative',
        '15%-fine result. Same current input, FP32 mathematical logits/output, total',
        'budget includes mandatory specials. The dense-topk control knows exact dense',
        'scores and is not deployable cheaply; it is not an oracle for MINIMUM output NMSE.', '',
        '| Total retention | Selector | Mean NMSE | Median NMSE | P90 head/layer/scene NMSE | Retained dense mass |',
        '|---|---|---:|---:|---:|---:|']
    for r in operator_aggregate:
        lines.append(f"| {100*r['retention']:.0f}% | {r['selector']} | {r['mean_nmse']:.6f} | {r['median_nmse']:.6f} | {r['p90_head_scene_layer_nmse']:.6f} | {100*r['retained_dense_mass']:.3f}% |")
    lines += ['', '## Per-layer shared-Value sensitivity', '',
        '| Layer (0-based) | 10% mean NMSE | 15% mean NMSE | 20% mean NMSE | 10% dense-TopK NMSE |',
        '|---|---:|---:|---:|---:|']
    for layer in range(24):
        vv = [next(r['mean_nmse'] for r in layer_rows if r['layer_zero_based'] == layer and r['selector'] == 'shared_value' and r['retention'] == rho) for rho in [.1,.15,.2]]
        oracle = next(r['mean_nmse'] for r in layer_rows if r['layer_zero_based'] == layer and r['selector'] == 'dense_topk_oracle' and r['retention'] == .1)
        lines.append(f'| {layer} | '+' | '.join(f'{v:.6f}' for v in vv+[oracle])+' |')
    lines += ['', '## Metric corrections and limits', '',
        '- True patch mass: dense probability in selected groups / total dense patch probability.',
        '- Dense-total mass retains the special-token mass denominator and is separately saved.',
        '- Group Top-K recall is count overlap, NOT attention mass (the old review mixed these).',
        '- Token coverage and mass lift are saved: large groups can improve raw mass simply',
        '  by covering more keys. Removing size bias need not increase raw retained mass.',
        '- Head/query errors are not interchangeable; per-scene/layer/head CSVs are retained.',
        '- Coarse size 32 entails roughly 4x sensing QK work versus 128 at equal heads/queries.',
        '- Sampling 64 queries on 128-frame inputs is a probe, NOT a full-query certification.',
        '- No NMSE safety threshold, alpha calibration, adaptive total budget or new shared/head',
        '  mask policy has been silently introduced. The slow indexed production path is not fixed',
        '  by these sensing-only changes and requires separate kernel/selection redesign.',
        '- A different future mask changes later features; new final AUC requires a new model run.', '']
    (root/'report.md').write_text('\n'.join(lines))
    (root/'aggregate.json').write_text(json.dumps(dict(sensing=sensing_aggregate, operators=operator_aggregate), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    report(p.parse_args().root)
