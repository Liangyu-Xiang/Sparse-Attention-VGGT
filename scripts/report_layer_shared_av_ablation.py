"""Chess-only sensitivity results; all trial times are cold, single forward."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.run_layer_shared_av_ablation import TRIALS


def main(args):
    rows = []
    for name, _ in TRIALS:
        path = args.root/name/'chess/result.json'
        r = json.loads(path.read_text())
        if r['config']['warmup'] != 0 or len(r['samples']) != 1:
            raise ValueError('Expected cold single-forward ablation')
        row = dict(trial=name, auc30=100*r['pose']['auc30'], auc05=100*r['pose']['auc05'],
            cold_seconds=r['median_seconds'], cold_global_seconds=r['median_global_seconds'],
            **{k:r['config'][k] for k in ('logit_scale', 'dominance', 'sampling', 'fine_ratio', 'coarse_size', 'query_chunk')})
        for label, a, b in [('early', 3, 10), ('middle', 10, 17), ('late', 17, 24)]:
            row[label+'_attention'] = float(np.mean([l['attention_keys']/(l['attention_keys']+l['value_keys']) for l in r['layers'][a:b]]))
        row['profile_mae'] = (row['early_attention']+1-row['middle_attention']+row['late_attention'])/3
        rows.append(row)
    with (args.root/'summary.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    text = ['# Chess128 sensitivity checks', '',
        'One cold forward per trial, not formal warm timing. Profile targets are Attention shares [0,1,0] for zero-based layers [3–9,10–16,17–23].',
        'Default trial configuration: multiplier32, tau.8, eta.25%, fine.15, coarse128, query chunk128.', '',
        '| Trial | AUC30 (%) | A 3–9 (%) | A 10–16 (%) | A 17–23 (%) | Profile MAE | Cold whole (s) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['trial']} | {r['auc30']:.3f} | {100*r['early_attention']:.2f} | {100*r['middle_attention']:.2f} | {100*r['late_attention']:.2f} | {r['profile_mae']:.3f} | {r['cold_seconds']:.2f} |")
    text += ['', 'The profile-only screen selected tau.8 but real sparse accuracy rejected it. Tau.9 was the only single-parameter trial above a Chess AUC30 floor of Dense-minus-3pp (85.466%).',
        'That accuracy gate is a post-screen selection using Chess only, not a preregistered threshold. Other scenes do not select the final threshold. Revalidate the selected policy on all seven scenes, with larger query chunks tested separately for numerical equality.',
        'Large accuracy swings across tau and sampling fractions indicate genuine fragility despite bitwise repeatability within a fixed configuration. Matching the desired regional means does not establish a robust method.', '']
    (args.root/'report.md').write_text('\n'.join(text))
    print('\n'.join(text))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/layer_shared_av/ablation_chess_n128'))
    main(p.parse_args())
