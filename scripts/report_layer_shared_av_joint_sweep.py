"""Select a correct-policy candidate using Chess, leaving six scenes held out."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path('outputs/layer_shared_av/joint_v3_chess_n128')


def main():
    rows = []
    for multiplier in (16, 32, 64):
        for tau in (.8, .9, .95):
            path = (Path('outputs/layer_shared_av/final_token_history_v3_n128/chess/result.json')
                if (multiplier, tau) == (32, .9) else ROOT/f'scale{multiplier}_tau{tau}'/'chess/result.json')
            r = json.loads(path.read_text())
            assert all(l.get('history_unit') == 'token_edges_v3_canonical_qk' for l in r['layers'])
            row = dict(logit_scale=multiplier, dominance=tau, sampling=.0025,
                fine_ratio=.15, coarse_size=128, query_chunk=1024, auc30=100*r['pose']['auc30'],
                source=str(path), repeats=len(r['samples']))
            for label, a, b in [('early', 3, 10), ('middle', 10, 17), ('late', 17, 24)]:
                row[label+'_attention'] = float(np.mean([l['attention_keys']/(l['attention_keys']+l['value_keys']) for l in r['layers'][a:b]]))
            row['profile_mae'] = (row['early_attention']+1-row['middle_attention']+row['late_attention'])/3
            row['dominance_profile'] = row['early_attention'] < .5 and row['middle_attention'] >= .5 and row['late_attention'] < .5
            rows.append(row)
    feasible = [r for r in rows if r['dominance_profile']]
    selected = max(feasible, key=lambda r: (r['auc30'], -r['profile_mae'])) if feasible else min(rows, key=lambda r: r['profile_mae'])
    with (ROOT/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (ROOT/'selection.json').write_text(json.dumps(dict(
        criterion='Exploratory: V-majority outside, A-majority middle on Chess, then highest Chess AUC30; no held-out scene selects parameters',
        selected=selected, trials=rows), indent=2)+'\n')
    text = ['# Correct v3 policy: joint Chess128 calibration', '',
        'All trials use token-edge history capacity and canonical coarse-QK tiles128. Eight trials are single cold forwards; (32,.9) reuses the completed three-repeat validation. Do not compare cold timings to warm medians.',
        'Exploratory selection: require Value-majority in3–9/17–23 and Attention-majority in10–16, then maximize Chess AUC30. This is a qualitative profile constraint, NOT a hardcoded runtime layer schedule. All other scenes remain out of this selection.', '',
        '| Scale | Tau | AUC30 (%) | A 3–9 (%) | A 10–16 (%) | A 17–23 (%) | Profile MAE |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['logit_scale']} | {r['dominance']} | {r['auc30']:.3f} | {100*r['early_attention']:.2f} | {100*r['middle_attention']:.2f} | {100*r['late_attention']:.2f} | {r['profile_mae']:.3f} |")
    text += ['', f'Selected calibration candidate: {json.dumps(selected)}', '',
        'Single-scene calibration is not evidence of generalization. Run the selected configuration on all seven scenes, report the independent six-scene macro mean separately, and retain all rejected configurations.', '']
    (ROOT/'report.md').write_text('\n'.join(text))
    print('\n'.join(text))


if __name__ == '__main__': main()
