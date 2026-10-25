"""Canonical user-required six-metric quality record (stdlib only)."""
import math

FIELDS = ('auc03', 'auc30', 'acc', 'comp', 'cd', 'f1')
LABELS = ('AUC@3 % ↑', 'AUC@30 % ↑', 'ACC m ↓', 'Comp m ↓', 'CD m ↓', 'F1 % ↑')


def quality_record(metrics, missing_reason='Task not evaluated'):
    pose = metrics.get('pose', {})
    recon = metrics.get('recon_unposed', {})
    raw = dict(auc03=pose.get('auc03'), auc30=pose.get('auc30'),
               acc=recon.get('acc'), comp=recon.get('comp'),
               cd=recon.get('overall'), f1=recon.get('fscore'))
    values, reasons = {}, {}
    for key, value in raw.items():
        if value is None or not math.isfinite(float(value)):
            values[key] = None
            reasons[key] = missing_reason if value is None else 'Nonfinite evaluation result'
        else:
            values[key] = float(value)
    return dict(values=values, complete=not reasons, unavailable=reasons,
                speedup_vs_vggt=None,
                speedup_unavailable='Requires matched VGGT Dense full-forward timing; see comparison report',
                units=dict(auc03='fraction', auc30='fraction', acc='m', comp='m', cd='m', f1='fraction'),
                cd_definition='(ACC + Comp) / 2; unsquared nearest-neighbour distances')


def quality_cells(record):
    return ['N/A' if record['values'][k] is None else
            f"{record['values'][k] * (100 if k in ('auc03', 'auc30', 'f1') else 1):.4f}"
            for k in FIELDS]


def quality_markdown(record):
    lines = ['| ' + ' | '.join(LABELS) + ' |', '| ' + ' | '.join(['---:'] * 6) + ' |',
             '| ' + ' | '.join(quality_cells(record)) + ' |', '',
             'CD = (ACC + Comp) / 2, using unsquared distances.', '',
             'Speedup vs VGGT: N/A until matched Dense full-forward timing is available; see comparison report.']
    if record['unavailable']:
        lines += ['', 'Unavailable: ' + '; '.join(f'{k}: {v}' for k, v in record['unavailable'].items())]
    return '\n'.join(lines) + '\n'
