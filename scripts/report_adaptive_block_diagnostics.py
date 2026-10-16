"""Independent query errors and stage costs, separate from formal timings."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def report(root,formal):
    rows=[]; stages=[]; verification=[]
    paths=sorted(root.glob('*/diagnostics_result.json')) or sorted(root.glob('*/result.json'))
    for path in paths:
        scene=path.parent.name
        result=json.loads(path.read_text())
        for method,r in result['methods'].items():
            if 'layers' not in r:
                continue
            baseline=formal/scene/f'{method}_poses.npz'
            if not baseline.exists():
                raise ValueError(f'No formal prediction for {scene}/{method}')
            suffix='_diagnostic' if path.name=='diagnostics_result.json' else ''
            current=np.load(path.parent/f'{method}{suffix}_poses.npz')['pose_enc']
            expected=np.load(baseline)['pose_enc']
            difference=float(np.max(np.abs(current-expected)))
            verification.append(dict(scene=scene,method=method,max_absolute_pose_difference=difference,
                                     exact_equal=bool(np.array_equal(current,expected))))
            if difference>1e-4:
                raise ValueError(f'Diagnostic changes predictions: {scene}/{method} max_abs={difference}')
            for l in r['layers']:
                val=l['independent_validation']
                for h,d in enumerate(l['decisions']):
                    rows.append(dict(scene=scene,method=method,layer_zero_based=l['layer_zero_based'],head=h,
                        calibration_nmse=d['audit_nmse'],validation_nmse=val['output_nmse'][h],
                        calibration_p90=d['audit_p90'],validation_p90=val['p90_query_nmse'][h],
                        calibration_passed=d['audit_passed'],validation_passed=(val['output_nmse'][h]<=.02 and val['p90_query_nmse'][h]<=.1),
                        actual_density=l['actual_density'][h],dense_fallback=d['dense_fallback']))
                stages.append(dict(scene=scene,method=method,layer_zero_based=l['layer_zero_based'],**l['stage_ms']))
    if not rows:
        raise ValueError('No diagnostics')
    write_csv(root/'diagnostic_independent_errors.csv',rows)
    # Some layers may use no dense fallback / no sparse path. Missing stages
    # mean zero, not a shorter denominator or an invented event duration.
    names=sorted(set(k for r in stages for k in r)-{'scene','method','layer_zero_based'})
    write_csv(root/'diagnostic_stages.csv',[{k:r.get(k,0) for k in ['scene','method','layer_zero_based',*names]} for r in stages])
    (root/'diagnostic_prediction_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines=['# Independent validation and diagnostic stage breakdown','',
        'These extra forwards are NOT formal speed measurements. Extra held-out output checks and diagnostic events are enabled.',
        'Their pose predictions must agree with the formal run (max absolute pose-encoding difference <=1e-4).',
        'Validation queries use offset 3/8 of each query block, distinct from routing offsets 1/4,3/4 and calibration offset 1/2.',
        'The reference is dense attention on the SAME current sparse-prefix QKV, not a Dense-prefix reference.',
        'Reference output uses BF16 Flash SDPA; reported kernel errors therefore include its small numerical differences.',
        'No validation output feeds back into masks or budget decisions. Sampled checks still do not certify all queries.','',
        '| Method | Mean calibration NMSE | Mean independent NMSE | P90 head/layer/scene NMSE | Independent pass % |',
        '|---|---:|---:|---:|---:|']
    aggregate={}
    for method in sorted(set(r['method'] for r in rows)):
        rr=[r for r in rows if r['method']==method]
        aggregate[method]=dict(mean_calibration_nmse=float(np.mean([r['calibration_nmse'] for r in rr])),
            mean_validation_nmse=float(np.mean([r['validation_nmse'] for r in rr])),
            p90_validation_nmse=float(np.quantile([r['validation_nmse'] for r in rr],.9)),
            independent_pass_fraction=float(np.mean([r['validation_passed'] for r in rr])))
        a=aggregate[method]
        lines.append(f"| {method} | {a['mean_calibration_nmse']:.6f} | {a['mean_validation_nmse']:.6f} | {a['p90_validation_nmse']:.6f} | {100*a['independent_pass_fraction']:.2f} |")
        ss=[r for r in stages if r['method']==method]
        scenes=len(set(r['scene'] for r in ss))
        a['stage_ms_sum24_macro_mean']={name:sum(r.get(name,0) for r in ss)/scenes for name in names}
    lines+=['','## 24-layer diagnostic stage totals','',
        'Per-scene sums, then scene macro mean (ms). Event intervals include intervening host dispatch/synchronization gaps.',
        'The separate independent validation checks are outside these stage intervals and outside formal timing.',
        'QKV/projection and bookkeeping outside marked intervals mean stage sums are not identical to attention-module totals.','',
        '| Stage | '+' | '.join(aggregate)+' |','|---|'+ '|'.join('---:' for _ in aggregate)+'|']
    for name in names:
        lines.append('| '+name+' | '+' | '.join(f"{a['stage_ms_sum24_macro_mean'][name]:.3f}" for a in aggregate.values())+' |')
    (root/'diagnostics_aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    (root/'diagnostics_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--formal',type=Path,required=True)
    args=p.parse_args();report(args.root,args.formal)
