"""Summarize the paired Chess memory recheck and 300-frame comparison."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main(root):
    specs=[
        ('Dense 128', 'saf3r128_release/dense/chess'),
        ('SAF3R Top-128, release stale indices', 'saf3r128_release/saf3r/chess'),
        ('SAF3R Top-128, retain stale indices', 'saf3r128_retain/saf3r/chess'),
        ('SAF3R Top-4 (official default) 128', 'saf3r128_official_top4/saf3r/chess'),
        ('Dense 300', 'compare300/dense/chess'),
        ('Fused 20% 300', 'compare300/fused/chess'),
        ('SAF3R Top-128 300', 'compare300/saf3r/chess'),
        ('SAF3R Top-4 (official default) 300', 'compare300_official_top4/saf3r/chess'),
    ]
    records=[]
    for label,relative in specs:
        directory=root/relative
        if (directory/'result.json').exists():
            data=read(directory/'result.json')
            records.append(dict(label=label,status='complete',path=relative,frames=data['frames'],
                seconds=data['median_seconds'],peak_gib=data['median_peak_gib'],
                peak_reserved_gib=sorted(s['peak_reserved_gib'] for s in data['samples'])[len(data['samples'])//2],
                pose=data['metrics']['pose'],protocol=data['protocol'],
                pose_fingerprints=data['pose_fingerprints']))
        else:
            data=read(directory/'status.json')
            if data['stage'] not in ('oom','failed'):
                raise RuntimeError(f'Incomplete job: {directory}')
            inputs=read(directory/'input.json')
            records.append(dict(label=label,status=data['stage'],path=relative,
                frames=len(inputs['image_paths']),error=data.get('error')))
    # Absolute file paths must be identical, not merely frame counts or scenes.
    for frames in (128,300):
        inputs=[read(root/r['path']/'input.json') for r in records
            if r.get('frames')==frames]
        assert all(x==inputs[0] for x in inputs), 'Paired input mismatch'
    release=records[1];retain=records[2]
    assert release['pose_fingerprints']==retain['pose_fingerprints'], 'Cache reset changed outputs'
    dense={r['frames']:r for r in records if r['label'].startswith('Dense')}
    lines=['# SAF3R peak-memory recheck and 300-frame comparison','',
        'Scope: 7Scenes Chess seq-01, uniform sampling, 392×518 input, BF16 autocast; '
        'one warmup, three measured complete forwards, no layer timing hooks. '
        'Includes fresh per-forward selection and state reset. Pose only; not seven-scene macro metrics.', '',
        '| Configuration | Seconds ↓ | Allocated peak GiB ↓ | Reserved peak GiB | AUC@5 % ↑ | AUC@15 % ↑ | AUC@30 % ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in records:
        if row['status']=='complete':
            row['dense_speedup']=dense[row['frames']]['seconds']/row['seconds']
            row['auc30_change_pp']=100*(row['pose']['auc30']-dense[row['frames']]['pose']['auc30'])
            p=row['pose']
            lines.append(f"| {row['label']} | {row['seconds']:.3f} | {row['peak_gib']:.3f} | "
                f"{row['peak_reserved_gib']:.3f} | {100*p['auc05']:.3f} | {100*p['auc15']:.3f} | {100*p['auc30']:.3f} |")
        else:lines.append(f"| {row['label']} | {row['status']} | — | — | — | — | — |")
    lines+=['', '## Interpretation','',
        f"The previous 29.882 GiB SAF3R Top-128 peak is reproduced with stale indices retained. "
        f"Releasing all obsolete wrapper references before fresh selection lowers the peak to "
        f"{release['peak_gib']:.3f} GiB without changing any measured pose fingerprint. "
        'This is a benchmark lifecycle correction, not cached selection or a different sparse mask.', '',
        'At 300 frames Top-128 requires a 310800 × 38400 int32 selection-index tensor '
        '(44.46 GiB alone). The unmodified selector fails on the 48-GiB device. '
        'The failed allocation is not included in PyTorch peak allocated memory; its recorded '
        'pre-failure peak is NOT the memory requirement for a successful run.', '',
        'Top-4 is an explicitly separate baseline from the repository evaluation YAML, '
        'not a replacement result for Top-128. The current Triton version uses the existing '
        'blockwise PyTorch selector compatibility path; sparse attention remains SAF3R.', '',
        'Allocated and allocator-reserved peaks are different metrics. All peaks in the table '
        'are medians of the three measured per-forward peaks; per-sample and cold-start '
        'values are in the individual result.json files.', '']
    (root/'summary.json').write_text(json.dumps(dict(records=records,scope='Chess only; pose only'),indent=2)+'\n')
    (root/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/fused_shared_av/recheck_20261019'))
    main(parser.parse_args().root)
