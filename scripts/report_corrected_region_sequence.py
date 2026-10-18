"""Report the paired single-sequence128-frame corrected Attention trial."""
import argparse
import csv
import json
import os
from pathlib import Path
import time


def run(root,scene):
    dense=json.loads((root/'dense'/scene/'result.json').read_text())
    sparse=json.loads((root/'corrected'/scene/'result.json').read_text())
    for key in ['image_paths','checkpoint','frame_indices','image_shape','gpu']:
        assert dense['input'][key]==sparse['input'][key]
    assert sparse['config']['frames']==128 and sparse['config']['variant']=='B'
    assert sparse['config']['retention']==.1 and sparse['config']['transfer_ratio']==.5
    assert dense['source_sha256']==sparse['source_sha256']
    assert sparse['no_dense_oracle_in_middle']
    assert [r['layer'] for r in sparse['layers']]==list(range(9,17))
    assert all(r['mode']=='sparse' for r in sparse['layers'] if r['layer']>=10)
    metrics={key:dict(dense=100*dense['pose'][key],corrected=100*sparse['pose'][key],
        change_pp=100*(sparse['pose'][key]-dense['pose'][key])) for key in ['auc05','auc15','auc30']}
    out=root/'report';out.mkdir(exist_ok=True)
    layer_rows=[]
    for r in sparse['layers']:
        if r['layer']<10:continue
        row=dict(r,seconds=sparse['samples'][0]['layer_seconds'][str(r['layer'])])
        layer_rows.append(row)
    with (out/'layers10_16.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(layer_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(layer_rows)
    summary=dict(scene=scene,frames=128,metrics=metrics,
        dense_seconds=dense['median_seconds'],corrected_seconds=sparse['median_seconds'],
        dense_middle_seconds=dense['median_middle_seconds'],corrected_middle_seconds=sparse['median_middle_seconds'],
        dense_peak_gib=dense['median_peak_gib'],corrected_peak_gib=sparse['median_peak_gib'],
        timing_protocol=sparse['timing_protocol'],source_sha256=sparse['source_sha256'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# One128-frame sequence: Dense / corrected Attention / Dense','',
        '7Scenes Chess seq-01,128uniformly sampled frames. Same image paths, frame indices, '
        'checkpoint, image dimensions, hardware type and implementation snapshots verified. '
        'Global0–9 Dense;10–16 corrected B,10%final budget including specials,50%history; '
        '17–23 Dense. Frame/local attention unchanged. No middle-layer Dense oracle.','',
        '| Pose metric | Dense % | Corrected % | Change(pp) |','|---|---:|---:|---:|']
    for key,value in metrics.items():lines.append(f"| {key} | {value['dense']:.3f} | {value['corrected']:.3f} | {value['change_pp']:+.3f} |")
    lines+=['','## Timing is diagnostic, NOT steady-state speedup','',
        f"Protocol: {sparse['timing_protocol']}. Both runs use0warmup/1forward. "
        'Compilation, Python scheduling, hierarchy maintenance, Dense layer9seed computation '
        'and per-layer synchronization are included. No oracle/capture comparison is done in the middle. '
        'One measured sample does not establish repeated-pose determinism.','',
        '| Range | Dense seconds | Corrected seconds |','|---|---:|---:|',
        f"| Full forward | {dense['median_seconds']:.3f} | {sparse['median_seconds']:.3f} |",
        f"| Global middle10–16 | {dense['median_middle_seconds']:.3f} | {sparse['median_middle_seconds']:.3f} |",
        f"| Layer9(includes seed only for corrected) | {dense['median_seed_layer9_seconds']:.3f} | {sparse['median_seed_layer9_seconds']:.3f} |",
        f"| Peak allocated GPU GiB | {dense['median_peak_gib']:.3f} | {sparse['median_peak_gib']:.3f} |",'',
        'The middle attention masks are not exported at128frames. Disabling capture now avoids '
        'unnecessary full-support CPU copies; selection is unchanged and23focused tests passed. '
        'Query chunk32limits pooled-key memory, rather than changing retention or routing policy.','',
        '## Middle layer diagnostics','',
        '| Layer | Seconds | Final density % | Inherited edges/query | New edges/query | Local terminal QK/query |',
        '|---|---:|---:|---:|---:|---:|']
    for row in layer_rows:
        lines.append(f"| {row['layer']} | {row['seconds']:.3f} | {100*row['density']:.4f} | "
            f"{row['history_retained']:.1f} | {row['explored']:.1f} | {row['terminal_member_qk']:.1f} |")
    lines+=['','This is ONE sequence, not the7Scenes macro average. Two-frame same-input '
        'output error being close to a TopK attention oracle does not guarantee similar end-to-end accuracy.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--scene',default='chess');p.add_argument('--wait',action='store_true')
    p.add_argument('--corrected-pid',type=int)
    args=p.parse_args()
    if args.wait:
        started=time.monotonic()
        while not all((args.directory/mode/args.scene/'result.json').exists() for mode in ['dense','corrected']):
            if args.corrected_pid:
                try:os.kill(args.corrected_pid,0)
                except ProcessLookupError:raise RuntimeError('Corrected process exited without a complete paired result')
            if time.monotonic()-started>4*3600:raise TimeoutError('Sequence trial exceeded4hours')
            time.sleep(20)
    run(args.directory,args.scene)
