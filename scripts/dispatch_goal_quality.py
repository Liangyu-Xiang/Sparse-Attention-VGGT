import concurrent.futures
import json
import os
import queue
import subprocess
import sys
from pathlib import Path
from threading import Lock

from evaluate_scannet500_beta import CONFIG, FIELDS, LABELS, ROOT, quality_cells, quality_record, record, save, sha

METHOD = 'fast_subgroup'
CHECKPOINT = Path('/tmp/vggt-checkpoints/model_tracker_fixed_e20.pt')
EXPECTED_SHA = 'b08a43baa2db1aad9718e71e098831b8ad32f6f6826c802e9eb714aa34420969'
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
scenes = (ROOT / 'configs/evaluation/fastvggt_scannet_50.txt').read_text().split()[:20]
assert len(set(scenes)) == 20
assert sha(CHECKPOINT) == EXPECTED_SHA
jobs = queue.Queue()
lock = Lock()
running = {}
rows = {}
historical_roots = sorted(Path('/tmp').glob('vggt-longseq*-results-20261106'))
unpaired = 'No contemporaneous paired Dense timing. Unchanged controls were not rerun.'
terminals = {'complete', 'quality_incomplete', 'failed'}

def path(scene):
    return out / 'scannet50' / '500' / scene / METHOD

def matched_dense(row):
    for folder in historical_roots:
        source = folder / 'scannet50' / '500' / row['scene'] / 'dense' / 'result.json'
        if not source.exists():
            continue
        dense = json.loads(source.read_text())
        if dense['quality']['complete'] and all(row.get(k) == dense.get(k) and row.get(k) is not None for k in ('input', 'protocol')):
            return dense, str(source)
    return None, None

def show(number):
    return 'N/A' if number is None else f'{number:.3f}'

def report():
    complete = [r for r in rows.values() if r['quality']['complete']]
    timed = [r for r in rows.values() if r.get('timing')]
    matched = []
    lines = [
        '# ScanNet50 · up to 500 frames · fast_subgroup', '',
        'Unified-Beta execution candidate, CUDA Welford enabled. FastVGGT-style BF16 VGGT with four retained layers. '
        'One warmup and three synchronized full forwards, including all in-forward method overhead. '
        'AUC and F1 are percentages. Distances are in metres. CD is the mean of unsquared ACC and Comp.', '',
        unpaired, '',
        '| Scene | Method | Frames | ' + ' | '.join(LABELS) + ' | Warm s | Peak alloc GiB | Dense speedup | Status |',
        '|---|---|---:|' + '---:|' * 9 + '---|',
    ]
    details = []
    for scene in scenes:
        row = rows[scene]
        timing = row.get('timing') or {}
        peaks = [s['peak_allocated_gib'] for s in timing.get('samples', [])]
        lines.append('| ' + ' | '.join([scene, METHOD, str(row.get('actual_frames', 'N/A')), *quality_cells(row['quality']), show(timing.get('median_seconds')), show(max(peaks, default=row.get('peak_allocated_gib'))), 'N/A', row['status']]) + ' |')
        dense, source = matched_dense(row)
        if dense:
            matched.append((row, dense))
            details.append(dict(scene=scene, source=source, quality=dense['quality'],
                                delta={k: row['quality']['values'][k] - dense['quality']['values'][k] for k in FIELDS} if row['quality']['complete'] else None))
            lines.append('| ' + ' | '.join([scene, 'historical Dense', str(dense['actual_frames']), *quality_cells(dense['quality']), 'N/A', 'N/A', '1.00x', 'reused quality']) + ' |')
    macro = quality_record({}, f'All 20 selected scenes required. {len(complete)} complete.')
    macro['speedup_unavailable'] = unpaired
    if len(complete) == 20:
        macro.update(complete=True, unavailable={}, values={k: sum(r['quality']['values'][k] for r in complete) / 20 for k in FIELDS})
    macro_time = sum(r['timing']['median_seconds'] for r in timed) / 20 if len(timed) == 20 else None
    lines.append('| ' + ' | '.join(['First-20 equal-scene macro', METHOD, '≤500', *quality_cells(macro), show(macro_time), 'N/A', 'N/A', f'{len(complete)}/20']) + ' |')
    dense_macro = None
    delta_macro = None
    if len(matched) == 20:
        dense_macro = quality_record({})
        dense_macro.update(complete=True, unavailable={}, speedup_vs_vggt=1.0, speedup_unavailable=None,
                           values={k: sum(d['quality']['values'][k] for _, d in matched) / 20 for k in FIELDS})
        lines.append('| ' + ' | '.join(['First-20 equal-scene macro', 'historical Dense', '≤500', *quality_cells(dense_macro), 'N/A', 'N/A', '1.00x', 'reused quality']) + ' |')
        if macro['complete']:
            delta_macro = {k: macro['values'][k] - dense_macro['values'][k] for k in FIELDS}
    lines.extend(['', f'Protocol-matched historical Dense quality: {len(matched)}/20.', ''])
    tmp = out / 'report.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.replace(out / 'report.md')
    save(out / 'summary.json', dict(quality=macro, warm_macro_seconds=macro_time, dense_quality=dense_macro, delta_vs_dense=delta_macro, historical_dense=details, per_scene=list(rows.values())))
    save(out / 'progress.json', dict(complete=len(complete), timed=len(timed), running=running, queued=jobs.qsize(), terminal=sum(r['status'] in terminals for r in rows.values())))

for scene in scenes:
    result = path(scene) / 'result.json'
    if result.exists() and (row := json.loads(result.read_text()))['status'] in terminals:
        rows[scene] = row
        continue
    row = dict(scene=scene, method=METHOD, status='queued', quality=quality_record({}, 'Evaluation queued'))
    row['quality']['speedup_unavailable'] = unpaired
    record(path(scene), row)
    rows[scene] = row
    jobs.put(scene)
report()

def worker(gpu):
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
               LD_LIBRARY_PATH='/usr/lib/x86_64-linux-gnu', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    while True:
        try:
            scene = jobs.get_nowait()
        except queue.Empty:
            return
        cmd = [sys.executable, str(ROOT / 'scripts/profile_scannet500_beta_local.py'), '--worker',
               '--dataset', 'scannet50', '--frames', '500', '--scene', scene, '--method', METHOD,
               '--output', str(out), '--checkpoint-sha256', EXPECTED_SHA]
        with (path(scene) / 'run.log').open('w') as log:
            proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with lock:
                running[gpu] = dict(scene=scene, pid=proc.pid)
                report()
            code = proc.wait()
        row = json.loads((path(scene) / 'result.json').read_text())
        if row['status'] not in terminals:
            row.update(status='failed', quality=quality_record({}, f'Worker exited {code}. See run.log.'))
        row['quality']['speedup_unavailable'] = unpaired
        record(path(scene), row)
        with lock:
            running.pop(gpu)
            rows[scene] = row
            report()

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(worker, range(4)))
