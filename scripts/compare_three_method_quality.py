"""Input-checked VGGT / SAF3R / current History80 six-metric comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.quality_metrics import FIELDS, LABELS, quality_record, quality_cells

SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def report(root, current, topk):
    rows, missing = [], []
    fresh_current = (root / 'region').exists()
    methods = [('VGGT Dense', root / 'dense'), (f'SAF3R Top-{topk}', root / 'saf3r'),
               ('Ours History80', root / 'region' if fresh_current else current / 'region')]
    for scene in SCENES:
        reference = json.loads((current / 'region' / scene / 'result.json').read_text())
        for method, directory in methods:
            path = directory / scene / 'result.json'
            if not path.exists():
                missing.append(f'{method}/{scene}')
                continue
            result = json.loads(path.read_text())
            if result['input'] != reference['input']:
                raise ValueError(f'Input/checkpoint mismatch: {path}')
            if result['frames'] != 128 or result['dataset'] != '7scenes':
                raise ValueError(f'Protocol mismatch: {path}')
            if method.startswith('SAF3R') and result['protocol']['saf3r_topk'] != topk:
                raise ValueError(f'SAF3R Top-K mismatch: {path}')
            record = quality_record(result['metrics'])
            if not record['complete']:
                missing.append(f'{method}/{scene}/metrics')
            rows.append(dict(method=method, scene=scene, source=str(path), quality=record,
                seconds=result.get('median_seconds') if method!='Ours History80' or fresh_current else None,
                peak_gib=result.get('median_peak_gib') if method!='Ours History80' or fresh_current else None))
    dense_times={r['scene']:r['seconds'] for r in rows if r['method']=='VGGT Dense'}
    for row in rows:
        baseline=dense_times.get(row['scene'])
        row['speedup_vs_vggt']=baseline/row['seconds'] if baseline and row['seconds'] else None
    macro = []
    for method, _ in methods:
        selected = [r for r in rows if r['method'] == method and r['quality']['complete']]
        if len(selected) == len(SCENES):
            values = {key: mean(r['quality']['values'][key] for r in selected) for key in FIELDS}
            seconds=mean(r['seconds'] for r in selected) if all(r['seconds'] for r in selected) else None
            peak=mean(r['peak_gib'] for r in selected) if all(r['peak_gib'] for r in selected) else None
            macro.append(dict(method=method, quality=dict(values=values, complete=True, unavailable={}),
                              seconds=seconds,mean_scene_peak_gib=peak))
    dense_seconds=next((r['seconds'] for r in macro if r['method']=='VGGT Dense'),None)
    for row in macro:
        row['speedup_vs_vggt']=dense_seconds/row['seconds'] if dense_seconds and row['seconds'] else None
    def speed_cells(row):
        return ['N/A' if row.get('seconds') is None else f"{row['seconds']:.4f}",
                'N/A' if row.get('speedup_vs_vggt') is None else f"{row['speedup_vs_vggt']:.3f}x"]
    table = ['| Method | ' + ' | '.join(LABELS) + ' | Full forward s ↓ | Speedup vs VGGT ↑ |', '|---|' + '---:|' * 8]
    for row in macro:
        table.append('| ' + row['method'] + ' | ' + ' | '.join(quality_cells(row['quality'])+speed_cells(row)) + ' |')
    lines = ['# VGGT / SAF3R / History80: 7Scenes, 128 frames', '',
             f'Complete: {not missing}. Missing: {missing}.', '',
             'Exact same frame paths, indices, checkpoint and input shape are checked per scene. '
             'SAF3R uses the repository ETH3D-calibrated head profile and the explicitly labelled Top-K. '
             'The official default is Top-4; prior project speed comparisons used Top-128.', '',
             'Seven scene entries, 128 uniform frames each (not every original sequence/frame). '
             'Pose and unposed reconstruction follow the existing SAF3R DA3-Bench APIs, seed 0. '
             'ACC/Comp/CD are metres; CD=(ACC+Comp)/2, unsquared. F1 uses the dataset threshold. '
             'Macro averages give all seven scenes equal weight. '
             'When region/ exists, History80 is rerun in the current batch; otherwise old quality is shown '
             'without a speed claim. Timing uses one warmup and three complete forwards, median, '
             'including initialization, clustering, selection and history. Each scene is assigned one GPU '
             'for all methods. Speedup = Dense seconds / method seconds; macro speedup is the ratio '
             'of scene-mean seconds, not the mean of scene ratios. Machine timings are noisy; consult raw samples.', '',
             '## Seven-scene macro (only complete methods)', '', *table, '',
             '## Per-scene scores', '',
             '| Scene | Method | ' + ' | '.join(LABELS) + ' | Full forward s ↓ | Speedup vs VGGT ↑ |', '|---|---|' + '---:|' * 8]
    for row in rows:
        lines.append('| ' + row['scene'] + ' | ' + row['method'] + ' | ' +
                     ' | '.join(quality_cells(row['quality'])+speed_cells(row)) + ' |')
    write(root / 'comparison.json', dict(complete=not missing, missing=missing, macro=macro, per_scene=rows))
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    return not missing


def run(args):
    if args.report_only:
        report(args.output_dir, args.current, args.saf3r_topk)
        return
    if args.output_dir.exists() and not args.resume:
        raise ValueError('Choose a fresh output directory; never overwrite previous results')
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('Distinct idle GPUs required')
    for gpu in args.gpus:
        memory = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if memory > 512:
            raise RuntimeError(f'GPU {gpu} not idle')
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    files = ['scripts/evaluate_fused_shared_av.py', 'scripts/quality_metrics.py',
             'scripts/compare_three_method_quality.py', 'configs/evaluation/default_datasets.yaml',
             'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json']
    hashes = {}
    for name in files:
        data = (ROOT / name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        target = args.output_dir / ('source_snapshot_resume' if args.resume else 'source_snapshot') / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write(args.output_dir / ('manifest_resume.json' if args.resume else 'manifest.json'), dict(started_unix=time.time(), gpus=args.gpus,
        current=str(args.current), saf3r_topk=args.saf3r_topk, frames=128, seed=0,
        warmup=1, repeats=3, tasks=['pose', 'depth', 'recon_unposed'], source_sha256=hashes,
        include_current=args.include_current,
        purpose='Six-metric quality and full-forward speed comparison; no parameter tuning'))

    def worker(rank, gpu):
        for i in range(rank, len(SCENES), len(args.gpus)):
            scene = SCENES[i]
            order=(('dense','region','saf3r') if i%2==0 else ('saf3r','region','dense')) if args.include_current else (('dense','saf3r') if i%2==0 else ('saf3r','dense'))
            for mode in order:
                existing=args.output_dir/mode/scene/'result.json'
                if args.resume and existing.exists():
                    data=json.loads(existing.read_text())
                    expected=json.loads((args.current/'region'/scene/'result.json').read_text())
                    if data['input']!=expected['input'] or not quality_record(data['metrics'])['complete']:
                        raise ValueError(f'Invalid existing result: {existing}')
                    print('REUSE_COMPLETED',gpu,scene,mode,flush=True)
                    continue
                log = args.output_dir / 'logs' / f'{scene}_{mode}.log'
                log.parent.mkdir(exist_ok=True)
                command = [sys.executable, str(ROOT / 'scripts/evaluate_fused_shared_av.py'),
                    '--output-dir', str(args.output_dir), '--mode', mode, '--scene', scene,
                    '--frames', '128', '--warmup', '1', '--repeats', '3',
                    '--tasks', 'pose', 'depth', 'recon_unposed', '--saf3r-topk', str(args.saf3r_topk)]
                if mode=='region':command+=['--schedule',str(args.current/'schedule.json'),'--parent-only']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT),
                           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
                print('START', gpu, scene, mode, flush=True)
                with log.open('w') as stream:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                print('COMPLETE', gpu, scene, mode, flush=True)
    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, i, gpu) for i, gpu in enumerate(args.gpus)]
            for future in futures:
                future.result()
        if not report(args.output_dir, args.current, args.saf3r_topk):
            raise RuntimeError('Incomplete six-metric comparison')
        write(args.output_dir / 'status.json', dict(stage='complete', finished_unix=time.time()))
    except Exception as error:
        report(args.output_dir, args.current, args.saf3r_topk)
        write(args.output_dir / 'status.json', dict(stage='failed', error=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--current', type=Path, default=ROOT / 'outputs/parent_region_av/history80_7scenes128/history80')
    parser.add_argument('--gpus', type=int, nargs='+', default=[4, 5, 6])
    parser.add_argument('--saf3r-topk', type=int, default=128)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--resume', action='store_true',help='Preserve and validate completed runs; never overwrite raw results')
    parser.add_argument('--include-current', action='store_true',help='Rerun current method for contemporary full-forward speed ratios')
    run(parser.parse_args())
