"""Paired 128-frame 7Scenes validation: dense / SAF3R / all-layer hierarchy."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENES = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']


def summarize(args):
    rows = []
    for scene in SCENES:
        path = args.output_dir/scene/'result.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        for method, metrics in result['methods'].items():
            rows.append({'scene': scene, 'method': method, 'frames': result['input']['frames'],
                         'seconds': metrics['median_seconds'],
                         'selected_ms': metrics.get('median_selected_seconds', metrics.get('median_l3_9_seconds'))*1000,
                         'peak_gib': metrics['median_peak_gib'],
                         **{k: 100*v for k, v in metrics['pose'].items()}})
    if not rows:
        raise ValueError('No completed results')
    with (args.output_dir/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# 7Scenes, 128 frames: hierarchical Value + validated Attention', '',
        f'Paired inputs; natural global L{args.sparse_start}--L{args.sparse_end} (zero-based {args.sparse_start-1}--{args.sparse_end-1}) change.',
        'All other global and frame layers remain dense. One layer-shared selection/history,',
        'head-mean Q/K/V for selection; actual per-head attention outputs are not averaged.',
        'Hierarchy: 15% fine clusters, coarse ceil(N_patch/128), two local iterations,',
        f'Three coarse Lloyd steps; initialization={args.initialization}. Uniform bootstrap is Value-only.',
        'Final total edge density is 10% INCLUDING all mandatory special keys.',
        'SAF3R retains the original profile and is not matched to a 10% budget.', '',
        'SAF3R correspondence TopK=128 PER FRAME (lazy QK selector); this is',
        'effective in the full 24-layer run, unlike the legacy L3--L9-only run.', '',
        'Full inference includes the lightweight first-sparse-layer initialization.',
        'No L2 global PAM time is removed, because no PAM initialization is used.',
        'Timings: scene macro mean of per-scene medians, one warmup/three repeats',
        '(actual settings are recorded in each input.json). AUC: scene macro mean (%).', '',
        '| Method | Scenes | Selected global layers ms | Overall s | Peak GiB | AUC30 % | AUC15 % | AUC5 % | AUC3 % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    aggregate = {}
    for method in args.variants:
        rr = [r for r in rows if r['method'] == method]
        if not rr:
            continue
        mean = {k: float(np.mean([r[k] for r in rr])) for k in ['selected_ms', 'seconds', 'peak_gib', 'auc30', 'auc15', 'auc05', 'auc03']}
        aggregate[method] = {'scenes': len(rr), **mean}
        lines.append(f"| {method} | {len(rr)} | "+' | '.join(f'{mean[k]:.4f}' for k in mean)+' |')
    lines += ['', '## Per-scene comparison', '', '| Scene | Method | Selected global layers ms | Overall s | AUC30 % | AUC5 % |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['scene']} | {r['method']} | {r['selected_ms']:.3f} | {r['seconds']:.3f} | {r['auc30']:.3f} | {r['auc05']:.3f} |")
    layer_rows = []
    for scene in SCENES:
        path = args.output_dir/scene/'hierarchy_result.json'
        if path.exists():
            for layer in json.loads(path.read_text())['layers']:
                layer_rows.append({'scene': scene, **layer})
    if layer_rows:
        with (args.output_dir/'layers.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_rows[0]))
            writer.writeheader()
            writer.writerows(layer_rows)
        lines += ['', '## Layer-shared adaptive budgets (scene macro means)', '',
                  '| Layer (zero-based) | Alpha | A patch keys | V patch keys | Total keys | Connection density % |',
                  '|---|---:|---:|---:|---:|---:|']
        for layer in range(args.sparse_start, args.sparse_end+1):
            rr = [r for r in layer_rows if r['layer'] == layer]
            means = [np.mean([r[k] for r in rr]) for k in ['alpha', 'attention_budget', 'value_budget', 'total_keys', 'connection_retention']]
            means[-1] *= 100
            lines.append(f'| {layer-1} | '+' | '.join(f'{v:.6f}' for v in means)+' |')
        if all(r['attention_budget'] == 0 for r in layer_rows):
            lines += ['', 'IMPORTANT: all measured layers/scenes rounded K_A to zero.',
                'Under this head-mean-Q/K configuration, C*S is very small; actual',
                'inference degenerates to Value-only coverage with empty Attention history.',
                'The reuse/explore/exact-validation path is implemented and unit-tested,',
                'but these dataset results do NOT establish its accuracy or benefit.',
                'No minimum A budget or rescaling of alpha was silently introduced.',
                'The shared-K/V SDPA fast path is used because every query then has',
                'the identical Value-only support and no logits need history validation.']
    lines += ['', '## Interpretation and limitations', '',
        '- This validates pose AUC only; no GT depth or reconstruction claims.',
        '- This is an algorithmic/correctness prototype, not a fused production selector.',
        '  Sorting, integer allocation, scalar synchronizations and hierarchy regrouping',
        '  are included in timing. A low connection density does not guarantee acceleration.',
        '- Attention history is validated from the SAME computed exact sparse logits.',
        '  There is no second history QK evaluation and no old weights in state.',
        '- A source-description boundary case is handled explicitly: after exhausting',
        '  fine representatives, exploration uses unique deterministic member tokens',
        '  from the SAME coarse group, preserving token-capacity budgets and uniqueness.',
        '- Config: configs/evaluation/eval_hierarchical_value_vggt.yaml.',
        '- Method details: docs/experiments/hierarchical_value_attention.md.',
        '- Tests: tests/test_hierarchical_value_attention.py (including CUDA/reference',
        '  logits/output agreement, uniqueness/capacity/coverage/anchors, reset, validation).', '']
    (args.output_dir/'report.md').write_text('\n'.join(lines))
    (args.output_dir/'aggregate.json').write_text(json.dumps(aggregate, indent=2)+'\n')
    print(json.dumps(aggregate, indent=2), flush=True)


def worker(args):
    import torch
    sys.path.insert(0, str(ROOT/'evaluation'))
    from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous
    from scripts.overlay_vggt_query_attention import build_model
    from scripts.run_layer_value_local import install_saf3r_layer39
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from saf3r.patch.hierarchical_value_attention import install_hierarchical_value_attention
    from saf3r.patch.patch_vggt import update_args_per_forward

    directory = args.output_dir/args.scene
    directory.mkdir(parents=True, exist_ok=True)
    status = {'state': 'initializing', 'scene': args.scene, 'pid': os.getpid()}
    def save():
        temp = directory/'status.tmp.json'
        temp.write_text(json.dumps(status, indent=2)+'\n')
        temp.replace(directory/'status.json')
    save()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        reader = SevenScenes()
        reader.data_root = str(args.data_root)
        data = reader.get_data(args.scene)
        # Match the repository benchmark's deterministic uniform sampling.
        indices = np.linspace(0, len(data.image_files)-1, args.frames, dtype=int)
        if len(np.unique(indices)) != args.frames:
            raise ValueError('Not enough distinct images for requested frame count')
        paths = [data.image_files[i] for i in indices]
        gt = np.asarray(data.extrinsics)[indices]
        images = load_and_process_images_vggt(paths).cuda()
        metadata = {'scene': args.scene, 'frames': len(paths), 'image_paths': paths,
                    'sample_indices': indices.tolist(), 'image_shape': list(images.shape),
                    'checkpoint': str(args.checkpoint.resolve()), 'selection': 'one shared layer mask, head-mean Q/K/V; mean exact head logits for validation',
                    'sparse_layers': list(range(args.sparse_start, args.sparse_end+1)),
                    'sparse_layers_zero_based': list(range(args.sparse_start-1, args.sparse_end)),
                    'initialization': args.initialization, 'retention_total_including_specials': args.retention,
                    'saf3r_correspondence_topk_per_frame': 128,
                    'fine_ratio': .15, 'coarse_size': 128, 'warmup': args.warmup, 'repeats': args.repeats}
        (directory/'input.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('INPUT', images.shape, flush=True)
        model = build_model(args.checkpoint, 'cuda')
        originals = [b.attn for b in model.aggregator.global_blocks]
        original_forward = model.aggregator.forward
        methods = {}
        for name in args.variants:
            state = None
            if name == 'hierarchy':
                state = install_hierarchical_value_attention(model, retention=args.retention,
                    query_chunk=args.query_chunk, sparse_start=args.sparse_start,
                    sparse_end=args.sparse_end, initialization=args.initialization)
            elif name == 'saf3r':
                if (args.sparse_start, args.sparse_end) == (3, 9):
                    install_saf3r_layer39(model, topk=128)
                elif (args.sparse_start, args.sparse_end) == (1, 24):
                    from addict import Dict
                    from saf3r.patch.patch_vggt import patch_model
                    patch_model(model, Dict(sparse_config_path=str(ROOT/'configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json'),
                        patch_module=dict(type='headsparse', topk_mode='token', lazy_dino_topk=True, topk=128)))
                else:
                    raise ValueError('SAF3R paired baseline supports full 1--24 or legacy 3--9')
                update_args_per_forward(model, *images.shape[-2:])
            events, handles = [], []
            def before(module, inputs):
                module._benchmark_start = torch.cuda.Event(enable_timing=True)
                module._benchmark_start.record()
            def after(module, inputs, output):
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                events.append((module._benchmark_start, end))
            for layer in range(args.sparse_start-1, args.sparse_end):
                module = model.aggregator.global_blocks[layer].attn
                handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
            samples = []
            for iteration in range(args.warmup+args.repeats):
                status.update(state='inference', variant=name, iteration=iteration)
                save()
                events.clear()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(images)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                peak = torch.cuda.max_memory_allocated()/2**30
                module_seconds = sum(a.elapsed_time(b) for a, b in events)/1000
                if iteration >= args.warmup:
                    samples.append({'seconds': elapsed, 'selected_seconds': module_seconds, 'peak_gib': peak})
                print(name, iteration, elapsed, module_seconds, peak, flush=True)
                pose = pred['pose_enc'].float().cpu()
                if not torch.isfinite(pose).all():
                    raise ValueError('Nonfinite pose output')
                del pred
            ext, intr = pose_encoding_to_extri_intri(pose, images.shape[-2:])
            metrics = compute_pose(torch.from_numpy(as_homogeneous(ext[0].numpy())), torch.from_numpy(gt))
            result = {'samples': samples, 'pose': {k: float(v) for k, v in metrics.items()}}
            result.update({'median_'+k: float(np.median([r[k] for r in samples])) for k in samples[0]})
            if state is not None:
                result['layers'] = [{k: float(v) if torch.is_tensor(v) else v for k, v in r.items()}
                                    for r in state.records]
                assert [r['layer'] for r in result['layers']] == list(range(args.sparse_start, args.sparse_end+1))
                if args.initialization == 'uniform':
                    assert result['layers'][0]['uniform_value_only_start']
                    assert result['layers'][0]['attention_budget'] == 0
                result['history_shape'] = list(state.history.shape)
                assert state.history.ndim == 2
                state.reset_hook_handle.remove()
                delattr(model, 'hierarchical_value_state')
            for handle in handles:
                handle.remove()
            for block, original in zip(model.aggregator.global_blocks, originals):
                block.attn = original
            model.aggregator.forward = original_forward
            if hasattr(model, '_saf3r_dummy'):
                delattr(model, '_saf3r_dummy')
            np.savez_compressed(directory/f'{name}_poses.npz', pose_enc=pose.numpy(), extrinsics=ext.numpy(), intrinsics=intr.numpy())
            methods[name] = result
            (directory/f'{name}_result.json').write_text(json.dumps(result, indent=2)+'\n')
            print('METRICS', name, result['pose'], flush=True)
            del state
            torch.cuda.empty_cache()
        (directory/'result.json').write_text(json.dumps({'input': metadata, 'methods': methods,
            'timing_note': 'Full forward including hierarchy initialization; no PAM seed or diagnostics excluded. Median of warmed runs. Selected global layers modified; all frame attention remains dense.'}, indent=2)+'\n')
        status['state'] = 'complete'
    except Exception as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        save()


def batch(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending, active = list(args.scenes), {}
    status = {'state': 'running', 'scenes': {}, 'pid': os.getpid()}
    def save():
        name = 'status.json' if set(args.scenes) == set(SCENES) else 'batch_status_'+'_'.join(map(str, args.gpus))+'.json'
        (args.output_dir/name).write_text(json.dumps(status, indent=2)+'\n')
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            scene = pending.pop(0)
            directory = args.output_dir/scene
            directory.mkdir(exist_ok=True)
            if (directory/'result.json').exists():
                existing = json.loads((directory/'result.json').read_text())
                if (existing['input']['frames'] != args.frames or
                    existing['input']['retention_total_including_specials'] != args.retention or
                    existing['input']['sparse_layers'] != list(range(args.sparse_start, args.sparse_end+1)) or
                    existing['input'].get('initialization', 'lightweight') != args.initialization or
                    set(existing['methods']) != set(args.variants)):
                    raise ValueError('Existing run configuration mismatch')
                status['scenes'][scene] = {'state': 'complete'}
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), '--scene', scene,
                   '--output-dir', str(args.output_dir), '--frames', str(args.frames),
                   '--retention', str(args.retention), '--warmup', str(args.warmup),
                   '--repeats', str(args.repeats), '--query-chunk', str(args.query_chunk),
                   '--sparse-start', str(args.sparse_start), '--sparse-end', str(args.sparse_end),
                   '--initialization', args.initialization,
                   '--checkpoint', str(args.checkpoint), '--data-root', str(args.data_root),
                   '--variants', *args.variants]
            log = (directory/'run.log').open('a')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(ROOT),
                   'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = (scene, child, log)
            status['scenes'][scene] = {'state': 'running', 'gpu': gpu, 'pid': child.pid}
            print('START', scene, gpu, flush=True)
        for gpu, (scene, child, log) in list(active.items()):
            if child.poll() is None:
                continue
            log.close()
            status['scenes'][scene]['state'] = 'complete' if child.returncode == 0 else 'failed'
            print('END', scene, child.returncode, flush=True)
            del active[gpu]
        save()
        if active:
            time.sleep(5)
    status['state'] = 'complete' if all(v['state'] == 'complete' for v in status['scenes'].values()) else 'failed'
    save()
    if status['state'] == 'failed':
        raise RuntimeError('One or more scene workers failed; see per-scene logs')
    summarize(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', choices=SCENES)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--scenes', nargs='+', default=SCENES)
    p.add_argument('--gpus', nargs='+', default=list(range(7)))
    p.add_argument('--frames', type=int, default=128)
    p.add_argument('--retention', type=float, default=.10)
    p.add_argument('--sparse-start', type=int, default=1, help='One-based first global layer')
    p.add_argument('--sparse-end', type=int, default=24, help='One-based last global layer')
    p.add_argument('--initialization', choices=['uniform', 'lightweight'], default='uniform')
    p.add_argument('--query-chunk', type=int, default=128)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--variants', nargs='+', choices=['dense', 'saf3r', 'hierarchy'], default=['dense', 'saf3r', 'hierarchy'])
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    p.add_argument('--data-root', type=Path, default=ROOT/'workspace/benchmark_dataset/7scenes')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/hierarchical_value/7scenes_n128_r10_all24_uniform')
    args = p.parse_args()
    if args.frames < 2 or args.repeats < 1 or args.warmup < 0:
        p.error('Invalid frames/repeats/warmup')
    if not 1 <= args.sparse_start <= args.sparse_end <= 24:
        p.error('Invalid global layer range')
    summarize(args) if args.summarize else worker(args) if args.scene else batch(args)
