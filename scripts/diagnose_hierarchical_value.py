"""Independent diagnostics, never included in formal performance measurements."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args):
    import torch
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.patch.hierarchical_value_attention import install_hierarchical_value_attention
    from saf3r.patch.value_diagnostics import ValueDiagnostics
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sys.path.insert(0, str(ROOT/'evaluation'))
    from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous
    directory = args.output_dir/args.scene
    directory.mkdir(parents=True, exist_ok=True)
    status = dict(state='initializing', scene=args.scene, pid=os.getpid())
    def save():
        (directory/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    save()
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        reader = SevenScenes()
        reader.data_root = str(ROOT/'workspace/benchmark_dataset/7scenes')
        data = reader.get_data(args.scene)
        positions = args.frame_indices or np.linspace(0, len(data.image_files)-1, args.frames, dtype=int).tolist()
        paths = [data.image_files[i] for i in positions]
        gt = np.asarray(data.extrinsics)[positions]
        images = load_and_process_images_vggt(paths).cuda()
        collector = ValueDiagnostics(directory, raw=args.raw, dense_output=True, dense_oracle=args.dense_oracle)
        model = build_model(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt', 'cuda')
        state = install_hierarchical_value_attention(model, retention=.10, diagnostics=collector)
        status['state'] = 'diagnosing'
        save()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(images)
        pose = prediction['pose_enc'].float().cpu()
        ext, intr = pose_encoding_to_extri_intri(pose, images.shape[-2:])
        metrics = {k: float(v) for k, v in compute_pose(torch.from_numpy(as_homogeneous(ext[0].numpy())), torch.from_numpy(gt)).items()}
        metadata = dict(scene=args.scene, frames=len(paths), image_paths=paths, frame_indices=positions,
            image_shape=list(images.shape), sparse_layers_zero_based=list(range(24)),
            retention_including_specials=.10, initialization='uniform_value_only_layer_zero',
            raw=args.raw, dense_oracle=args.dense_oracle, algorithm='layer_shared_head_mean_QKV',
            scope='24 global layers; frame/local attention unchanged')
        assert len(collector.rows) == 24
        collector.save(metadata, metrics)
        np.savez_compressed(directory/'poses.npz', pose_enc=pose.numpy(), extrinsics=ext.numpy(), intrinsics=intr.numpy())
        # Verify diagnostics have not changed predictions on the same input.
        if args.verify_formal:
            ref = args.verify_formal/args.scene
            official = json.loads((ref/'input.json').read_text())
            if paths != official['image_paths']:
                raise ValueError('Diagnostic/formal image order mismatch')
            expected = torch.from_numpy(np.load(ref/'hierarchy_poses.npz')['pose_enc'])
            torch.testing.assert_close(pose, expected, rtol=2e-3, atol=2e-3)
            status['formal_prediction_verified'] = True
        status['state'] = 'complete'
        print(args.scene, metrics, flush=True)
    except Exception as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        save()


def batch(args):
    from scripts.benchmark_hierarchical_value import SCENES
    pending, active = list(SCENES), {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            scene = pending.pop(0)
            directory = args.output_dir/scene
            directory.mkdir(exist_ok=True)
            if (directory/'status.json').exists() and json.loads((directory/'status.json').read_text()).get('state') == 'complete':
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), '--scene', scene,
                   '--frames', str(args.frames), '--output-dir', str(args.output_dir)]
            if args.verify_formal:
                cmd += ['--verify-formal', str(args.verify_formal)]
            log = (directory/'run.log').open('a')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(ROOT),
                   'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = scene, p, log
            print('START', scene, gpu, flush=True)
        for gpu, (scene, p, log) in list(active.items()):
            if p.poll() is None:
                continue
            log.close()
            if p.returncode:
                failures.append(scene)
            print('END', scene, p.returncode, flush=True)
            del active[gpu]
        if active:
            time.sleep(5)
    if failures:
        raise RuntimeError(f'Diagnostic workers failed: {failures}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default=None)
    p.add_argument('--gpus', nargs='+', default=list(range(7)))
    p.add_argument('--frames', type=int, default=128)
    p.add_argument('--frame-indices', nargs='+', type=int)
    p.add_argument('--raw', action='store_true')
    p.add_argument('--dense-oracle', action='store_true')
    p.add_argument('--verify-formal', type=Path)
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/hierarchical_value/7scenes_n128_r10_all24_uniform/diagnostics/full128')
    args = p.parse_args()
    if args.dense_oracle and not args.raw:
        p.error('--dense-oracle requires --raw')
    if args.raw and not args.scene:
        p.error('Raw diagnostics require an explicit small --scene/--frame-indices input')
    run(args) if args.scene else batch(args)
