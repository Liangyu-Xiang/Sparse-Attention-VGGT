"""Revised fixed-10% layer-shared A/V policy; independent 7Scenes runs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args):
    import torch
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.patch.layer_shared_av import install_layer_shared_av, finish_records
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sys.path.insert(0, str(ROOT/'evaluation'))
    from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous
    directory = args.output_dir/args.scene
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'result.json').exists():
        raise ValueError('Refusing to overwrite completed results')
    source_files = ['saf3r/patch/layer_shared_av.py', 'saf3r/triton/hierarchy_attention.py',
        'saf3r/patch/hierarchical_value_attention.py', 'saf3r/patch/value_clustering_gpu.py',
        'saf3r/patch/value_evolving_attention.py', 'scripts/evaluate_layer_shared_av.py']
    hashes = {}
    for name in source_files:
        source = ROOT/name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        dest = directory/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    config = vars(args).copy(); config['output_dir'] = str(config['output_dir'])
    (directory/'config.json').write_text(json.dumps(dict(config=config, source_sha256=hashes), indent=2))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reader = SevenScenes(); reader.data_root = str(ROOT/'workspace/benchmark_dataset/7scenes')
    data = reader.get_data(args.scene)
    positions = np.linspace(0, len(data.image_files)-1, args.frames, dtype=int)
    paths = [data.image_files[i] for i in positions]
    images = load_and_process_images_vggt(paths).cuda()
    model = build_model(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt', 'cuda')
    state = install_layer_shared_av(model, sampling=args.sampling,
        fine_ratio=args.fine_ratio, coarse_size=args.coarse_size,
        dominance=None if args.dominance == 0 else args.dominance,
        logit_scale=args.logit_scale, query_chunk=args.query_chunk,
        cluster_iterations=args.cluster_iterations, sensing_only=args.sensing_only,
        capture_logits=args.capture_logits)
    events = []
    def before(module, inputs):
        module._av_event = torch.cuda.Event(enable_timing=True); module._av_event.record()
    def after(module, inputs, output):
        end = torch.cuda.Event(enable_timing=True); end.record(); events.append((module._av_event, end))
    for block in model.aggregator.global_blocks:
        block.attn.register_forward_pre_hook(before)
        block.attn.register_forward_hook(after)
    samples, fingerprints = [], []
    for iteration in range(args.warmup+args.repeats):
        (directory/'status.json').write_text(json.dumps(dict(state='inference', iteration=iteration, pid=os.getpid())))
        events.clear(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(images)
        torch.cuda.synchronize()
        sample = dict(seconds=time.perf_counter()-started,
            global_seconds=sum(a.elapsed_time(b) for a, b in events)/1000,
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
        pose = pred['pose_enc'].float().cpu(); del pred
        if not torch.isfinite(pose).all(): raise ValueError('nonfinite pose')
        if iteration >= args.warmup:
            samples.append(sample)
            fingerprints.append(hashlib.sha256(pose.numpy().tobytes()).hexdigest())
        print(args.scene, iteration, sample, flush=True)
    ext, intr = pose_encoding_to_extri_intri(pose, images.shape[-2:])
    metrics = compute_pose(torch.from_numpy(as_homogeneous(ext[0].numpy())),
        torch.from_numpy(np.asarray(data.extrinsics)[positions]))
    layers = finish_records(state)
    result = dict(scene=args.scene, config=config, source_sha256=hashes, samples=samples,
        measured_pose_fingerprints=fingerprints, measured_pose_identical=len(set(fingerprints)) == 1,
        pose={k: float(v) for k, v in metrics.items()}, layers=layers,
        input=dict(image_paths=paths, frame_indices=positions.tolist(), image_shape=list(images.shape),
            checkpoint=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'),
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__),
        **{'median_'+k: float(np.median([r[k] for r in samples])) for k in samples[0]})
    if state.sampled_logits:
        torch.save(state.sampled_logits, directory/'sampled_logits.pt')
    np.savez_compressed(directory/'poses.npz', pose_enc=pose.numpy(), extrinsics=ext.numpy(), intrinsics=intr.numpy())
    (directory/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (directory/'status.json').write_text(json.dumps(dict(state='complete')))
    print('METRICS', result['pose'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='chess')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frames', type=int, default=128)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--fine-ratio', type=float, default=.15)
    p.add_argument('--coarse-size', type=int, default=128)
    p.add_argument('--cluster-iterations', type=int, default=2)
    p.add_argument('--sampling', type=float, default=.01)
    p.add_argument('--dominance', type=float, default=.9, help='0 disables snapping')
    p.add_argument('--logit-scale', type=float, default=1., help='Explicit optional global coarse-logit multiplier; reference=1')
    p.add_argument('--query-chunk', type=int, default=128)
    p.add_argument('--sensing-only', action='store_true', help='OFFLINE diagnostic using dense outputs, not the sparse policy')
    p.add_argument('--capture-logits', action='store_true')
    run(p.parse_args())
