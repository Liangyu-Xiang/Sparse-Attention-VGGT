"""Controlled sensing ablations and isolated sparse-operator tests on dense inputs.

No cumulative sparse forward, invented alpha calibration or speedup claims.
All sensing variants use the SAME current dense-prefix QKV/fine partition.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from saf3r.patch.coarse_sensing import coarse_attention
from saf3r.patch.hierarchical_value_attention import HierarchicalValueState, lightweight_clusters, token_layout


def masked_output(logits, values, indices, chunk=16):
    """FP32 same-input operator oracle, with bounded query/key gathers."""
    if indices.ndim == 1:
        indices = indices[None, None].expand(len(logits), logits.shape[1], -1)
    outputs = []
    for start in range(0, logits.shape[1], chunk):
        ix = indices[:, start:start+chunk]
        score = logits[:, start:start+chunk].gather(-1, ix)
        vv = values[torch.arange(len(values), device=values.device)[:, None, None], ix]
        outputs.append((score.softmax(-1)[..., None]*vv).sum(-2))
    return torch.cat(outputs, 1)


def localization_metrics(p, dense_group_mass, dense_patch_mass, topk_groups, valid_patch, population, fraction):
    """True dense MASS and Top-K recall are distinct, separately named metrics."""
    p = p.expand(len(dense_group_mass), -1, -1)
    ng = max(1, math.ceil(p.shape[-1]*fraction))
    chosen = p.topk(ng, -1).indices
    mask = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, chosen, True)
    retained = (dense_group_mass*mask).sum(-1)
    mass_fraction = retained/dense_patch_mass.clamp_min(1e-12)
    topk_hit = mask.gather(-1, topk_groups) & valid_patch
    recall = topk_hit.sum(-1)/valid_patch.sum(-1).clamp_min(1)
    coverage = (mask*population[None, None]).sum(-1)/population.sum()
    return dict(dense_patch_mass_fraction=mass_fraction.mean(-1),
                dense_total_mass=retained.mean(-1), dense_patch_topk_recall=recall.mean(-1),
                token_coverage=coverage.mean(-1), mass_lift=(mass_fraction/coverage.clamp_min(1e-12)).mean(-1))


class ControlledEvaluation:
    def __init__(self, args, frames, height, width):
        self.args, self.output = args, args.output_dir/args.scene
        self.output.mkdir(parents=True, exist_ok=True)
        self.frames, self.height, self.width = frames, height, width
        self.sensing = HierarchicalValueState(fine_ratio=.15)
        # The 10/15/20% operator sweep fixes ONE common 20%-fine partition;
        # otherwise a 15%-fine partition cannot supply 20% unique reps.
        self.operator = HierarchicalValueState(fine_ratio=.20)
        for state in (self.sensing, self.operator):
            state.reset(frames, height, width)
        self.rows, self.operator_rows = [], []
        self.handles = []

    def attach(self, model):
        for layer, block in enumerate(model.aggregator.global_blocks):
            def hook(module, inputs, kwargs, layer=layer):
                self.evaluate(module, inputs[0], kwargs.get('pos'), layer)
            self.handles.append(block.attn.register_forward_pre_hook(hook, with_kwargs=True))

    def evaluate(self, original, x, pos, layer):
        started = time.perf_counter()
        b, n, c = x.shape
        qkv = original.qkv(x).reshape(b, n, 3, original.num_heads, original.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        patches, specials = token_layout(self.frames, self.height*self.width, 5, x.device)
        with torch.autocast(device_type=x.device.type, enabled=False):
            mean_v = v[0, :, patches].float().mean(0)
            hierarchy = self.sensing.update(mean_v, layer+1)
            operator_hierarchy = self.operator.update(mean_v, layer+1)
        q, k = original.q_norm(q), original.k_norm(k)
        if original.rope is not None:
            q, k = original.rope(q, pos), original.rope(k, pos)
        with torch.autocast(device_type=x.device.type, enabled=False):
            all_q, k, v = q[0].float(), k[0].float(), v[0].float()
            sample = torch.arange(n, device=x.device) if not self.args.query_samples else (
                torch.linspace(0, n-1, min(n, self.args.query_samples), device=x.device).long())
            q = all_q[:, sample]
            dense_logits = q@k.transpose(1, 2)*original.scale
            dense_p = dense_logits.softmax(-1)
            dense_output = dense_p@v
            top_budget = max(1, round(.1*n))
            dense_top = dense_logits.topk(top_budget, -1).indices
            inverse = torch.full((n,), -1, device=x.device, dtype=torch.long)
            inverse[patches] = torch.arange(len(patches), device=x.device)
            top_patch = inverse[dense_top]
            valid_patch = top_patch >= 0
            patch_p = dense_p[..., patches]
            patch_mass = patch_p.sum(-1)
            z, reps, ref_mapping, _ = hierarchy
            raw = dict(layer_zero_based=layer, query_indices=sample.cpu(),
                       fine_cluster_id=z.int().cpu(), fine_representative_index=reps.int().cpu(), variants={})
            if self.args.save_tensors:
                raw.update(queries=q.cpu(), keys=k.cpu(), values=v.cpu(), scale=original.scale,
                           dense_topk_indices=dense_top.int().cpu())
            for coarse_size in self.args.coarse_sizes:
                nc = min(len(reps), math.ceil(len(patches)/coarse_size))
                mapping = ref_mapping if coarse_size == 128 else lightweight_clusters(mean_v[reps], nc, 3)
                groups = mapping[z]
                # Group mass is exact for these dense oracle weights, NOT the
                # earlier diagnostic's coarse-group dense Top-K recall.
                mass = patch_p@F.one_hot(groups, nc).float()
                top_groups = groups[top_patch.clamp_min(0)]
                for mode in ('legacy_mean_qk', 'per_head'):
                    for bias in (True, False):
                        p, statistics, population = coarse_attention(q, k[:, patches], groups, nc,
                            mode=mode, population_bias=bias, scale=original.scale)
                        name = f'c{coarse_size}_{mode}_{"size_bias" if bias else "no_bias"}'
                        metric_sets = {fraction: localization_metrics(p, mass, patch_mass, top_groups,
                            valid_patch, population, fraction) for fraction in (.1, .2, .3)}
                        for head in range(len(q)):
                            sh = 0 if mode == 'legacy_mean_qk' else head
                            record = dict(layer_zero_based=layer, head=head, coarse_size=coarse_size,
                                mode=mode, population_bias=bias, queries=len(sample),
                                concentration=float(statistics['concentration'][sh]),
                                specificity=float(statistics['specificity'][sh]), cs=float(statistics['cs'][sh]),
                                # Record OLD map only for diagnosis, not adopted
                                # as a new calibrated budget policy.
                                hypothetical_cs_patch_budget=round(float(statistics['cs'][sh])*(top_budget-len(specials))))
                            for fraction, metrics in metric_sets.items():
                                for key, value in metrics.items():
                                    record[f'{key}_top{round(fraction*100)}pct'] = float(value[head])
                            self.rows.append(record)
                        if self.args.save_tensors:
                            raw['variants'][name] = dict(probabilities=p.half().cpu(),
                                reliability={key: value.cpu() for key, value in statistics.items()},
                                fine_to_coarse=mapping.int().cpu(), population=population.int().cpu())
            # Local operator sensitivity at fixed input and common fine ratio.
            oz, oreps, omap, occupancy = operator_hierarchy
            maximum = max(round(r*n)-len(specials) for r in self.args.retentions)
            ranked_patch = dense_logits[..., patches].topk(maximum, -1).indices
            for retention in self.args.retentions:
                total = round(retention*n)
                budget = total-len(specials)
                if budget < 1 or budget > len(oreps):
                    raise ValueError('Operator budget must fit the common fine partition')
                if layer == 0:
                    priority = torch.arange(budget, device=x.device)*len(oreps)//budget
                else:
                    priority = occupancy.argsort(descending=True, stable=True)[:budget]
                value_indices = torch.cat((patches[oreps[priority]], specials))
                uniform_indices = torch.cat((patches[torch.arange(budget, device=x.device)*len(patches)//budget], specials))
                oracle_indices = torch.cat((patches[ranked_patch[..., :budget]],
                    specials[None, None].expand(len(q), len(sample), -1)), -1)
                for selector, ix in [('shared_value', value_indices), ('uniform', uniform_indices),
                                     ('dense_topk_oracle', oracle_indices)]:
                    output = masked_output(dense_logits, v, ix)
                    nmse = (output-dense_output).square().sum((1, 2))/dense_output.square().sum((1, 2)).clamp_min(1e-12)
                    if ix.ndim == 1:
                        retained_mass = dense_p[..., ix].sum(-1).mean(-1)
                    else:
                        retained_mass = dense_p.gather(-1, ix).sum(-1).mean(-1)
                    for head in range(len(q)):
                        self.operator_rows.append(dict(layer_zero_based=layer, head=head,
                            retention_target=retention, connection_retention=total/n, selector=selector,
                            fine_ratio=.20, queries=len(sample), output_nmse=float(nmse[head]),
                            relative_l2=float(nmse[head].sqrt()), retained_dense_mass=float(retained_mass[head])))
            if self.args.save_tensors:
                raw['operator_fine_partition'] = dict(labels=oz.int().cpu(), reps=oreps.int().cpu(), sizes=occupancy.cpu())
                torch.save(raw, self.output/f'layer_{layer:02d}.pt')
        print(self.args.scene, 'layer', layer, 'diagnostic seconds', round(time.perf_counter()-started, 3), flush=True)

    def save(self, metadata, dense_pose):
        for name, rows in [('sensing.csv', self.rows), ('operators.csv', self.operator_rows)]:
            with (self.output/name).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
                writer.writeheader(); writer.writerows(rows)
        (self.output/'summary.json').write_text(json.dumps(dict(metadata=metadata, dense_pose=dense_pose,
            sensing=self.rows, operators=self.operator_rows), indent=2)+'\n')


def main(args):
    sys.path.insert(0, str(ROOT/'evaluation'))
    from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    directory = args.output_dir/args.scene
    directory.mkdir(parents=True, exist_ok=True)
    status = dict(state='initializing', scene=args.scene)
    def save_status():
        (directory/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    save_status()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        reader = SevenScenes()
        reader.data_root = str(ROOT/'workspace/benchmark_dataset/7scenes')
        data = reader.get_data(args.scene)
        indices = args.frame_indices or np.linspace(0, len(data.image_files)-1, args.frames, dtype=int).tolist()
        images = load_and_process_images_vggt([data.image_files[i] for i in indices]).cuda()
        model = build_model(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt', 'cuda')
        experiment = ControlledEvaluation(args, len(indices), images.shape[-2]//14, images.shape[-1]//14)
        experiment.attach(model)
        status['state'] = 'evaluating'
        save_status()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(images)
        pose = pred['pose_enc'].float().cpu()
        ext, intr = pose_encoding_to_extri_intri(pose, images.shape[-2:])
        metrics = {key: float(value) for key, value in compute_pose(torch.from_numpy(as_homogeneous(ext[0].numpy())),
            torch.from_numpy(np.asarray(data.extrinsics)[indices])).items()}
        meta = dict(scene=args.scene, frames=len(indices), frame_indices=indices,
            query_samples=args.query_samples or 'all', coarse_sizes=args.coarse_sizes,
            sensing_fine_ratio=.15, operator_fine_ratio=.20, retentions=args.retentions,
            feature_source='unmodified dense model prefix; no cumulative sparse errors',
            note='No new sparse end-to-end AUC/speed measured; CS is uncalibrated; mass != topk recall')
        meta['layer0_sensing'] = 'Counterfactual measurement only; deployed bootstrap remains Value-only'
        assert len(experiment.rows) == 24*16*len(args.coarse_sizes)*4
        assert len(experiment.operator_rows) == 24*16*len(args.retentions)*3
        experiment.save(meta, metrics)
        np.savez_compressed(directory/'dense_poses.npz', pose_enc=pose.numpy(), extrinsics=ext.numpy(), intrinsics=intr.numpy())
        if len(indices) == 128 and not args.frame_indices:
            ref = ROOT/'outputs/hierarchical_value/7scenes_n128_r10_all24_uniform'/args.scene/'dense_poses.npz'
            if ref.is_file():
                expected = torch.from_numpy(np.load(ref)['pose_enc'])
                torch.testing.assert_close(pose, expected, rtol=1e-4, atol=1e-4)
                status['dense_prefix_verified_against_formal'] = True
        status['state'] = 'complete'
    except Exception as exc:
        status.update(state='failed', error=repr(exc))
        raise
    finally:
        save_status()


def batch(args):
    from scripts.benchmark_hierarchical_value import SCENES
    pending, active, failures = list(SCENES), {}, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    while pending or active:
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            scene = pending.pop(0)
            directory = args.output_dir/scene
            directory.mkdir(exist_ok=True)
            if (directory/'status.json').exists() and json.loads((directory/'status.json').read_text()).get('state') == 'complete':
                meta = json.loads((directory/'summary.json').read_text())['metadata']
                if (meta['frames'] != args.frames or meta['query_samples'] != (args.query_samples or 'all')
                    or meta['coarse_sizes'] != args.coarse_sizes or meta['retentions'] != args.retentions):
                    raise ValueError(f'Existing controlled evaluation settings mismatch: {scene}')
                continue
            command = [sys.executable, str(Path(__file__).resolve()), '--scene', scene,
                '--frames', str(args.frames), '--query-samples', str(args.query_samples),
                '--output-dir', str(args.output_dir), '--coarse-sizes', *map(str, args.coarse_sizes),
                '--retentions', *map(str, args.retentions)]
            log = (directory/'run.log').open('a')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(ROOT),
                'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1',
                'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = scene, process, log
            print('START', scene, gpu, flush=True)
        for gpu, (scene, process, log) in list(active.items()):
            if process.poll() is None:
                continue
            log.close()
            if process.returncode:
                failures.append(scene)
            print('END', scene, process.returncode, flush=True)
            del active[gpu]
        if active:
            time.sleep(5)
    if failures:
        raise RuntimeError(f'Failed controlled evaluations: {failures}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='chess')
    p.add_argument('--gpus', nargs='+')
    p.add_argument('--frames', type=int, default=2)
    p.add_argument('--frame-indices', nargs='+', type=int)
    p.add_argument('--query-samples', type=int, default=0)
    p.add_argument('--coarse-sizes', nargs='+', type=int, default=[32, 64, 128])
    p.add_argument('--retentions', nargs='+', type=float, default=[.10, .15, .20])
    p.add_argument('--save-tensors', action='store_true')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/coarse_sensing_review/chess_0_20')
    args = p.parse_args()
    if args.frames < 2 or args.query_samples < 0 or min(args.coarse_sizes) < 1 or any(not 0 < r <= .20 for r in args.retentions):
        p.error('Invalid frame/retention settings')
    if args.gpus and (args.frame_indices or args.save_tensors):
        p.error('Batch full-frame probes use summaries, not small raw frame-index runs')
    batch(args) if args.gpus else main(args)
