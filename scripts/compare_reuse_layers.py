"""Compare full layer tensors against a verified historical reuse implementation.

No geometry truth is used. Instrumented forwards are not speed measurements.
Each source root runs in a fresh Python process to prevent mixed old/new imports.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--teacher-forced', action='store_true')
    a = p.parse_args()
    root = a.source_root.resolve()
    sys.path.insert(0, str(root))
    # Keep reporting implementation current even when loading historical attention.
    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    import yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig, install_reuse_residual_exploration
    from scripts.quality_metrics import quality_record, quality_markdown

    a.output_dir.mkdir(parents=True, exist_ok=False)
    reason = 'Full-token layer-output diagnostic, no pose/recon_unposed evaluation; no ground-truth tuning'
    quality = quality_record({}, reason)
    quality['speedup_unavailable'] = 'Instrumented diagnostic with tensor I/O; no paired warmed Dense timing'
    save(a.output_dir / 'quality.json', quality)
    (a.output_dir / 'quality.md').write_text(quality_markdown(quality))
    save(a.output_dir / 'status.json', {'stage': 'loading'})
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(a.config.read_text())
    paths = json.loads(a.input.read_text())['image_paths']
    cfg = Dict(yaml.safe_load((root / 'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.pop('patch_module')
    cfg.pop('sparse_config_path')
    cfg.ckpt_path = str(root / 'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model = build_model(cfg, 'cuda')
    if a.reference:
        images = torch.load(a.reference / 'images.pt', weights_only=True).cuda()
        provenance = json.loads((a.reference / 'input.json').read_text())
        assert provenance['image_paths'] == paths
    else:
        images = load_images(model, 'vggt', paths).cuda()
        torch.save(images.cpu(), a.output_dir / 'images.pt')
    checkpoint_sha = hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest()
    if a.reference:
        assert checkpoint_sha == provenance['checkpoint_sha256']
    per_frame = (images.shape[-2] // 14) * (images.shape[-1] // 14) + 5
    n = len(images) * per_frame
    token_id = torch.arange(n)
    subsets = {'all': torch.ones(n, dtype=torch.bool), 'camera': token_id % per_frame == 0,
               'register': (token_id % per_frame > 0) & (token_id % per_frame < 5),
               'patch': token_id % per_frame >= 5,
               'first_frame_patch': (token_id < per_frame) & (token_id % per_frame >= 5),
               'other_frame_patch': (token_id >= per_frame) & (token_id % per_frame >= 5)}
    if a.reference:
        subsets = {name: mask.cuda() for name, mask in subsets.items()}
    sha = {}
    for rel in ['saf3r/patch/reuse_residual_exploration.py', 'saf3r/triton/reuse_explore_av.py',
                'saf3r/utils/model_utils.py', 'saf3r/utils/data_utils.py',
                'saf3r/models/official_vggt/models/aggregator.py']:
        path = root / rel
        if path.exists():
            sha[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(a.output_dir / 'input.json', dict(image_paths=paths, shape=list(images.shape),
         checkpoint_sha256=checkpoint_sha, source_root=str(root), source_sha256=sha, config=config,
         reference=str(a.reference), teacher_forced=a.teacher_forced,
         gpu=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
         precision='BF16 autocast; TF32 disabled; seed 0',
         scope='All input frames and all tokens, 24 Global + 24 Frame attention layers; not a dataset quality evaluation',
         reconstruction_threshold=None, reconstruction_unavailable=reason,
         saf3r_profile='N/A: user Value/reuse method, not SAF3R Top-K'))
    state = install_reuse_residual_exploration(model, ReuseExploreConfig(**config))
    rows = []

    def metrics(x, y):
        # Chunked FP32 arithmetic, FP64 scalar reductions; all tokens participate.
        xx = yy = xy = dd = 0.
        cosine, relative = [], []
        zero_x = zero_y = 0
        for start in range(0, len(x), 8192):
            u, v = x[start:start+8192].float(), y[start:start+8192].float()
            ux, vy = u.square().sum(-1), v.square().sum(-1)
            dot, diff = (u*v).sum(-1), (u-v).square().sum(-1)
            xx += ux.double().sum().item(); yy += vy.double().sum().item()
            xy += dot.double().sum().item(); dd += diff.double().sum().item()
            valid = (ux > 0) & (vy > 0)
            cosine.append(dot[valid] / (ux[valid]*vy[valid]).sqrt())
            relative.append((diff[vy > 0]/vy[vy > 0]).sqrt())
            zero_x += int((ux == 0).sum()); zero_y += int((vy == 0).sum())
        c = torch.cat(cosine); r = torch.cat(relative)
        return dict(tokens=len(x), cosine=xy/(xx*yy)**.5 if xx*yy else None,
                    relative_l2=(dd/yy)**.5 if yy else None, norm_ratio=(xx/yy)**.5 if yy else None,
                    token_cosine_mean=c.mean().item() if c.numel() else None,
                    token_relative_l2_quantiles=torch.quantile(r, torch.tensor([.5,.9,.99],device=r.device)).tolist() if r.numel() else None,
                    zero_current_tokens=zero_x, zero_reference_tokens=zero_y)

    def capture(kind, layer, tensor):
        x = tensor.detach().reshape(-1, tensor.shape[-1])
        key = f'{kind}_{layer:02d}'
        if not a.reference:
            torch.save(x.cpu(), a.output_dir / (key + '.pt'))
        else:
            y = torch.load(a.reference / (key + '.pt'), weights_only=True, mmap=True).to(x.device)
            assert x.shape == y.shape, (key, x.shape, y.shape)
            row = dict(kind=kind, layer=layer, subsets={k: metrics(x[m], y[m]) for k,m in subsets.items()})
            rows.append(row)
            save(a.output_dir / 'layers.json', rows)
        print('LAYER', key, flush=True)

    for family, blocks in [('global', model.aggregator.global_blocks), ('frame', model.aggregator.frame_blocks)]:
        for layer, block in enumerate(blocks):
            if family == 'global':
                def block_pre(module, args, layer=layer):
                    if a.teacher_forced:
                        ref = torch.load(a.reference / f'global_block_input_{layer:02d}.pt', weights_only=True, mmap=True)
                        return (ref.reshape_as(args[0]).to(device=args[0].device), *args[1:])
                    capture('global_block_input', layer, args[0])
                block.register_forward_pre_hook(block_pre)
                original = block.attn.original
                original.proj.register_forward_pre_hook(lambda module,args,layer=layer: capture('global_pre_wo',layer,args[0]))
            block.attn.register_forward_hook(lambda module,args,out,layer=layer,family=family: capture(f'{family}_attention',layer,out))
    save(a.output_dir / 'status.json', {'stage': 'diagnostic_forward'})
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        raw = model(images)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    predictions = {key: raw[key].detach().cpu() for key in ('pose_enc','depth','depth_conf') if key in raw}
    torch.save(predictions, a.output_dir/'predictions.pt')
    def cpu(value):
        if torch.is_tensor(value): return value.detach().cpu().tolist()
        if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [cpu(v) for v in value]
        return value
    save(a.output_dir/'selector_layers.json',cpu(state.records))
    save(a.output_dir/'result.json',dict(quality=quality,instrumented_seconds=elapsed,
         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,config=config,
         reference=str(a.reference),teacher_forced=a.teacher_forced,layer_records=len(rows)))
    lines = ['All-token comparison; relative L2 denominator is the original reuse output norm.',
             'Teacher-forced mode replaces each Global block input by the original trajectory input.' if a.teacher_forced else 'Independent complete forward trajectories.', '',
             '| Kind | Layer (0-based) | Cosine | Relative L2 | Norm ratio |', '|---|---:|---:|---:|---:|']
    for row in rows:
        m=row['subsets']['all']
        lines.append(f"| {row['kind']} | {row['layer']} | {m['cosine']:.8f} | {m['relative_l2']:.8f} | {m['norm_ratio']:.8f} |")
    (a.output_dir/'layers.md').write_text('\n'.join(lines)+'\n')
    save(a.output_dir/'status.json',dict(stage='diagnostic_complete'))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        if '--output-dir' in sys.argv:
            folder=Path(sys.argv[sys.argv.index('--output-dir')+1])
            if folder.exists():
                from scripts.quality_metrics import quality_record,quality_markdown
                reason=f'Failed layer diagnostic: {type(exc).__name__}: {exc}'
                q=quality_record({},reason)
                q['speedup_unavailable']='Failed diagnostic; no paired warmed Dense timing'
                save(folder/'quality.json',q)
                (folder/'quality.md').write_text(quality_markdown(q))
                save(folder/'status.json',dict(stage='failed',error=reason,traceback=traceback.format_exc()))
        raise
