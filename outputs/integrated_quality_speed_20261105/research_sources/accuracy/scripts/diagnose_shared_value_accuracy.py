"""Same-input, same-mask causal attention audit; never consumes geometry truth.

Dense attention is computed only for sampled queries on the sparse trajectory.
Counterfactual weights do not change the actual forward. No quality/speed claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import numpy as np
import torch
from scripts.quality_metrics import quality_record, quality_markdown


def save(path, value):
    def convert(item):
        if torch.is_tensor(item):return item.detach().cpu().tolist()
        if isinstance(item,np.generic):return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=convert) + '\n')


def vector_similarity(sparse,dense):
    """Rows are query vectors. Preserve sums for exact pooled layer metrics."""
    error=(sparse-dense).square().sum()
    reference=dense.square().sum()
    cosine=torch.nn.functional.cosine_similarity(sparse,dense,dim=-1,eps=1e-12)
    return dict(queries=len(dense),error_sum_sq=float(error),dense_sum_sq=float(reference),
        cosine_sum=float(cosine.sum()),cosine_mean=float(cosine.mean()),
        relative_l2=float((error/reference.clamp_min(1e-20)).sqrt()),
        zero_dense_vectors=int((dense.square().sum(-1)<=1e-20).sum()))


class Audit:
    def __init__(self, state, groups, out, group_offsets=None, camera_ids=()):
        self.state, self.target_groups, self.out = state, groups, out
        self.group_offsets=group_offsets
        self.camera_ids=set(camera_ids)
        self.rows, self.supports, self.priors = [], {}, {}
        self.start = 0

    def prepare(self, original, module, x, pos):
        q, k, v = original(module, x, pos)
        self.q, self.k, self.v = q.to(v.dtype), k.to(v.dtype), v
        self.scale, self.module, self.start = module.scale, module, 0
        self.priors={}
        return q, k, v

    def prior(self, original, state, start, old_ids, obs, return_uniform=False):
        answer = original(state, start, old_ids, obs, return_uniform=return_uniform)
        prior, pi = answer[:2]
        for group in self.target_groups:
            local = group - start
            if 0 <= local < len(old_ids):
                self.priors[group] = prior[local].clone()
        return answer

    @torch.autocast('cuda', enabled=False)
    def attention(self, original, q, k, v, ids, specials, scale, bias=None, **kwargs):
        result, evidence = original(q, k, v, ids, specials, scale, bias, **kwargs)
        s = self.state
        layer = s.previous_layer + 1
        for group in self.target_groups:
            local = group - self.start
            if not 0 <= local < len(ids):
                continue
            offsets=self.group_offsets[group] if self.group_offsets is not None else [4,12,20,28]
            positions = (local * 32 + torch.tensor(offsets, device=q.device)).clamp_max(q.shape[1]-1)
            logits = (q[:, positions].float() @ k.float().transpose(-1, -2)) * scale
            dense_p = logits.softmax(-1)
            dense_out = dense_p @ v.float()
            selected = ids[local].long()
            selected = selected[selected >= 0]
            patch_ix = torch.searchsorted(s.patches, selected)
            support = torch.cat((selected, specials.long()))
            support_logits = logits[:, :, support]
            sv = v[:, support].float()
            labels = s.fine.labels[patch_ix]
            regions = s.grid[patch_ix]
            population = torch.bincount(s.fine.labels, minlength=s.fine.base_count).float()
            cluster_counts = torch.bincount(labels, minlength=s.fine.base_count).float()
            grid_counts = torch.bincount(regions, minlength=s.regions).float()
            variants = dict(
                executed=bias[local][ids[local]>=0].float().exp(),
                unit=torch.ones(len(selected), device=q.device),
                global_count=torch.full((len(selected),), len(s.patches)/max(1,len(selected)), device=q.device),
                cluster_count=population[labels]/cluster_counts[labels],
                grid_count=s.grid_sizes[regions]/grid_counts[regions],
            )
            prior = self.priors.get(group)
            if prior is not None:
                pi = torch.zeros(s.fine.base_count, device=q.device).scatter_add_(0, s.fine.labels, prior)
                kept = torch.zeros_like(pi).scatter_add_(0, labels, prior[patch_ix])
                variants['cluster_prior'] = pi[labels]/kept[labels].clamp_min(1e-30)
            ref_norm = dense_out.square().sum().clamp_min(1e-20)
            row = dict(layer=layer, group=group, history_sampled=(group % 8 == 0), selected_keys=len(selected),
                       patch_budget=s.budget, density=(len(selected)+len(s.specials))/s.n,
                       cluster_coverage=float((cluster_counts>0).float().mean()),
                       retained_dense_mass=float(dense_p[:,:,support].sum(-1).mean()),
                       dense_special_mass=float(dense_p[:,:,specials].sum(-1).mean()),
                       head_mean_value_energy_ratio=float(v.mean(0).float().square().mean()/v.float().square().mean()),
                       errors={})
            for name, weights in variants.items():
                logw = torch.cat((weights, torch.ones(len(specials),device=q.device))).log()
                p = (support_logits+logw).softmax(-1)
                output = p @ sv
                delta = output-dense_out
                flat = output.transpose(0,1).reshape(len(positions),-1)
                target = dense_out.transpose(0,1).reshape(len(positions),-1)
                projected_delta = torch.nn.functional.linear(flat-target,self.module.proj.weight.float(),None)
                projected_target = torch.nn.functional.linear(target,self.module.proj.weight.float(),None)
                row['errors'][name] = dict(nmse=float(delta.square().sum()/ref_norm),
                    projected_nmse=float(projected_delta.square().sum()/projected_target.square().sum().clamp_min(1e-20)),
                    cosine=float(torch.nn.functional.cosine_similarity(output,dense_out,dim=-1).mean()),
                    special_mass=float(p[:,:,-len(specials):].sum(-1).mean()))
            actual = result[:,positions].float()
            actual_flat=actual.transpose(0,1).reshape(len(positions),-1)
            dense_flat=dense_out.transpose(0,1).reshape(len(positions),-1)
            weight=self.module.proj.weight.float()
            output_bias=None if self.module.proj.bias is None else self.module.proj.bias.float()
            actual_projected=torch.nn.functional.linear(actual_flat,weight,output_bias)
            dense_projected=torch.nn.functional.linear(dense_flat,weight,output_bias)
            camera_mask=torch.tensor([group*32+o in self.camera_ids for o in offsets],device=q.device)
            row['output_similarity']={}
            for subset,mask in [('all',torch.ones_like(camera_mask)),('camera',camera_mask),('probe_non_camera',~camera_mask)]:
                if not bool(mask.any()):continue
                row['output_similarity'][subset]=dict(
                    av=vector_similarity(actual_flat[mask],dense_flat[mask]),
                    projected=vector_similarity(actual_projected[mask],dense_projected[mask]))
            dropped=cluster_counts[s.fine.labels]==0
            row['dropped_patch_population_fraction']=float(dropped.float().mean())
            row['dropped_cluster_dense_mass']=float(dense_p[:,:,s.patches][:,:,dropped].sum(-1).mean())
            execution_bias = torch.cat((bias[local][ids[local]>=0].float(),torch.zeros(len(specials),device=q.device)))
            expected = (support_logits+execution_bias).softmax(-1) @ sv
            row['kernel_vs_explicit_nmse'] = float((actual-expected).square().sum()/expected.square().sum().clamp_min(1e-20))
            if prior is not None:
                truth = dense_p[:,:,s.patches].mean((0,1))
                truth = truth/truth.sum()
                estimate = prior/prior.sum().clamp_min(1e-30)
                row['prior_total_variation'] = float((truth-estimate).abs().sum()/2)
                row['prior_selected_mass'] = float(estimate[patch_ix].sum())
                row['dense_patch_selected_mass'] = float(truth[patch_ix].sum())
            self.rows.append(row)
            self.supports[f'{layer}:{group}'] = patch_ix.cpu()
        self.start += len(ids)
        if self.start >= s.groups:
            save(self.out/'audit_partial.json', self.rows)
            print('AUDIT_LAYER', layer, 'rows', len(self.rows), flush=True)
        return result, evidence


def run(args):
    from addict import Dict
    import yaml
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from saf3r.patch import reuse_residual_exploration as implementation
    from saf3r.patch import reuse_sampled_history as history
    from saf3r.triton import reuse_direct_sampled_probability as kernel
    from saf3r.patch import reuse_window_graph as selector
    args.output.mkdir(parents=True, exist_ok=False)
    quality = quality_record({}, 'Sampled same-input attention diagnostic; pose/recon_unposed not evaluated')
    quality['speedup_unavailable'] = 'Diagnostic instrumentation; no matched warmed Dense full-forward timings'
    save(args.output/'quality.json', quality)
    (args.output/'quality.md').write_text(quality_markdown(quality))
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dataset = MV_REGISTRY.get(args.dataset)()
    indices, scene = sample_frames(dataset.get_data(args.scene),args.scene,args.frames,sample_mode='uniform',seed=0)
    cfg = Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.pop('patch_module');cfg.pop('sparse_config_path')
    cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda')
    images=load_images(model,'vggt',scene.image_files).cuda()
    config=implementation.ReuseExploreConfig(**json.loads(args.config.read_text()))
    state=implementation.install_reuse_residual_exploration(model,config)
    n=len(images)*(images.shape[-2]//14*images.shape[-1]//14+5)
    groups=sorted(set(np.linspace(0,(n+31)//32-1,args.groups,dtype=int).tolist()+[0,1,8,9]))
    groups=[g for g in groups if g<(n+31)//32]
    group_offsets={g:sorted(set(min(g*32+o,n-1)-g*32 for o in (4,12,20,28))) for g in groups}
    tokens_per_frame=images.shape[-2]//14*images.shape[-1]//14+5
    camera_ids=list(range(0,n,tokens_per_frame)) if args.camera_probes else []
    for index in camera_ids:group_offsets.setdefault(index//32,[]).append(index%32)
    group_offsets={g:sorted(set(offsets)) for g,offsets in group_offsets.items()}
    groups=sorted(group_offsets)
    audit=Audit(state,groups,args.output,group_offsets,camera_ids)
    trajectory={}
    handles=[]
    def positions_for(x):
        return torch.tensor([g*32+offset for g in groups for offset in group_offsets[g]],device=x.device)
    def capture_sparse(layer):
        def hook(module, inputs, output):
            indices=positions_for(inputs[0])
            trajectory[layer]=dict(input=inputs[0][:,indices].detach().cpu(),output=output[:,indices].detach().cpu())
        return hook
    if args.dense_trajectory:
        for layer,block in enumerate(model.aggregator.global_blocks):
            handles.append(block.attn.register_forward_hook(capture_sparse(layer)))
    prepare,attn,prior=implementation.prepare_qkv,kernel.direct_sparse_attention_sampled,history.build_prior
    original_select=selector.select
    if args.representative_control:
        def rep_control(s, layer, *a, **kw):
            ids,real,weights,record=original_select(s,layer,*a,**kw)
            if len(s.fine.reps)!=s.budget:raise ValueError('Representative control requires cluster count == patch budget')
            population=torch.bincount(s.fine.labels,minlength=len(s.fine.reps)).float()
            ids=s.fine.reps.int()[None].expand(len(ids),-1).contiguous()
            weights=population[None].expand_as(ids).contiguous()
            record={key:torch.zeros_like(value) for key,value in record.items()}
            for key in ('counts','representatives','state_1_clusters'):
                if key in record:record[key].fill_(s.budget)
            return ids,ids>=0,weights,record
        selector.select=rep_control
    implementation.prepare_qkv=lambda m,x,p:audit.prepare(prepare,m,x,p)
    kernel.direct_sparse_attention_sampled=lambda *a,**kw:audit.attention(attn,*a,**kw)
    history.build_prior=lambda *a,**kw:audit.prior(prior,*a,**kw)
    save(args.output/'input.json',dict(dataset=args.dataset,scene=args.scene,frame_indices=indices,image_paths=scene.image_files,
        shape=list(images.shape),checkpoint=cfg.ckpt_path,checkpoint_size=Path(cfg.ckpt_path).stat().st_size,
        config=json.loads(args.config.read_text()),sampled_groups=groups,
        sampled_query_indices=[g*32+o for g in groups for o in group_offsets[g]],camera_query_indices=camera_ids,
        comparison_point='After output projection Wo and its bias, before residual/LayerScale; local projection evaluated in FP32 from actual sparse AV kernel output',
        protocol='Sparse-trajectory QKV; explicitly listed regular-probe and optional camera queries; Dense and counterfactual weights share QKV and mask; no geometry truth',
        mask_override='all representatives, population weighting; diagnostic control only' if args.representative_control else None,
        f1_threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample,saf3r_topk=None,saf3r_profile='N/A: no SAF3R installed',
        hardware=torch.cuda.get_device_name(),config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest()))
    try:
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            raw=model(images)
        torch.cuda.synchronize()
        save(args.output/'runtime.json',dict(instrumented_wall_seconds=time.perf_counter()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,quality_evaluation=False,
            speedup_vs_dense=None,reason='Instrumented diagnostic, not a warmed speed comparison'))
        assert torch.isfinite(raw['pose_enc']).all()
        save(args.output/'layers.json',state.records)
        save(args.output/'audit.json',audit.rows)
        torch.save(audit.supports,args.output/'supports.pt')
        summary={}
        for name in sorted(set().union(*(r['errors'] for r in audit.rows))):
            valid=[r['errors'][name] for r in audit.rows if name in r['errors']]
            summary[name]={key:float(np.mean([r[key] for r in valid])) for key in valid[0]}
            summary[name]['sampled_group_layers']=len(valid)
        summary['kernel_vs_explicit_nmse_max']=max(r['kernel_vs_explicit_nmse'] for r in audit.rows)
        save(args.output/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        if args.dense_trajectory:
            for handle in handles:handle.remove()
            handles=[]
            state.reset_handle.remove()
            trajectory_rows=[]
            def capture_dense(layer):
                def hook(module,inputs,output):
                    indices=positions_for(inputs[0])
                    row=dict(layer=layer)
                    for key,dense in [('input',inputs[0][:,indices]),('output',output[:,indices])]:
                        sparse=trajectory[layer][key].float().to(dense.device)
                        dense=dense.float()
                        row[key+'_nmse']=float((sparse-dense).square().sum()/dense.square().sum().clamp_min(1e-20))
                        row[key+'_cosine']=float(torch.nn.functional.cosine_similarity(sparse,dense,dim=-1).mean())
                        row[key+'_relative_l2']=row[key+'_nmse']**.5
                        camera_mask=torch.tensor([int(i) in set(camera_ids) for i in indices.tolist()],device=dense.device)
                        for subset,mask in [('camera',camera_mask),('probe_non_camera',~camera_mask)]:
                            if bool(mask.any()):row[key+'_'+subset]=vector_similarity(sparse[0,mask],dense[0,mask])
                    trajectory_rows.append(row)
                    print('TRAJECTORY_LAYER',layer,row,flush=True)
                return hook
            for layer,block in enumerate(model.aggregator.global_blocks):
                block.attn=block.attn.original
                handles.append(block.attn.register_forward_hook(capture_dense(layer)))
            del raw
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                dense_raw=model(images)
            save(args.output/'dense_trajectory.json',trajectory_rows)
            del dense_raw
    except Exception as exc:
        failed = quality_record({}, f'Attention diagnostic failed: {type(exc).__name__}: {exc}; pose/recon_unposed not evaluated')
        save(args.output/'quality.json', failed)
        (args.output/'quality.md').write_text(quality_markdown(failed))
        raise
    finally:
        for handle in handles:handle.remove()
        implementation.prepare_qkv=prepare
        kernel.direct_sparse_attention_sampled=attn
        history.build_prior=prior
        selector.select=original_select


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='eth3d',choices=['eth3d','7scenes'])
    p.add_argument('--scene',default='kicker')
    p.add_argument('--frames',type=int,default=8)
    p.add_argument('--groups',type=int,default=12)
    p.add_argument('--dense-trajectory',action='store_true',help='Additional Dense diagnostic forward to measure accumulated layer drift; no quality or speed claim')
    p.add_argument('--representative-control',action='store_true',help='Diagnostic only: replace dynamic masks by all current representatives; requires C=B and non-unit weighting')
    p.add_argument('--camera-probes',action='store_true',help='Additionally audit every frame camera query, separately from regular probes')
    p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/reuse_shared_value_loss_256.json')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
