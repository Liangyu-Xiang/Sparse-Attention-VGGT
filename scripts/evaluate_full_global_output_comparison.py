"""Matched all24-global-layer Dense/SAF3R/History80 output and quality diagnosis.

Run a scene's three methods sequentially on one GPU; each uses a fresh model.
Capture real AV and projected attention outputs, plus own-input Dense local
counterfactuals. No original result is overwritten and no NxN mask is saved.
"""
import argparse
import csv
from dataclasses import asdict,replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))
from scripts.compare_deep_head_rollouts import head_output_metrics,original_attention,SCOPES
from scripts.quality_metrics import quality_record,quality_cells,LABELS
from scripts.evaluate_attention_first_residual_value import write_quality

METHODS=('dense','saf3r','history80')
COMPARISONS=('rollout_vs_dense','same_input_vs_dense')
SOURCE_FILES=(
    'scripts/evaluate_full_global_output_comparison.py','scripts/compare_deep_head_rollouts.py',
    'scripts/quality_metrics.py','scripts/evaluate_attention_first_residual_value.py',
    'saf3r/patch/value_region_av.py','saf3r/patch/fused_shared_av.py',
    'saf3r/patch/parent_region_selection.py','saf3r/patch/value_clustering_gpu.py',
    'saf3r/patch/region_history_attention.py','saf3r/patch/patch_vggt.py',
    'saf3r/patch/wrapper.py','saf3r/patch/sparse_attention.py','saf3r/patch/utils.py',
    'saf3r/triton/fused_shared_av.py','saf3r/triton/value_region_selection.py',
    'saf3r/triton/shared_av_routing.py','saf3r/triton/shared_av_terminal.py',
    'saf3r/triton/sparse_topk_attention.py','saf3r/triton/fused_qk_topk.py',
    'configs/evaluation/eval_saf3r_vggt.yaml','configs/evaluation/parent_region_history80.json',
    'configs/evaluation/default_datasets.yaml','configs/sparse_attn/vggt/eth3d-train-fltr-calib_cmpmse.json',
    'saf3r/utils/model_utils.py','saf3r/utils/data_utils.py','evaluation/evaluate_da3bench.py',
    'evaluation/benchmarks/da3/bench/constants.py','evaluation/benchmarks/da3/bench/utils.py',
    'evaluation/benchmarks/da3/bench/datasets/sevenscenes.py',
    'evaluation/benchmarks/da3/bench/datasets/eth3d.py')


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
    tmp.replace(path)


def history80_config():
    from saf3r.patch.value_region_av import ValueRegionAVConfig
    c=ValueRegionAVConfig(**json.loads((ROOT/'configs/evaluation/parent_region_history80.json').read_text()))
    assert c.history_fraction==.8 and c.parent_only and not c.parent_topk and not c.parent_descriptors
    assert not c.adaptive_gate and not c.query_shift_step and c.retentions==(.2,)*24
    assert c.attention_ratios==(0.,)+(.2,)*9+(.8,)*7+(.2,)*7
    return c


class FullOutputCapture:
    def __init__(self,model,method,out,reference,frames,ph,pw):
        import torch
        from saf3r.patch.region_history_attention import prepare_qkv
        self.method=method;self.out=out;self.reference=reference;self.layers=[];self.handles=[]
        self.pending={};self.per=ph*pw+5;self.n=frames*self.per
        for layer,block in enumerate(model.aggregator.global_blocks):
            base=original_attention(block.attn)
            if method!='dense':
                def before(module,inputs,kwargs,l=layer,o=base):
                    x=inputs[0];pos=kwargs.get('pos',inputs[1] if len(inputs)>1 else None)
                    q,k,v=prepare_qkv(o,x,pos);q,k=q.to(v.dtype),k.to(v.dtype)
                    av=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None],scale=o.scale)
                    flat=av.transpose(1,2).reshape(1,x.shape[1],-1).contiguous()
                    # Functional projection avoids calling projection-input
                    # capture hooks recursively for this extra Dense reference.
                    projected=torch.nn.functional.linear(flat,o.proj.weight,o.proj.bias)
                    self.pending[l]=(flat,o.proj_drop(projected))
                self.handles.append(block.attn.register_forward_pre_hook(before,with_kwargs=True))
            def av_hook(module,inputs,l=layer):
                actual=inputs[0].detach()
                if actual.shape!=(1,self.n,1024):raise ValueError('Unexpected VGGT AV geometry')
                file=self.reference/'dense_outputs'/f'AV_L{l:02d}.pt'
                if method=='dense':
                    file.parent.mkdir(parents=True,exist_ok=True);torch.save(actual.cpu(),file)
                    ref=actual
                else:ref=torch.load(file,map_location='cpu',weights_only=True)
                hs,agg=head_output_metrics(actual,ref,self.per)
                local_hs,local_agg=(None,None)
                if method!='dense':local_hs,local_agg=head_output_metrics(actual,self.pending[l][0],self.per)
                heads=[]
                for h in range(16):
                    r=dict(head=h,rollout_vs_dense=hs[h])
                    if local_hs is not None:r['same_input_vs_dense']=local_hs[h]
                    heads.append(r)
                if len(self.layers)!=l:raise ValueError('Invalid global layer capture order')
                self.layers.append(dict(layer=l,heads=heads,preprojection=dict(rollout_vs_dense=agg)))
                if local_agg is not None:self.layers[-1]['preprojection']['same_input_vs_dense']=local_agg
            def module_hook(module,inputs,output,l=layer):
                actual=output.detach();file=self.reference/'dense_outputs'/f'OUT_L{l:02d}.pt'
                if method=='dense':torch.save(actual.cpu(),file);ref=actual
                else:ref=torch.load(file,map_location='cpu',weights_only=True)
                _,metrics=head_output_metrics(actual,ref,self.per,heads=1)
                row=self.layers[l];row['module_output']=dict(rollout_vs_dense=metrics)
                if method!='dense':
                    _,metrics=head_output_metrics(actual,self.pending.pop(l)[1],self.per,heads=1)
                    row['module_output']['same_input_vs_dense']=metrics
                save(out/'output_metrics.json',dict(method=method,layers=self.layers,complete=len(self.layers)==24))
                save(out/'capture_progress.json',dict(layer=l,heads=16))
                print('CAPTURE',method,l,flush=True)
            self.handles.append(base.proj.register_forward_pre_hook(av_hook))
            self.handles.append(block.attn.register_forward_hook(module_hook))

    def close(self):
        for h in self.handles:h.remove()


def snapshot(scene_root):
    hashes={}
    for name in SOURCE_FILES:
        b=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(b).hexdigest()
        dest=scene_root/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    save(scene_root/'source_sha256.json',hashes)
    return hashes


def run_method(args,method,scene_root,hashes):
    import numpy as np
    import torch
    import open3d as o3d
    import yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from saf3r.patch.value_region_av import install_value_region_av,ValueRegionAVAttention
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from evaluate_da3bench import load_scene_data,evaluate
    out=scene_root/method
    if (out/'result.json').exists() or (out/'smoke.json').exists():raise ValueError('Refuse to overwrite completed run')
    save(out/'status.json',dict(stage='loading'))
    write_quality(out,quality_record({},'Output diagnostic/quality run in progress'))
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);o3d.utility.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=MV_REGISTRY.get(args.dataset)()
    positions,scene=sample_frames(dataset.get_data(args.scene),args.scene,args.frames,sample_mode='uniform',seed=0)
    if len(scene.image_files)<2:raise ValueError('Insufficient actual frames')
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    cfg.sparse_config_path=str(ROOT/cfg.sparse_config_path)
    if cfg.patch_module.topk!=4 or not cfg.patch_module.lazy_dino_topk:raise ValueError('SAF3R default changed')
    if method!='saf3r':
        del cfg['patch_module'];del cfg['sparse_config_path']
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    ph,pw=images.shape[-2]//14,images.shape[-1]//14
    state=None
    if method=='history80':
        config=replace(history80_config(),population_compensation=bool(getattr(args,'population_compensation',False)))
        override=getattr(args,'history80_config_path',None)
        if override:
            from saf3r.patch.value_region_av import ValueRegionAVConfig
            config=ValueRegionAVConfig(**json.loads(Path(override).read_text()))
        state=install_value_region_av(model,config)
        assert all(isinstance(b.attn,ValueRegionAVAttention) for b in model.aggregator.global_blocks)
    elif method=='saf3r':
        from saf3r.patch.patch_vggt import HeadwiseSparseAttentionWrapper
        from saf3r.patch import update_args_per_forward
        assert all(isinstance(b.attn,HeadwiseSparseAttentionWrapper) for b in model.aggregator.global_blocks)
        update_args_per_forward(model,'vggt',*images.shape[-2:])
    canonical=dict(image_paths=scene.image_files,frame_indices=positions,shape=list(images.shape),checkpoint=cfg.ckpt_path)
    if method!='dense':
        reference_input=json.loads((scene_root/'dense/input.json').read_text())
        if canonical!=reference_input:raise ValueError('Input/checkpoint/preprocessing mismatch')
        if not json.loads((scene_root/'dense/output_metrics.json').read_text())['complete']:
            raise ValueError('Dense reference incomplete')
    save(out/'input.json',canonical)
    save(out/'config.json',dict(method=method,model_config=dict(cfg),
        history80=asdict(config) if method=='history80' else None,
        population_logit_compensation=method=='history80' and config.population_compensation,
        global_layers=list(range(24)),frame_local_unchanged=True,
        retention_target=.2 if method=='history80' else None,
        saf3r_selector='lazy first-correspondence-head per-frame Top4' if method=='saf3r' else None,
        query_sampling='All Queries; all/patch/special scopes',seed=0,
        comparisons=['independent Dense rollout','same-method-current-input Dense counterfactual'],
        capture_positions=['AV before output projection','module output after projection before residual'],
        quality_tasks=['pose','recon_unposed'],warmup=0,repeats=1,benchmark_timing=False))
    recorder=FullOutputCapture(model,method,out,scene_root/'dense',len(images),ph,pw)
    save(out/'status.json',dict(stage='forward'));torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/2**30
    recorder.close()
    assert len(recorder.layers)==24 and all(len(l['heads'])==16 and 'module_output' in l for l in recorder.layers)
    if state is not None:save(out/'history80_runtime.json',state.records)
    if method=='saf3r':
        save(out/'saf3r_runtime.json',[dict(layer=l,heads=b.attn.head_cfg,
            first_correspondence_selector_executed=b.attn.first_dino_flag,
            cached_indices_shape=list(b.attn.est_topk_idx.shape)) for l,b in enumerate(model.aggregator.global_blocks)])
    finite=all(bool(torch.isfinite(v).all()) for v in raw.values() if torch.is_tensor(v))
    if not finite:raise ValueError('Nonfinite model outputs')
    common=dict(dataset=args.dataset,scene=args.scene,method=method,frames=len(images),max_frames=args.frames,
        population_compensation=method=='history80' and config.population_compensation,
        input=canonical,source_sha256=hashes,outputs_complete=True,
        diagnostic_seconds=seconds,peak_allocated_gib=peak,
        diagnostic_timing='Includes capture I/O, comparisons and extra Dense computations; NOT speed benchmark',
        hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),
        quality_protocol=dict(reconstruction='recon_unposed',f1_threshold_m=dataset.eval_threshold,
            downsample_m=dataset.down_sample,cd_definition='(ACC+Comp)/2, unsquared',seed=0))
    if args.smoke:
        q=quality_record({},'Two-frame implementation diagnostic, no pose/reconstruction or warmed speed evaluation')
        save(out/'smoke.json',dict(common,quality=q,finite_outputs=finite))
        write_quality(out,q);save(out/'status.json',dict(stage='smoke_complete'))
    else:
        ext,intr=pose_encoding_to_extri_intri(raw['pose_enc'],images.shape[-2:])
        pred=Dict(extrinsics=ext.squeeze().float().cpu().numpy(),intrinsics=intr.squeeze().float().cpu().numpy(),
            depth=raw['depth'].squeeze().float().cpu().numpy(),conf=raw['depth_conf'].squeeze().float().cpu().numpy(),
            processed_images=raw['images'].squeeze().permute(0,2,3,1).cpu().numpy())
        del raw;raw=None
        # Free model before CPU fusion; sampled GT matches exactly. There is no
        # warm-timing claim, and auxiliary references are no longer needed.
        del recorder,model;model=None;gc.collect();torch.cuda.empty_cache()
        save(out/'status.json',dict(stage='quality'))
        metrics={};errors={}
        try:
            o3d.utility.random.seed(0);np.random.seed(0)
            _,gt=load_scene_data(dataset,args.dataset,args.scene,args.frames,'uniform',seed=0)
            for task in ('pose','recon_unposed'):
                try:metrics.update(evaluate(dataset,gt,pred,[task]))
                except Exception as exc:errors[task]=f'{type(exc).__name__}: {exc}'
        except Exception as exc:errors['ground_truth']=f'{type(exc).__name__}: {exc}'
        q=quality_record(metrics,'; '.join(f'{k}: {v}' for k,v in errors.items()) or 'Task unavailable')
        q['speedup_vs_vggt']=1. if method=='dense' else None
        q['speedup_unavailable']=None if method=='dense' else 'Instrumented output diagnosis; no paired warmed uninstrumented full-forward timing'
        save(out/'result.json',dict(common,quality=q,metrics=metrics,quality_errors=errors))
        write_quality(out,q);save(out/'status.json',dict(stage='complete' if q['complete'] else 'outputs_complete_quality_incomplete'))
    del raw,model,images,state;gc.collect();torch.cuda.empty_cache()
    print('METHOD_COMPLETE',args.dataset,args.scene,method,flush=True)


def scene_report(scene_root):
    output={};quality={};rows=[];frames={}
    for method in METHODS:
        p=scene_root/method/'output_metrics.json'
        if p.exists():
            x=json.loads(p.read_text())
            if x['complete']:output[method]=x
        p=scene_root/method/'result.json'
        if p.exists():
            x=json.loads(p.read_text());quality[method]=x;frames[method]=x['frames']
    if quality:
        vals=list(quality.values())
        if any(v['input']!=vals[0]['input'] or v['quality_protocol']!=vals[0]['quality_protocol'] for v in vals):
            raise ValueError('Mismatched inputs or quality protocol across methods')
    for method,x in output.items():
        for l in x['layers']:
            for comp in COMPARISONS:
                for scope,m in l['module_output'].get(comp,{}).items():
                    rows.append(dict(method=method,layer=l['layer'],scope=scope,comparison=comp,position='module_output',head=-1,**m))
                for scope,m in l['preprojection'].get(comp,{}).items():
                    rows.append(dict(method=method,layer=l['layer'],scope=scope,comparison=comp,position='AV_aggregate',head=-1,**m))
                for h in l['heads']:
                    for scope,m in h.get(comp,{}).items():
                        rows.append(dict(method=method,layer=l['layer'],scope=scope,comparison=comp,position='head_AV',head=h['head'],**m))
    if rows:
        with (scene_root/'output_comparison.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    complete=set(output)==set(METHODS)
    qcomplete=set(quality)==set(METHODS) and all(x['quality']['complete'] for x in quality.values())
    summary=dict(outputs_complete=complete,quality_complete=qcomplete,methods=output,quality=quality,frames=frames)
    save(scene_root/'comparison.json',summary)
    corrected=quality.get('history80',{}).get('population_compensation',False)
    configuration=scene_root/'history80/config.json'
    if configuration.exists():corrected=json.loads(configuration.read_text()).get('population_logit_compensation',corrected)
    lines=['# Full24-global-layer output comparison','',
        f'Output complete: {complete}; six-metric quality complete: {qcomplete}.',
        'SAF3R=repository full lazy Top4; History80=full24 layers,20% target; exact branch/history ratios in config.json; population compensation='+str(corrected)+'.',
        'Primary layer output: after output projection,before residual. Separate raw-head AV metrics in CSV.',
        'rollout_vs_dense includes accumulated upstream drift; same_input_vs_dense uses each method\'s OWN current input, not a common teacher input.',
        'All Queries, no sampling. Instrumented time/memory are diagnostic, no deployment speed comparison.',
        '', '| Method | '+' | '.join(LABELS)+' | Speedup |','|---|'+'---:|'*7]
    for method in METHODS:
        q=quality[method]['quality'] if method in quality else quality_record({},'Run pending/diagnostic/failed; see method status')
        lines.append('| '+method+' | '+' | '.join(quality_cells(q))+' | '+('1.00x reference' if method=='dense' and method in quality else 'N/A')+' |')
    for comp in COMPARISONS:
        lines+=['',f'## Module output: {comp}','', '| Layer | SAF3R error % | SAF3R cosine | History80 error % | History80 cosine |',
            '|---|---:|---:|---:|---:|']
        for l in range(24):
            cells=[]
            for method in ('saf3r','history80'):
                m=output.get(method,{}).get('layers',[])
                v=m[l]['module_output'].get(comp,{}).get('all_queries') if len(m)==24 else None
                cells+= [f"{v['relative_l2_frobenius']*100:.4f}",f"{v['mean_per_query_cosine']:.6f}"] if v else ['N/A','N/A']
            lines.append('| '+str(l)+' | '+' | '.join(cells)+' |')
    (scene_root/'comparison.md').write_text('\n'.join(lines)+'\n')
    return summary


def run(args):
    scene_root=args.output_dir/args.dataset/args.scene
    if scene_root.exists():raise ValueError('Fresh scene root required; do not overwrite prior results')
    scene_root.mkdir(parents=True)
    hashes=snapshot(scene_root);failures={}
    for method in METHODS:
        try:run_method(args,method,scene_root,hashes)
        except Exception as exc:
            out=scene_root/method;reason=f'{type(exc).__name__}: {exc}'
            failures[method]=reason
            write_quality(out,quality_record({},reason))
            save(out/'status.json',dict(stage='failed',error=reason,traceback=traceback.format_exc()))
            print('METHOD_FAILED',method,reason,flush=True)
            import torch
            gc.collect();torch.cuda.empty_cache()
            if method=='dense':
                for blocked in ('saf3r','history80'):
                    why='Dense reference failed: '+reason
                    write_quality(scene_root/blocked,quality_record({},why))
                    save(scene_root/blocked/'status.json',dict(stage='blocked',error=why))
                break
        scene_report(scene_root)
    summary=scene_report(scene_root)
    save(scene_root/'status.json',dict(stage='smoke_complete' if args.smoke and summary['outputs_complete'] else
        'complete' if summary['outputs_complete'] and summary['quality_complete'] else 'incomplete',failures=failures))
    if failures:raise RuntimeError('One or more method failures: '+json.dumps(failures))
    if not summary['outputs_complete']:raise RuntimeError('Incomplete output capture')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['7scenes','eth3d'],required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--frames',type=int,default=128)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--population-compensation',action='store_true')
    args=p.parse_args()
    if args.smoke and args.frames!=2:p.error('Smoke requires frames2')
    run(args)
