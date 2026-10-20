"""Two-frame masks/Dense maps and unsampled full-sequence layer-output errors."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'evaluation'))
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record


def output_errors(current,reference,chunk=8192):
    """Full tensors, no sampled Queries; streaming FP32 arithmetic/FP64 sums."""
    import torch
    a=current.reshape(-1,current.shape[-1]);b=reference.reshape_as(a)
    if current.shape!=reference.shape:raise ValueError('Layer shape mismatch')
    aa=bb=ab=error=token_cos=0.
    for start in range(0,len(a),chunk):
        x=a[start:start+chunk].float();y=b[start:start+chunk].float()
        xx=x.square().sum(-1);yy=y.square().sum(-1);xy=(x*y).sum(-1)
        aa+=xx.sum(dtype=torch.float64).item();bb+=yy.sum(dtype=torch.float64).item()
        ab+=xy.sum(dtype=torch.float64).item();error+=(x-y).square().sum(dtype=torch.float64).item()
        denom=(xx*yy).sqrt()
        cos=torch.where(denom>0,xy/denom.clamp_min(1e-30),((xx==0)&(yy==0)).float())
        token_cos+=cos.sum(dtype=torch.float64).item()
    denominator=math.sqrt(aa*bb)
    return dict(relative_l2=math.sqrt(error/max(bb,1e-30)),relative_squared_l2=error/max(bb,1e-30),
        cosine=ab/denominator if denominator else float(aa==bb),mean_token_cosine=token_cos/len(a),
        tokens=len(a),channels=a.shape[-1],scope='Every token/channel, no Query sampling')


def masks_from_capture(capture,n,method):
    import torch
    mask=torch.zeros((n,n),dtype=torch.bool)
    if method=='old':
        common=capture['common'].cpu().long();indices=capture['attention'].cpu().long()
        mask[:,common]=True
        for group,ids in enumerate(indices):
            mask[group*32:min(n,(group+1)*32),ids[ids>=0]]=True
        offset=capture.get('query_offset',0)
        if offset:mask=mask.roll(offset,0)
    else:
        mask[:,capture['specials'].cpu().long()]=True
        group=0
        for window in capture['windows']:
            for ids in window['union'].cpu().long():
                mask[group*32:min(n,(group+1)*32),ids]=True;group+=1
    return mask.numpy()


def build(args,method,frames,capture=False):
    import torch
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from saf3r.patch.attention_first_residual_value import AttentionFirstConfig,install_attention_first_residual_value
    from saf3r.patch.value_region_av import ValueRegionAVConfig,install_value_region_av
    torch.manual_seed(0);np.random.seed(0)
    dataset=MV_REGISTRY.get('7scenes')()
    positions,scene=sample_frames(dataset.get_data(args.scene),args.scene,frames,sample_mode='uniform',seed=0)
    checkpoint=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(Dict(name='VGGT',dpt_only=True,ckpt_path=checkpoint),'cuda')
    images=load_images(model,'vggt',scene.image_files).cuda();state=None
    if method=='old':state=install_value_region_av(model,ValueRegionAVConfig(parent_only=True,history_fraction=.8,capture=capture))
    elif method=='attention_first':
        settings=json.loads(args.config.read_text());settings['capture']=capture
        state=install_attention_first_residual_value(model,AttentionFirstConfig(**settings))
    info=dict(images=scene.image_files,positions=positions,shape=list(images.shape),checkpoint=checkpoint)
    return model,images,state,info


def diagnose(args):
    import torch
    torch.set_num_threads(4)
    root=args.output_dir/args.action/args.scene;root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():return
    frames=2 if args.action=='visualize' else args.frames
    timing={};results={};input_info=None
    for method in ('dense','old','attention_first'):
        save(root/'status.json',dict(stage='running',method=method,action=args.action))
        model,images,state,info=build(args,method,frames,capture=args.action=='visualize')
        if input_info is not None and info!=input_info:raise ValueError('Inputs differ')
        input_info=info;save(root/'input.json',info)
        handles=[];errors=[]
        if args.action=='visualize' and method=='dense':
            from saf3r.patch.region_history_attention import prepare_qkv
            def pre(module,inputs,kwargs,layer):
                q,k,_=prepare_qkv(module,inputs[0],kwargs.get('pos',inputs[1] if len(inputs)>1 else None))
                probabilities=[]
                for start in range(0,q.shape[1],128):
                    with torch.autocast('cuda',enabled=False):
                        z=q[:,start:start+128].float()@k.float().transpose(-1,-2)*module.scale
                        probabilities.append(z.softmax(-1).mean(0).cpu())
                np.save(root/f'dense_L{layer:02}.npy',torch.cat(probabilities).numpy())
            for layer,block in enumerate(model.aggregator.global_blocks):
                handles.append(block.attn.register_forward_pre_hook(lambda m,i,k,l=layer:pre(m,i,k,l),with_kwargs=True))
        elif args.action=='deviation':
            def hook(module,inputs,output,layer,scope):
                actual=output.detach().cpu()
                file=root/'dense_outputs'/f'{scope}_L{layer:02}.pt'
                if method=='dense':
                    file.parent.mkdir(parents=True,exist_ok=True);torch.save(actual,file)
                else:
                    reference=torch.load(file,map_location='cpu',weights_only=True)
                    errors.append(dict(layer=layer,scope=scope,**output_errors(actual,reference)))
                    # Direct old-new comparison, not inferred from two Dense distances.
                    oldfile=root/'old_outputs'/file.name
                    if method=='old':
                        oldfile.parent.mkdir(parents=True,exist_ok=True);torch.save(actual,oldfile)
                    else:
                        previous=torch.load(oldfile,map_location='cpu',weights_only=True)
                        errors.append(dict(layer=layer,scope=scope,comparison='new_vs_old',**output_errors(actual,previous)))
                print('LAYER',method,layer,scope,flush=True)
            for layer,block in enumerate(model.aggregator.global_blocks):
                handles.append(block.register_forward_hook(lambda m,i,o,l=layer:hook(m,i,o,l,'global_block')))
                handles.append(block.attn.register_forward_hook(lambda m,i,o,l=layer:hook(m,i,o,l,'attention_output')))
        torch.cuda.synchronize();begin=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=model(images)
        torch.cuda.synchronize();timing[method]=time.perf_counter()-begin
        for handle in handles:handle.remove()
        if args.action=='visualize' and state is not None:
            for layer in range(24):
                mask=masks_from_capture(state.captured[layer],state.n,method)
                np.savez_compressed(root/f'{method}_L{layer:02}_mask.npz',mask=mask)
        if errors:save(root/f'{method}_errors.json',errors)
        results[method]=dict(errors=errors,layers=[] if state is None else state.records)
        del handles,model,images,state,output;gc.collect();torch.cuda.empty_cache()
    if args.action=='visualize':results['mask_comparison']=plot_masks(root)
    else:
        lines=['| Method comparison | Scope | Global layer | Relative L2 ↓ | Cosine ↑ | Mean token cosine ↑ |',
               '|---|---|---:|---:|---:|---:|']
        for method in ('old','attention_first'):
            for row in results[method]['errors']:
                label=row.get('comparison',method+'_vs_dense')
                lines.append(f"| {label} | {row['scope']} | {row['layer']} | {row['relative_l2']:.6f} | {row['cosine']:.6f} | {row['mean_token_cosine']:.6f} |")
        (root/'deviations.md').write_text('\n'.join(lines)+'\n')
    quality=quality_record({},'Mask/output diagnostic, not a task-quality or performance evaluation')
    write_quality(root,quality)
    save(root/'result.json',dict(scene=args.scene,frames=frames,input=input_info,results=results,quality=quality,
        diagnostic_seconds_including_hooks_and_disk=timing,benchmark_timing=False,
        semantics='Independent autoregressive rollouts on identical inputs; layer errors include accumulated earlier-layer error, not Dense teacher forcing'))
    save(root/'status.json',dict(stage='complete'))


def plot_masks(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for layer in range(24):
        p=np.load(root/f'dense_L{layer:02}.npy')
        old=np.load(root/f'old_L{layer:02}_mask.npz')['mask']
        new=np.load(root/f'attention_first_L{layer:02}_mask.npz')['mask']
        k=math.ceil(p.shape[1]*.1);top=np.argpartition(p,-k,axis=1)[:,-k:]
        rr=np.arange(len(p))[:,None]
        row=dict(layer=layer,old_density=float(old.mean()),new_density=float(new.mean()),
            old_top10_recall=float(old[rr,top].mean()),new_top10_recall=float(new[rr,top].mean()),
            mask_iou=float((old&new).sum()/max((old|new).sum(),1)),
            symmetric_difference_density=float((old^new).mean()),oracle='Top10% of per-head-softmax then head-averaged Dense probabilities per Query')
        rows.append(row)
        fig,axes=plt.subplots(1,3,figsize=(13,4.5))
        draw(axes,p,old,new,row)
        fig.suptitle(f'Global L{layer:02}: original token order, two frames; rows=Query, columns=Key')
        fig.tight_layout();fig.savefig(root/f'L{layer:02}_comparison.png',dpi=160);plt.close(fig)
    for first in (0,8,16):
        fig,axes=plt.subplots(8,3,figsize=(12,32))
        for line,layer in enumerate(range(first,first+8)):
            draw(axes[line],np.load(root/f'dense_L{layer:02}.npy'),
                np.load(root/f'old_L{layer:02}_mask.npz')['mask'],
                np.load(root/f'attention_first_L{layer:02}_mask.npz')['mask'],rows[layer],short=True)
        fig.tight_layout();fig.savefig(root/f'layers_{first:02}_{first+7:02}.png',dpi=120);plt.close(fig)
    save(root/'mask_comparison.json',rows)
    return rows


def draw(axes,p,old,new,row,short=False):
    axes[0].imshow(np.log10(p+1e-8),vmin=-6,vmax=0,cmap='magma',interpolation='nearest',rasterized=True)
    axes[0].set_title(f"L{row['layer']} Dense log10 probability [-6,0]")
    for ax,mask,name in zip(axes[1:],(old,new),('old','new')):
        ax.imshow(mask,vmin=0,vmax=1,cmap='gray',interpolation='nearest',rasterized=True)
        ax.set_title(f"{name}: {row[name+'_density']:.1%} edges; R@Top10 {row[name+'_top10_recall']:.1%}")
    for ax in axes:
        ax.axhline(len(p)/2-.5,color='cyan',lw=.5);ax.axvline(len(p)/2-.5,color='cyan',lw=.5)
        if short:ax.set_xticks([]);ax.set_yticks([])
        else:ax.set_xlabel('Original Key token');ax.set_ylabel('Original Query token')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--scene',required=True)
    p.add_argument('--action',choices=['visualize','deviation'],required=True);p.add_argument('--frames',type=int,default=128)
    p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/attention_first_residual_value.json')
    args=p.parse_args()
    try:diagnose(args)
    except Exception as exc:
        out=args.output_dir/args.action/args.scene
        write_quality(out,quality_record({},f'Diagnostic failed: {exc}'))
        save(out/'status.json',dict(stage='failed',error=str(exc)));raise
