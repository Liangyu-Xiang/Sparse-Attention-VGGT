"""Controlled and native middle-only region attention, exact two-frame masks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from saf3r.patch.region_history_attention import (
    RegionHistoryState,RegionHistoryAttention,prepare_qkv,install_region_history_attention)
from saf3r.patch.value_evolving_attention import token_layout
from scripts.overlay_vggt_query_attention import build_model
from saf3r.utils.data_utils import load_and_process_images_vggt

ROOT=Path(__file__).resolve().parents[1]


def dense_weights(q,k,chunk=128):
    return torch.cat([((q[:,i:i+chunk].float()@k.float().transpose(-1,-2))*q.shape[-1]**-.5)
        .softmax(-1).mean(0) for i in range(0,q.shape[1],chunk)])


def mask_from_ids(ids,n):
    result=torch.zeros((len(ids),n),device=ids.device,dtype=torch.bool)
    rr,cc=(ids>=0).nonzero(as_tuple=True)
    result[rr,ids[rr,cc]]=True
    return result


def recalls(mask,targets,frames,height,width):
    values={}
    for radius in [0,1,2,4]:
        expanded=mask if radius==0 else F.max_pool2d(mask.float().reshape(-1,1,height,width),
            2*radius+1,stride=1,padding=radius).reshape_as(mask).bool()
        values[f'recall_r{radius}']=float(expanded.gather(1,targets).float().mean())
    return values


def run(args):
    out=args.output_dir
    if (out/'result.json').exists():raise ValueError('Use fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=json.loads((ROOT/'outputs/global_attention/chess_gap10_layer_mean_all24/result.json').read_text())
    images=load_and_process_images_vggt(meta['image_paths']).cuda()
    model=build_model(Path(meta['checkpoint']),'cuda')
    dense_inputs={};handles=[]
    for layer,block in enumerate(model.aggregator.global_blocks):
        def capture(module,inputs,kwargs,output,l=layer):
            pos=kwargs.get('pos',inputs[1] if len(inputs)>1 else None)
            q,k,v=prepare_qkv(module,inputs[0],pos)
            dense_inputs[l]=dict(x=inputs[0].cpu(),pos=None if pos is None else pos.cpu(),
                q=q.cpu(),k=k.cpu(),v=v.cpu(),projected=output.cpu())
        handles.append(block.attn.register_forward_hook(capture,with_kwargs=True))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        dense_pose=model(images)['pose_enc'].float().cpu()
    for handle in handles:handle.remove()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.equal(dense_pose,model(images)['pose_enc'].float().cpu())
    frames,height,width=2,28,37;n=2082
    patches,specials=token_layout(frames,height*width,5,'cuda')
    oracle={};saved={};rows=[];timings=[];records=[]
    for layer in range(10,17):
        data=dense_inputs[layer];q,k,v=[data[key].cuda() for key in ['q','k','v']]
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):weights=dense_weights(q,k)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=F.scaled_dot_product_attention(q[None],k[None],v[None])[0].float()
        patch_budget=int(args.retention*n)-len(specials)
        target=weights[:,patches].argsort(dim=1,descending=True,stable=True)
        oracle[layer]=dict(weights=weights.cpu(),top10=target[:,:int(.1*len(patches))].cpu(),
            matched=target[:,:patch_budget].cpu(),output=output.cpu())
        saved[f'dense_L{layer}_matched_patch_indices']=oracle[layer]['matched'].numpy().astype(np.int32)
        saved[f'dense_L{layer}_top10_patch_indices']=oracle[layer]['top10'].numpy().astype(np.int32)

    def evaluate(scope,name,state,elapsed):
        timings.append(dict(scope=scope,setting=name,diagnostic_seconds=elapsed))
        for rec in state.records:records.append(dict(scope=scope,setting=name,**rec))
        for layer,data in state.captured.items():
            ids=data['indices'].cuda();mask=mask_from_ids(ids,n);patchmask=mask[:,patches]
            targets=oracle[layer]['top10'].cuda();matched=oracle[layer]['matched'].cuda()
            r=dict(scope=scope,setting=name,retention=args.retention,variant=args.variant,layer=layer,
                **recalls(patchmask,targets,frames,height,width))
            r['matched_budget_recall']=float(patchmask.gather(1,matched).float().mean())
            weights=oracle[layer]['weights'].cuda()
            r['dense_patch_mass']=float((weights[:,patches]*patchmask).sum(1).mean())
            r['density']=float(mask.float().mean())
            q,k,v=[data[key].cuda() for key in ['q','k','v']]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                local_output=F.scaled_dot_product_attention(q[None],k[None],v[None])[0].float()
            actual=data['output'].cuda().float();original=oracle[layer]['output'].cuda()
            r['local_output_error']=float(torch.linalg.vector_norm(actual-local_output)/torch.linalg.vector_norm(local_output))
            r['trajectory_output_error']=float(torch.linalg.vector_norm(actual-original)/torch.linalg.vector_norm(original))
            if scope=='native':
                with torch.inference_mode(),torch.autocast('cuda',enabled=False):local_weights=dense_weights(q,k)
                local_target=local_weights[:,patches].argsort(dim=1,descending=True,stable=True)[:,:targets.shape[1]]
                r['same_input_exact_recall']=float(patchmask.gather(1,local_target).float().mean())
                if layer==10:
                    assert torch.equal(data['q'],dense_inputs[layer]['q'])
                    assert torch.equal(data['k'],dense_inputs[layer]['k'])
            else:r['same_input_exact_recall']=r['recall_r0']
            rows.append(r)
            saved[f'{scope}_{name}_L{layer}_indices']=ids.cpu().numpy().astype(np.int32)
            assert bool((mask[:,specials]).all())
            assert (mask.sum(1)>=int(args.retention*n)).all()
        (out/'progress.json').write_text(json.dumps(dict(scope=scope,setting=name,completed_rows=len(rows))))
        print(scope,name,'done',round(elapsed,3),flush=True)

    settings=[('h0',0.,False),('h25',.25,False),('h50',.5,False),('h100',1.,False),('adaptive',0.,True)]
    originals=[block.attn for block in model.aggregator.global_blocks]
    for name,ratio,adaptive in settings:
        state=RegionHistoryState(variant=args.variant,retention=args.retention,
            transfer_ratio=ratio,adaptive=adaptive,query_chunk=128,capture=True)
        state.reset(frames,height,width,'cuda')
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            for layer in range(24):
                data=dense_inputs[layer]
                result=RegionHistoryAttention(originals[layer],state,layer).eval()(
                    data['x'].cuda(),None if data['pos'] is None else data['pos'].cuda())
                if not 10<=layer<=16:
                    torch.testing.assert_close(result.cpu(),data['projected'],atol=0,rtol=0)
        torch.cuda.synchronize();evaluate('controlled',name,state,time.perf_counter()-started)
    state=install_region_history_attention(model,variant=args.variant,retention=args.retention,
        query_chunk=128,capture=True)
    poses={}
    for name,ratio,adaptive in settings:
        state.transfer_ratio,state.adaptive=ratio,adaptive
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pose=model(images)['pose_enc'].float().cpu()
        torch.cuda.synchronize();evaluate('native',name,state,time.perf_counter()-started)
        poses[name]=pose.numpy()
    np.savez_compressed(out/'masks.npz',**saved)
    np.savez_compressed(out/'poses.npz',dense=dense_pose.numpy(),**poses)
    with (out/'layers.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sources=['saf3r/patch/region_history_attention.py','scripts/evaluate_region_history_two_frames.py']
    for path in sources:
        target=out/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/path).read_bytes())
    result=dict(input=meta['image_paths'],checkpoint=meta['checkpoint'],retention=args.retention,
        variant=args.variant,rows=rows,records=records,timings=timings,
        dense_capture_bitwise_equal=True,dense_layers_controlled_bitwise_equal=True,
        first_sparse_input_bitwise_equal=True,neighborhood='Chebyshev patch radius within the same frame',
        recall_target='Dense top10% PATCH Keys (207); excludes mandatory specials; fixed across10/20% budgets',
        matched_target='Dense top patch_budget PATCH Keys; same total budget with mandatory specials',
        fine='Fixed connected spatial cells,504 across2frames; Value hierarchy supplies unchanged global coarse17 groups',
        search='A: ~10fine/group; B: same parents then connected2–3fine/child. Boundary cells trimmed by numeric ID, not fine QK.',
        adaptive='floor(C*patch_budget) per Query; C=1-m80/coarse_count; fixed ratios are independent ablations',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--retention',type=float,choices=[.1,.2],required=True)
    p.add_argument('--variant',choices=['A','B'],required=True)
    run(p.parse_args())
