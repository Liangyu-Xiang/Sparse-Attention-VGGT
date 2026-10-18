"""Same-Dense-input diagnostics: four failure hypotheses, no speed claims."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from saf3r.patch.region_history_attention import (
    prepare_qkv,spatial_search_layout,coarse_probabilities,transfer_history)
from saf3r.patch.region_history_corrections import (
    select_regions_corrected,materialize_cells,influence_scores)
from saf3r.patch.layer_shared_av import LayerSharedAVState
from saf3r.patch.value_clustering_gpu import stable_cluster_mean
from saf3r.patch.value_evolving_attention import token_layout
from saf3r.patch.residual_history_av import pack_mask,qk_logits,logits_av
from scripts.overlay_vggt_query_attention import build_model
from saf3r.utils.data_utils import load_and_process_images_vggt

ROOT=Path(__file__).resolve().parents[1]
CASES={
    'legacy':('gate','mean','representative','B'),
    'hard_history':('hard','mean','representative','B'),
    'size_bias_only':('gate','size_bias','representative','B'),
    'hard_size_rep':('hard','size_bias','representative','B'),
    'exact_mass':('gate','exact_mass','representative','B'),
    'hard_exact_rep':('hard','exact_mass','representative','B'),
    'corrected_B':('hard','size_bias','local_top1','B'),
    'corrected_A':('hard','size_bias','local_top1','A'),
    'full_oracle':('hard','exact_mass','best_member','B'),
}
ALL_CASES=list(CASES)+['best_member_frozen','exact_mass_best_frozen','uniform','dense_topk','influence_topk']


def to_keys(patch_ids,patches,specials):
    return torch.cat((patches[patch_ids.clamp_min(0)].masked_fill(patch_ids<0,-1),
        specials.expand(len(patch_ids),-1)),1)


def run(args):
    out=args.output_dir
    if (out/'result.json').exists():raise ValueError('Use fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=json.loads((ROOT/'outputs/global_attention/chess_gap10_layer_mean_all24/result.json').read_text())
    images=load_and_process_images_vggt(meta['image_paths']).cuda()
    model=build_model(Path(meta['checkpoint']),'cuda')
    patches,specials=token_layout(2,28*37,5,'cuda');n=2082;npats=len(patches)
    value=LayerSharedAVState();value.reset(2,28,37)
    features={};handles=[]
    for layer,block in enumerate(model.aggregator.global_blocks):
        def capture(module,inputs,kwargs,output,l=layer):
            pos=kwargs.get('pos',inputs[1] if len(inputs)>1 else None)
            q,k,v=prepare_qkv(module,inputs[0],pos)
            with torch.autocast('cuda',enabled=False):
                labels,reps,mapping,_=value.update(v[:,patches].float().mean(0),l)
                if 9<=l<=16:
                    pooled=torch.stack([stable_cluster_mean(kh[reps],mapping,int(mapping.max())+1)
                        for kh in k[:,patches].float()])
                    features[l]=dict(q=q.cpu(),k=k.cpu(),v=v.cpu(),pooled=pooled.cpu(),token_coarse=mapping[labels].cpu())
        handles.append(block.attn.register_forward_hook(capture,with_kwargs=True))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pose=model(images)['pose_enc'].cpu()
    for handle in handles:handle.remove()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        assert torch.equal(pose,model(images)['pose_enc'].cpu())
    del model
    layout=spatial_search_layout(2,28,37,'cuda');budget=int(args.retention*n)-len(specials)
    prior=ROOT/f'outputs/region_history_attention/chess_two_frames/budget{round(100*args.retention)}_B'
    previous=np.load(prior/'masks.npz');previous_json=json.loads((prior/'result.json').read_text())
    histories={name:None for name in CASES};rows=[];saved={};legacy_equal=True
    for layer in range(9,17):
        data=features[layer];q,k,v=[data[key].cuda() for key in ['q','k','v']]
        pooled=data['pooled'].cuda();token_coarse=data['token_coarse'].cuda()
        kpatch=k[:,patches].float().contiguous()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            dense_reference=F.scaled_dot_product_attention(q[None],k[None],v[None])[0].float()
        accum={name:dict(error=torch.zeros(q.shape[0],device='cuda'),den=torch.zeros(q.shape[0],device='cuda'),
            top10_hits=0,matched_hits=0,mass=0.,patch_mass=0.,keys=0,retained=0,terminal_probes=0)
            for name in ALL_CASES}
        layer_ids={name:[] for name in ALL_CASES};new_hist={name:[] for name in CASES}
        seed_parts=[]
        for start in range(0,n,128):
            end=min(n,start+128);rq=q[:,start:end];batch=end-start
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                logits=(rq.float()@k.float().transpose(-1,-2))*q.shape[-1]**-.5
                a=logits.softmax(-1);mean=a.mean(0);patch_mean=mean[:,patches]
                p=coarse_probabilities(rq.float(),pooled)
                ranking=patch_mean.argsort(dim=1,descending=True,stable=True)
                if layer==9:
                    ids=ranking[:,:budget]
                    seed_parts.append(transfer_history(patch_mean.gather(1,ids),ids,p,budget,.5))
                    continue
                target10=ranking[:,:int(.1*npats)];matched=ranking[:,:budget]
                computed={};details_by_case={}
                for name,(history_policy,scoring,terminal,variant) in CASES.items():
                    details={}
                    ids,stat=select_regions_corrected(rq.float(),kpatch,layout,p,token_coarse,
                        histories[name][start:end],budget,variant,history_policy=history_policy,
                        group_scoring=scoring,terminal=terminal,
                        diagnostic_logits=logits[:,:,patches] if scoring=='exact_mass' else None,
                        diagnostic_mean_weights=patch_mean if terminal=='best_member' else None,details=details)
                    computed[name]=(ids,stat);details_by_case[name]=details
                for name,source in [('best_member_frozen','legacy'),('exact_mass_best_frozen','exact_mass')]:
                    details=details_by_case[source]
                    mask,_=materialize_cells(rq.float(),kpatch,layout[0],layout[1],
                        details['selected_fine'],details['retained'],'best_member',patch_mean)
                    assert torch.equal(mask.sum(1),computed[source][0].ge(0).sum(1))
                    computed[name]=(pack_mask(mask),computed[source][1])
                uniform=(torch.arange(budget,device='cuda')*npats//budget).expand(batch,-1)
                computed['uniform']=(uniform,None)
                computed['dense_topk']=(matched,None)
                exact_output=a@v.float()
                influence=influence_scores(a,v.float(),exact_output)[:,patches]
                computed['influence_topk']=(influence.argsort(dim=1,descending=True,stable=True)[:,:budget],None)
                for name,(ids,stat) in computed.items():
                    keys=to_keys(ids,patches,specials)
                    z=qk_logits(rq.to(v.dtype),k.to(v.dtype).contiguous(),keys,q.shape[-1]**-.5)
                    actual=logits_av(z,v,keys).float();reference=dense_reference[:,start:end]
                    record=accum[name]
                    record['error']+=(actual-reference).square().sum((1,2))
                    record['den']+=reference.square().sum((1,2))
                    mask=torch.zeros((batch,npats),device='cuda',dtype=torch.bool)
                    rr,cc=ids.ge(0).nonzero(as_tuple=True);mask[rr,ids[rr,cc]]=True
                    record['top10_hits']+=int(mask.gather(1,target10).sum())
                    record['matched_hits']+=int(mask.gather(1,matched).sum())
                    record['patch_mass']+=float((patch_mean*mask).sum())
                    record['mass']+=float((patch_mean*mask).sum()+mean[:,specials].sum())
                    record['keys']+=int(keys.ge(0).sum())
                    if stat is not None:
                        record['retained']+=int(stat['history_retained'].sum())
                        record['terminal_probes']+=int(stat['terminal_member_qk'].sum())
                    assert bool((mask.sum(1)==budget).all())
                    layer_ids[name].append(keys.cpu().numpy().astype(np.int32))
                    if name in CASES:
                        sparse_mean=z.softmax(-1).mean(0)[:,:ids.shape[1]]
                        new_hist[name].append(transfer_history(sparse_mean,ids,p,budget,.5))
        if layer==9:
            seed=torch.cat(seed_parts)
            histories={name:seed.clone() for name in CASES}
            continue
        for name in ALL_CASES:
            record=accum[name];err=record['error'].sum();den=record['den'].sum()
            row=dict(layer=layer,method=name,retention=args.retention,
                relative_output_error=float((err/den).sqrt()),mean_head_nmse=float((record['error']/record['den']).mean()),
                recall_top10=record['top10_hits']/(n*int(.1*npats)),matched_budget_recall=record['matched_hits']/(n*budget),
                retained_dense_mass=record['mass']/n,retained_patch_mass=record['patch_mass']/n,
                density=record['keys']/n**2,history_retained=record['retained']/n,
                terminal_member_qk_per_query=record['terminal_probes']/n,
                uses_dense_oracle=name in ['exact_mass','hard_exact_rep','full_oracle','best_member_frozen',
                    'exact_mass_best_frozen','dense_topk','influence_topk'])
            rows.append(row)
            saved[f'{name}_L{layer}']=np.concatenate(layer_ids[name])
            if name=='legacy':
                assert np.array_equal(saved[f'{name}_L{layer}'],previous[f'controlled_h50_L{layer}_indices'])
                old=next(r for r in previous_json['rows'] if r['scope']=='controlled' and r['setting']=='h50' and r['layer']==layer)
                assert abs(row['relative_output_error']-old['local_output_error'])<1e-5
        histories={name:torch.cat(new_hist[name]) for name in CASES}
        print('layer',layer,'validated legacy and completed14methods',flush=True)
    np.savez_compressed(out/'masks.npz',**saved)
    with (out/'layers.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    sources=['saf3r/patch/region_history_attention.py','saf3r/patch/region_history_corrections.py',
             'scripts/diagnose_region_history_oracles.py']
    for source in sources:
        target=out/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/source).read_bytes())
    result=dict(retention=args.retention,rows=rows,input=meta['image_paths'],checkpoint=meta['checkpoint'],
        dense_capture_bitwise_equal=True,legacy_masks_bitwise_equal=True,legacy_output_error_reproduced=True,
        scope='All layers receive original Dense inputs; each recursive method advances its own history at50%transfer.',
        frozen='Best-member-frozen reuses EXACT retained history and selected fine regions from the named source at each layer; it does not alter later source histories.',
        exact_mass='Per-head logsumexp of actual eligible member logits, normalized over all eligible groups, then head mean. Dense diagnostic only.',
        influence='Exact single-edge-deletion influence using FP32 Dense probabilities/output; shared mean-head ranking is NOT an optimal multi-edge support oracle.',
        error='Relative Frobenius norm before output projection, same BF16 Dense SDPA reference across methods. Mean-head NMSE is a DIFFERENT aggregation.',
        caveats=['No additive error decomposition: effects interact.','No128-frame run, speedup claim or dataset accuracy claim.',
                 'Original mean-Q/K criticism does not describe the current per-head-softmax-then-mean implementation.'],
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retention',type=float,choices=[.1,.2],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())
