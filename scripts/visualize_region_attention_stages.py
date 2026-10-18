"""Observe actual selector locals at return, without changing any decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scripts.overlay_vggt_query_attention import build_model
from saf3r.utils.data_utils import load_and_process_images_vggt
from saf3r.patch.value_evolving_attention import token_layout
import saf3r.patch.region_history_attention as branch

ROOT=Path(__file__).resolve().parents[1]
STAGES=[
    ('history_in','1. Incoming history [edges]'),
    ('coarse_gate','2. Top50% coarse gate [regions; HISTORY ONLY]'),
    ('history_filtered','3. Filtered history [edges]'),
    ('history_covered_fine','4. History-covered fine cells [regions]'),
    ('unresolved_fine','5. Unresolved fine cells [regions]'),
    ('level1_selected','6. Level1 selected descendants [regions]'),
    ('level2_selected_groups','7. Level2 groups used, incl boundary [regions]'),
    ('selected_fine','8. Quota-selected fine cells [regions]'),
    ('new_representatives','9. One representative/new fine cell [edges]'),
    ('history_plus_new','10. History + new patch Keys [edges]'),
    ('final_with_specials','11. Final mask incl specials [edges]'),
    ('history_out','12. Outgoing history [edges]')]


def ids_mask(ids,size):
    mask=torch.zeros((len(ids),size),device=ids.device,dtype=torch.bool)
    r,c=(ids>=0).nonzero(as_tuple=True)
    mask[r,ids[r,c].long()]=True
    return mask


def selector_stages(local):
    labels,reps,parent,child=local['layout']
    final,retained,used,selected=[local[name] for name in ['final','retained','used','selected']]
    incoming=torch.zeros_like(final) if local['history'] is None else ids_mask(local['history'],len(labels))
    second=local['second']
    selected_children=torch.zeros((len(final),int(child.max())+1),device=final.device,dtype=torch.long)
    selected_children.scatter_add_(1,child.expand(len(final),-1),selected.long())
    stages=dict(history_in=incoming,coarse_gate=local['gate'][:,local['token_coarse']],
        history_filtered=retained,history_covered_fine=(used[:,labels]>0),
        unresolved_fine=local['eligible'][:,labels],level1_selected=second[:,labels],
        level2_selected_groups=((selected_children[:,child]>0)&second)[:,labels],
        selected_fine=selected[:,labels],new_representatives=final&~retained,history_plus_new=final)
    assert not bool((stages['history_filtered']&~stages['history_in']).any())
    assert not bool((stages['history_filtered']&~stages['coarse_gate']).any())
    assert not bool((stages['level1_selected']&~stages['unresolved_fine']).any())
    assert not bool((stages['selected_fine']&~stages['level2_selected_groups']).any())
    assert not bool((stages['selected_fine']&stages['history_covered_fine']).any())
    assert bool(((final&~retained).sum(1)==selected.sum(1)).all())
    return stages


def run(args):
    out=args.output_dir
    if (out/'result.json').exists():raise ValueError('Use fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=json.loads((ROOT/'outputs/global_attention/chess_gap10_layer_mean_all24/result.json').read_text())
    images=load_and_process_images_vggt(meta['image_paths']).cuda()
    model=build_model(Path(meta['checkpoint']),'cuda')
    state=branch.install_region_history_attention(model,variant='B',retention=args.retention,
        transfer_ratio=.5,query_chunk=128,capture=True)
    patches,specials=token_layout(2,28*37,5,'cuda')
    context={'layer':None};pieces={l:{name:[] for name,_ in STAGES} for l in range(10,17)}
    coarse_probs={l:[] for l in pieces};offsets={l:0 for l in pieces}
    original=branch.select_regions

    def observe(*positional,**kwargs):
        def callback(frame,event,result):
            if frame.f_code is original.__code__ and event=='return':
                local=frame.f_locals;layer=context['layer']
                stages=selector_stages(local)
                for name,mask in stages.items():
                    pieces[layer][name].append(np.packbits(mask.cpu().numpy(),axis=-1))
                coarse_probs[layer].append(local['coarse_prob'].cpu().numpy())
                offsets[layer]+=local['q'].shape[1]
        previous=sys.getprofile();sys.setprofile(callback)
        try:return original(*positional,**kwargs)
        finally:sys.setprofile(previous)
    handles=[]
    for layer,block in enumerate(model.aggregator.global_blocks):
        def before(module,inputs,l=layer):context['layer']=l
        handles.append(block.attn.register_forward_pre_hook(before))
        def after(module,inputs,output,l=layer):
            if l not in pieces:return
            ids=state.captured[l]['indices'].to(patches.device)
            full=ids_mask(ids,2082)
            pieces[l]['final_with_specials'].append(np.packbits(full.cpu().numpy(),axis=-1))
            hist=ids_mask(state.history,2072)
            pieces[l]['history_out'].append(np.packbits(hist.cpu().numpy(),axis=-1))
        handles.append(block.attn.register_forward_hook(after))
    branch.select_regions=observe
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pose=model(images)['pose_enc'].float().cpu().numpy()
    finally:
        branch.select_regions=original
        for handle in handles:handle.remove()
    baseline=ROOT/f'outputs/region_history_attention/chess_two_frames/budget{round(args.retention*100)}_B'
    masks=np.load(baseline/'masks.npz');poses=np.load(baseline/'poses.npz')
    assert np.array_equal(pose,poses['h50'])
    exact={};stats=[];patch_ids=patches.cpu().numpy()
    for layer in pieces:
        assert offsets[layer]==2082
        assert np.array_equal(state.captured[layer]['indices'].numpy(),masks[f'native_h50_L{layer}_indices'])
        for name,_ in STAGES:
            exact[f'L{layer}_{name}']=np.concatenate(pieces[layer][name])
        exact[f'L{layer}_coarse_probabilities']=np.concatenate(coarse_probs[layer])
        final=np.unpackbits(exact[f'L{layer}_final_with_specials'],axis=-1,count=2082).astype(bool)
        new=np.unpackbits(exact[f'L{layer}_history_plus_new'],axis=-1,count=2072).astype(bool)
        assert np.array_equal(final[:,patch_ids],new)
        assert np.array_equal(np.unpackbits(exact[f'L{layer}_history_out'],axis=-1,count=2072).astype(bool),
            np.unpackbits(exact[f'L{layer+1}_history_in'],axis=-1,count=2072).astype(bool)) if layer<16 and f'L{layer+1}_history_in' in exact else True
    # Validate the inter-layer handoff after every layer has been assembled.
    for layer in range(10,16):
        assert np.array_equal(exact[f'L{layer}_history_out'],exact[f'L{layer+1}_history_in'])
    np.savez_compressed(out/'exact_stages.npz',**exact)
    cmap=ListedColormap(['#eeeeee','#144f83'])
    for layer in pieces:
        stages={}
        for name,_ in STAGES:
            if name=='final_with_specials':
                full=np.unpackbits(exact[f'L{layer}_{name}'],axis=-1,count=2082).astype(bool)
            else:
                full=np.zeros((2082,2082),dtype=bool)
                full[:,patch_ids]=np.unpackbits(exact[f'L{layer}_{name}'],axis=-1,count=2072).astype(bool)
            stages[name]=full
            stats.append(dict(layer=layer,stage=name,fraction=float(full.mean())))
        for first,title in [(0,'filtering'),(6,'exploration')]:
            fig,axes=plt.subplots(2,3,figsize=(14,9.2),constrained_layout=True)
            for ax,(name,label) in zip(axes.flat,STAGES[first:first+6]):
                mask=stages[name]
                ax.imshow(mask,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
                ax.set_title(f'{label}\n{100*mask.mean():.2f}% original matrix positions',fontsize=10)
                ax.axhline(1040.5,color='cyan',lw=.5);ax.axvline(1040.5,color='cyan',lw=.5)
                ax.set_xticks([0,1041,2082]);ax.set_yticks([0,1041,2082]);ax.tick_params(labelsize=7)
            fig.suptitle(f'L{layer}; B; {args.retention:.0%}final budget;50%history; actual sparse forward\nRows=Query; columns=original Key order. REGION masks are NOT computed original-token QK.',fontsize=12)
            fig.savefig(out/f'L{layer}_{title}.png',dpi=170);plt.close(fig)
        # A single fixed Query in patch-image coordinates, preserving both frames.
        query=5+14*37+18
        fig,axes=plt.subplots(3,4,figsize=(15,8),constrained_layout=True)
        for ax,(name,label) in zip(axes.flat,STAGES):
            planes=stages[name][query,patch_ids].reshape(2,28,37)
            canvas=np.concatenate([planes[0],np.zeros((28,2),bool),planes[1]],axis=1)
            ax.imshow(canvas,cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
            ax.axvline(37.5,color='cyan',lw=1);ax.set_title(label,fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f'L{layer}; Query=frame0,row14,col18; left/right=Key frame0/frame1\nEach pixel is a patch; final specials omitted in this spatial-only view',fontsize=12)
        fig.savefig(out/f'L{layer}_one_query.png',dpi=170);plt.close(fig)
    result=dict(retention=args.retention,variant='B',transfer_ratio=.5,input=meta['image_paths'],
        stages=STAGES,statistics=stats,pose_bitwise_equal_to_previous_run=True,
        final_masks_bitwise_equal_to_previous_run=True,all_layers2082rows_captured=True,
        history_handoff_exact=True,
        storage='Packed last axis. All stages count2072patch Keys EXCEPT final_with_specials count2082original Keys. Coarse probabilities are float32.',
        coarse_gate='Filters history ONLY; unresolved exploration begins from ALL uncovered spatial fine cells, not just this gate.',
        level2='Union of children contributing selected fine cells, with boundary child whole; reconstructed from actual selection, not an additional QK pass.',
        semantics='Region membership is displayed on original Key coordinates, not falsely counted as individually computed real QK edges. Virtual pooled-Key probes have no single original-token location.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
            ['saf3r/patch/region_history_attention.py','scripts/visualize_region_attention_stages.py']})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Actual intermediate Attention branch masks','',
        'Chess frames0/10, Dense0–9/sparse10–16/Dense17–23; B;50%history. '
        'All final masks and pose are bitwise identical to the previous unobserved run. '
        'No decisions or scores were recomputed by a substitute selector: a Python return observer captured the actual selector locals.','',
        'Blue marks membership in original2082x2082matrix coordinates. Region panels mean a Key '
        'belongs to a region under consideration; they do NOT imply that its real QK was computed. '
        'Only final_with_specials is the actual original-token QK/AV execution support. '
        'Pooled virtual-Key probes have no single original-token position.','',
        'The coarse top50%gate filters HISTORY ONLY. Unresolved exploration is the complement '
        'of history-covered fine cells over the whole patch space. It is NOT a sequential subset '
        'of the initial coarse gate. Level2panel is the union of groups containing quota-selected '
        'fine cells, including the partially used boundary group.','',
        'Stages1–6 show history/filtering and level1selection; stages7–12 show level2selection, '
        'fine quota, representative collapse, final support and outgoing history.','']
    for layer in pieces:
        lines += [f'## Layer{layer}','',f'![filter](L{layer}_filtering.png)','',
            f'![explore](L{layer}_exploration.png)','',f'![one query](L{layer}_one_query.png)','']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps({key:value for key,value in result.items() if 'equal' in key or 'exact' in key}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--retention',type=float,choices=[.1,.2],required=True)
    run(p.parse_args())
