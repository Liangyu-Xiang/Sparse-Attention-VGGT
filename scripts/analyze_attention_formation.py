"""Dense head-mean attention evolution and next-layer candidate predictability."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]


def top_indices(a,k):
    # Stable token-index tie break; common ordering for nested 10/20/40% sets.
    return np.argsort(-a,axis=-1,kind='stable')[:,:k]


def recall(candidate,target,n):
    mask=np.zeros((len(candidate),n),dtype=bool)
    np.put_along_axis(mask,candidate,True,axis=1)
    return float(np.take_along_axis(mask,target,axis=1).mean())


def statistics(a):
    a=a.astype(np.float64)
    a/=a.sum(-1,keepdims=True)
    entropy=-(a*np.log(np.maximum(a,1e-300))).sum(-1)
    m=a.mean(0)
    cosine=(a@m)/(np.linalg.norm(a,axis=1)*np.linalg.norm(m))
    return dict(C=float(1-entropy.mean()/math.log(a.shape[1])),
                D=float(1-cosine.mean()))


def coarse_recall(scores,groups,target,fraction):
    counts=np.bincount(groups,minlength=scores.shape[1])
    nonempty=np.flatnonzero(counts)
    k=max(1,math.ceil(fraction*len(nonempty)))
    selected=nonempty[top_indices(scores[:,nonempty],k)]
    selected_mask=np.zeros_like(scores,dtype=bool)
    np.put_along_axis(selected_mask,selected,True,axis=1)
    hit=np.take_along_axis(selected_mask,groups[target],axis=1)
    density=(selected_mask@counts)/len(groups)
    per_query=hit.mean(1)
    # Random Key candidates of the SAME per-query sizes have this expected recall.
    return dict(recall=float(per_query.mean()),density=float(density.mean()),
                chance_corrected=float(np.mean((per_query-density)/np.maximum(1-density,1e-12))),
                selected_groups=k,nonempty_groups=len(nonempty))


def capture_groups(source,directory):
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.patch.layer_shared_av import LayerSharedAVState
    from saf3r.patch.value_clustering_gpu import stable_cluster_mean
    metadata=json.loads((source/'result.json').read_text())
    images=load_and_process_images_vggt(metadata['image_paths']).cuda()
    model=build_model(Path(metadata['checkpoint']),'cuda')
    state=LayerSharedAVState(fine_ratio=.15,coarse_size=128,cluster_iterations=2)
    state.reset(2,28,37)
    patches=torch.tensor([f*1041+5+j for f in range(2) for j in range(1036)],device='cuda')
    cache={};handles=[]
    def capture(layer):
        def hook(module,inputs,kwargs,output):
            x=inputs[0];pos=kwargs.get('pos',inputs[1] if len(inputs)>1 else None)
            q,k,v=module.qkv(x).reshape(1,2082,3,16,64).permute(2,0,3,1,4).unbind(0)
            q,k=module.q_norm(q),module.k_norm(k)
            if module.rope is not None: q,k=module.rope(q,pos),module.rope(k,pos)
            with torch.autocast('cuda',enabled=False):
                labels,reps,groups,_=state.update(v[0,:,patches].float().mean(0),layer)
                if 6<=layer<=16:
                    gids=groups[labels]
                    nc=int(groups.max())+1
                    pooled=stable_cluster_mean(k[0,:,patches].float().mean(0),gids,nc)
                    score=q[0,:,patches].float().mean(0)@pooled.T*module.scale
                    cache[f'g{layer}']=gids.cpu().numpy()
                    cache[f'qk{layer}']=score.cpu().numpy()
            print('Value groups on Dense trajectory',layer,flush=True)
        return hook
    for i,block in enumerate(model.aggregator.global_blocks):
        handles.append(block.attn.register_forward_hook(capture(i),with_kwargs=True))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=model(images)['pose_enc'].float().cpu()
    for h in handles:h.remove()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        b=model(images)['pose_enc'].float().cpu()
    assert torch.equal(a,b)
    np.savez_compressed(directory/'dense_value_groups.npz',**cache)
    return cache


def run(source,directory):
    directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cache_path=directory/'dense_value_groups.npz'
    cache=dict(np.load(cache_path)) if cache_path.exists() else capture_groups(source,directory)
    patches=np.array([f*1041+5+j for f in range(2) for j in range(1036)])
    n=len(patches);k=math.floor(.1*n)
    matrices={};stats={};fullstats={};rankings={};fullrank={};hashes={}
    for l in range(6,17):
        path=source/f'layer_{l:02d}_full.npy'
        full=np.load(path)
        assert full.shape==(2082,2082) and np.allclose(full.sum(1),1,atol=1e-6)
        a=full[np.ix_(patches,patches)].copy();a/=a.sum(1,keepdims=True)
        matrices[l]=a;stats[l]=statistics(a);fullstats[l]=statistics(full)
        rankings[l]=top_indices(a,4*k)
        fullrank[l]=top_indices(full,4*math.floor(.1*2082))
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    rows=[];fullrows=[]
    for l in range(6,16):
        a=matrices[l];target=rankings[l+1][:,:k];groups=cache[f'g{l}']
        nc=int(groups.max())+1
        membership=np.eye(nc,dtype=np.float32)[groups]
        # Exact group mass from previous Dense attention: no future-layer information.
        mass=a@membership
        row=dict(layer_pair=f'{l}→{l+1}',source_layer=l,C_prev=stats[l]['C'],C_next=stats[l+1]['C'],
                 D_prev=stats[l]['D'],D_next=stats[l+1]['D'])
        for r in (1,2,4):
            value=recall(rankings[l][:,:r*k],target,n)
            row[f'fine_r{r}']=value
            row[f'fine_r{r}_chance']=r*k/n
        # Query-independent Key saliency control: can high recall be explained by shared sinks?
        global_keys=top_indices(a.mean(0)[None],2*k)
        row['key_saliency_r2']=recall(np.repeat(global_keys,n,0),target,n)
        for fraction in (.1,.2,.4):
            for name,score in [('dense_mass',mass),('current_mean_qk',cache[f'qk{l}'])]:
                metric=coarse_recall(score,groups,target,fraction)
                for key,value in metric.items():row[f'{name}_{int(fraction*100)}_{key}']=value
        rows.append(row)
        fk=math.floor(.1*2082)
        fullrows.append(dict(layer_pair=row['layer_pair'],C_prev=fullstats[l]['C'],C_next=fullstats[l+1]['C'],
            **{f'fine_r{r}':recall(fullrank[l][:,:r*fk],fullrank[l+1][:,:fk],2082) for r in (1,2,4)}))
    for filename,data in [('patch_metrics',rows),('all_token_metrics',fullrows)]:
        with (directory/(filename+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    x=np.arange(6,17);pairs=[r['layer_pair'] for r in rows]
    axes[0,0].plot(x,[stats[l]['C'] for l in x],'o-',label='Concentration C')
    axes[0,0].plot(x,[stats[l]['D'] for l in x],'s-',label='Query specificity D')
    axes[0,0].set_title('Patch-only conditional head-mean attention');axes[0,0].set_xlabel('Global layer')
    for r in (1,2,4):
        axes[0,1].plot(pairs,[100*a[f'fine_r{r}'] for a in rows],'o-',label=f'Previous top {r*10}%')
        axes[0,1].axhline(100*r*k/n,color='gray',ls=':',alpha=.5)
    axes[0,1].set_title('Predict next-layer top 10%: exact Key candidates')
    for name,label in [('dense_mass','Dense group mass'),('current_mean_qk','Current mean-Q/K sensing')]:
        axes[1,0].plot(pairs,[100*a[f'{name}_20_recall'] for a in rows],'o-',label=label+' recall')
        axes[1,0].plot(pairs,[100*a[f'{name}_20_density'] for a in rows],'--',label=label+' Key coverage')
    axes[1,0].set_title('Top 20% nonempty previous-layer Value groups')
    axes[1,1].plot(pairs,[100*a['fine_r2'] for a in rows],'o-',label='Per-query previous top 20%')
    axes[1,1].plot(pairs,[100*a['key_saliency_r2'] for a in rows],'s-',label='Query-independent previous top 20%')
    axes[1,1].set_title('Is predictability more than shared Key saliency?')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.set_ylabel('Percent');ax.tick_params(axis='x',rotation=45);ax.set_ylim(0,100)
    fig.savefig(directory/'formation_curves.png',dpi=180);plt.close(fig)
    metadata=dict(input_source=str(source),source_matrix_sha256=hashes,
        head_aggregation='mean of 16 per-head softmax matrices',
        primary='All2072 patch Queries and Keys; remove10 specials, renormalize each row',
        secondary='All2082 tokens, no exclusions, K=208',
        K=k,N=n,candidate_multipliers=[1,2,4],
        coarse='Previous-layer current Value fine/coarse groups on the unmodified Dense trajectory; '
               'rank by exact summed Dense mass OR actual mean-Q/K pooled-Key score; '
               'top ceil(x * nonempty groups), x=10/20/40%; report actual Key coverage',
        hierarchy=dict(fine_ratio=.15,coarse_size=128,cluster_iterations=2),
        group_instrumentation_pose_identical=True,
        layers=stats,transitions=rows,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory/'result.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# Attention structure formation: Dense Chess frames0 and10','',
        'No sparse model changes. Head-mean **probabilities**, not softmax of mean QK. '
        'Main table excludes special Queries/Keys and conditionally normalizes patch attention; '
        'all-token results are separately recorded in all_token_metrics.csv. '
        'All2072 patch Queries, no Query sampling. K=207; candidate sizes207/414/828.', '',
        '| Pair | C previous | C next | Top10 overlap % | Top20 recall % | Top40 recall % | Coarse20 recall % | Coarse20 Key coverage % | Actual QK coarse20 recall % | Actual QK Key coverage % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in rows:
        lines.append(f"| {a['layer_pair']} | {a['C_prev']:.4f} | {a['C_next']:.4f} | "+
            ' | '.join(f"{100*a[key]:.2f}" for key in ('fine_r1','fine_r2','fine_r4','dense_mass_20_recall',
                'dense_mass_20_density','current_mean_qk_20_recall','current_mean_qk_20_density'))+' |')
    lines+=['','Coarse groups are computed independently at each source layer, but BOTH the source '
        'candidate and next-layer target are mapped into that SAME source grouping. No future '
        'grouping or future attention is used to rank candidates. Dense mass is a diagnostic '
        'reference, not the cheap sensing operator. High coarse recall is not evidence of '
        'useful localization without inspecting its actual Key coverage and the random baseline.', '',
        'Chance fine recall is9.99/19.98/39.96% for10/20/40% candidates. '
        'Query-independent Key-saliency recall, query specificity D, coarse10/40%, and '
        'chance-corrected coarse recall are included in patch_metrics.csv.', '',
        'This two-frame single-scene diagnostic cannot establish a universal bootstrap layer, '
        'or guarantee that a cheap sparse sensing/tracking mechanism attains Dense-oracle predictability.','']
    (directory/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/global_attention/chess_gap10_layer_mean_all24')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/attention_formation/chess_gap10')
    args=p.parse_args();run(args.source,args.output_dir)
