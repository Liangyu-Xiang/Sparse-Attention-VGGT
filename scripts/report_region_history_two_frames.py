"""Report all middle-only spatial Attention sweeps and original-coordinate masks."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LogNorm
import numpy as np
import torch

from scripts.evaluate_region_history_two_frames import recalls,mask_from_ids

ROOT=Path(__file__).resolve().parents[1]


def run(root):
    torch.set_num_threads(4)
    rows=[];summary=[];results={}
    for budget in [10,20]:
        for variant in ['A','B']:
            path=root/f'budget{budget}_{variant}'
            result=json.loads((path/'result.json').read_text());results[budget,variant]=result
            assert result['dense_capture_bitwise_equal'] and result['dense_layers_controlled_bitwise_equal']
            assert result['first_sparse_input_bitwise_equal']
            assert len(result['rows'])==70
            rows.extend(result['rows'])
            for scope in ['controlled','native']:
                for setting in ['h0','h25','h50','h100','adaptive']:
                    subset=[r for r in result['rows'] if r['scope']==scope and r['setting']==setting]
                    summary.append(dict(budget=budget,variant=variant,scope=scope,setting=setting,
                        **{key:mean(r[key] for r in subset) for key in subset[0]
                           if key not in ['scope','setting','retention','variant','layer']}))
    assert len({tuple(r['input']) for r in results.values()})==1
    assert len({json.dumps(r['source_sha256'],sort_keys=True) for r in results.values()})==1
    out=root/'report';out.mkdir(exist_ok=True)
    for name,data in [('all_layers',rows),('settings',summary)]:
        with (out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    baseline=[]
    patches=torch.tensor([f*1041+5+j for f in range(2) for j in range(1036)],device='cuda')
    for budget in [10,20]:
        masks=np.load(root/f'budget{budget}_A/masks.npz')
        count=masks['dense_L10_matched_patch_indices'].shape[1]
        generator=torch.Generator(device='cuda').manual_seed(771)
        random_ids=torch.rand((2082,2072),device='cuda',generator=generator).topk(count,dim=1).indices
        random_mask=mask_from_ids(random_ids,2072)
        for layer in range(10,17):
            targets=torch.from_numpy(masks[f'dense_L{layer}_top10_patch_indices'].astype(np.int64)).cuda()
            baseline.append(dict(budget=budget,layer=layer,**recalls(random_mask,targets,2,28,37)))
    with (out/'random_neighborhood_baseline.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(baseline[0]));writer.writeheader();writer.writerows(baseline)
    # Primary plots are fixed50%history, A and B, both final budgets.
    diff_cmap=ListedColormap(['#eeeeee','#39a65a','#df4545','#357ad7'])
    norm=LogNorm(1e-6,.1);color=plt.get_cmap('magma').copy();color.set_bad('#dddddd')
    image_names=[]
    for budget in [10,20]:
        for variant in ['A','B']:
            masks=np.load(root/f'budget{budget}_{variant}/masks.npz')
            for layers in [list(range(10,14)),list(range(14,17))]:
                fig,axes=plt.subplots(len(layers),3,figsize=(12,3.5*len(layers)),constrained_layout=True)
                for row,layer in enumerate(layers):
                    target_ids=torch.from_numpy(masks[f'dense_L{layer}_matched_patch_indices'].astype(np.int64)).cuda()
                    target=mask_from_ids(patches[target_ids],2082).cpu().numpy()
                    for f in range(2):target[:,f*1041:f*1041+5]=True
                    sparse=mask_from_ids(torch.from_numpy(masks[f'native_h50_L{layer}_indices'].astype(np.int64)).cuda(),2082).cpu().numpy()
                    weights=np.load(ROOT/f'outputs/global_attention/chess_gap10_layer_mean_all24/layer_{layer:02d}_full.npy')
                    diff=np.zeros_like(target,dtype=np.uint8)
                    diff[target&sparse]=1;diff[target&~sparse]=2;diff[~target&sparse]=3
                    for col,(mask,title) in enumerate([(target,'Matched-budget Dense Top-K'),(sparse,'Actual sparse Mask')]):
                        axes[row,col].imshow(np.ma.array(weights,mask=~mask),cmap=color,norm=norm,interpolation='nearest')
                        axes[row,col].set_title(f'L{layer}: {title}',fontsize=10)
                    axes[row,2].imshow(diff,cmap=diff_cmap,vmin=0,vmax=3,interpolation='nearest')
                    axes[row,2].set_title('Green=both; red=missed; blue=extra',fontsize=9)
                    for ax in axes[row]:
                        ax.axhline(1040.5,color='cyan',lw=.4);ax.axvline(1040.5,color='cyan',lw=.4)
                        ax.set_xticks([0,1041,2082]);ax.set_yticks([0,1041,2082]);ax.tick_params(labelsize=7)
                fig.suptitle(f'Native middle-only sparse; {budget}%final budget; {variant};50%history\nOriginal Dense weights under masks; original token order; specials inside budget',fontsize=11)
                filename=f'budget{budget}_{variant}_L{layers[0]}_{layers[-1]}.png'
                fig.savefig(out/filename,dpi=160);plt.close(fig);image_names.append(filename)
    lines=['# Dense / region Attention / Dense: two-frame comparison','',
        'Chess frames0/10; Dense0–9, sparse10–16, Dense17–23. No Value support edges. '
        'Final10%/20%budgets include all specials. Results below average7middle layers, '
        'not a dataset accuracy result. Diagnostic runtimes are not speed benchmarks.','',
        'Primary table: fixed50%history. Exact/nearby recall targets the SAME Dense '
        'top10%patch Keys for both budgets (207targets/Query). Output error is relative '
        'Frobenius error before the attention output projection.','',
        '| Budget | Search | Scope | Exact % | r1 % | r2 % | r4 % | Local output error % | Trajectory error % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def line(r):
        return f"| {r['budget']}% | {r['variant']} | {r['scope']} | "+' | '.join(f'{100*r[k]:.2f}' for k in
            ['recall_r0','recall_r1','recall_r2','recall_r4','local_output_error','trajectory_output_error'])+' |'
    lines.extend(line(r) for r in summary if r['setting']=='h50')
    lines+=['','## History ablation: same Dense inputs','',
        '| Budget | Search | Transfer | Exact % | r2 % | Output error % |',
        '|---|---|---|---:|---:|---:|']
    for r in summary:
        if r['scope']=='controlled':lines.append(f"| {r['budget']}% | {r['variant']} | {r['setting']} | {100*r['recall_r0']:.2f} | {100*r['recall_r2']:.2f} | {100*r['local_output_error']:.2f} |")
    lines+=['','## Per-layer primary A,50%history (controlled)','',
        '| Layer |10%exact |10%r2 |10%output error |20%exact |20%r2 |20%output error |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for layer in range(10,17):
        selected=[next(r for r in rows if r['scope']=='controlled' and r['setting']=='h50'
            and r['variant']=='A' and r['retention']==budget/100 and r['layer']==layer) for budget in [10,20]]
        lines.append(f'| {layer} | '+' | '.join(f'{100*r[k]:.2f}' for r in selected
            for k in ['recall_r0','recall_r2','local_output_error'])+' |')
    lines+=['','## Random-support sanity baseline','',
        'A uniformly random support with the SAME patch budget, without crossing frame boundaries in dilation. '
        'Large neighborhood recall is not meaningful without this baseline.','',
        '| Budget | Exact % | r1 % | r2 % | r4 % |','|---|---:|---:|---:|---:|']
    for budget in [10,20]:
        values=[r for r in baseline if r['budget']==budget]
        lines.append(f'| {budget}% | '+' | '.join(f'{100*mean(r[k] for r in values):.2f}'
            for k in ['recall_r0','recall_r1','recall_r2','recall_r4'])+' |')
    lines+=['','## Exact masks in original Dense matrix coordinates','',
        'Native masks come from the actual sparse trajectory; original Dense colors are a backdrop, '
        'not native sparse attention weights. The matched-budget Dense mask contains all specials '
        'plus198/406highest-probability patch Keys. No head Q/K averaging.']
    lines.extend(f'\n![{name}]({name})' for name in image_names)
    lines+=['','All per-layer/scenario results: all_layers.csv; averaged settings: settings.csv. '
        'Exact Key indices, not downsampled masks, are saved in each run masks.npz. '
        'No128-frame7Scenes accuracy or steady-state performance claim is made.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(settings=summary,random_baseline=baseline),indent=2)+'\n')
    print(json.dumps([r for r in summary if r['setting']=='h50'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    run(p.parse_args().directory)
