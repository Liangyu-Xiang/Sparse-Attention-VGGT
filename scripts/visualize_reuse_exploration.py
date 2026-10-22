"""Original-coordinate two-frame support masks and allocation diagnostic."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch


def run(root):
    data=torch.load(root/'supports.pt',map_location='cpu',weights_only=False)
    layers=json.loads((root/'layers.json').read_text())
    input_info=json.loads((root/'input.json').read_text())
    frames=input_info['shape'][0]
    if frames!=2:raise ValueError('This full-resolution diagnostic is limited to two frames')
    n=len(data[0]['canonical'])+len(data[0]['specials']);per=n//frames
    directory=root/'visualization';directory.mkdir(exist_ok=True)
    cmap=ListedColormap(['white','#0072B2','#009E73','#D55E00','#777777'])
    norm=BoundaryNorm(np.arange(-.5,5.5),5)
    arrays={}
    fig,axes=plt.subplots(6,4,figsize=(15,20),constrained_layout=True)
    for layer,ax in enumerate(axes.flat):
        info=data[layer];mask=np.zeros((n,n),dtype=np.uint8);mask[:,info['specials'].numpy()]=4
        cursor=0
        for window in info['windows']:
            for row,ids in enumerate(window['ids']):
                valid=ids>=0;chosen=ids[valid].long().numpy();real=window['genuine'][row][valid].numpy()
                # Initial diagnostic counter had shape[group,4], fixed before
                # quality runs. Summing recovers the exact packed history count.
                reused=int(window['reuse'][row].sum())
                kind=np.where(real,3,2).astype(np.uint8);kind[:reused]=1
                stop=min(n,cursor+32)
                mask[cursor:stop,chosen]=kind[None]
                cursor=stop
        assert cursor==n
        arrays[f'layer_{layer:02d}']=mask
        ax.imshow(mask,cmap=cmap,norm=norm,origin='upper',extent=(0,n,n,0),interpolation='nearest')
        ax.axhline(per,color='black',lw=.5);ax.axvline(per,color='black',lw=.5)
        ax.set_title(f'L{layer}',fontsize=10);ax.set_xticks([0,per,n]);ax.set_yticks([0,per,n])
    fig.legend(handles=[Patch(facecolor=cmap(i),label=label) for i,label in enumerate(
        ['Not computed','Reuse real Key','Population representative','Exploration exact Key','Special Key'])],
        loc='outside lower center',ncol=5)
    fig.suptitle('heads TWO-frame diagnostic: actual computed support in original Query/Key coordinates\nNo reordering, no interpolation; full-resolution categorical arrays saved separately')
    fig.savefig(directory/'all24_masks.png',dpi=140);plt.close(fig)
    np.savez_compressed(directory/'original_coordinate_masks.npz',**arrays)
    # Correct initial reporting counter using the verified support-count identity;
    # do not overwrite the preserved original diagnostic JSON or source snapshot.
    rep=np.array([r['representatives_mean'] for r in layers]);exact=np.array([r['exact_edges_mean'] for r in layers])
    total=np.array([r['counts_mean'] for r in layers]);budget=np.array([r['patch_budget'] for r in layers])
    reuse=total-rep-exact;unused=budget-total
    fig,ax=plt.subplots(figsize=(12,4.5),constrained_layout=True)
    ax.stackplot(np.arange(24),100*rep/budget,100*reuse/budget,100*exact/budget,100*unused/budget,
                 labels=['Representative','Reuse real edges','Exploration exact','Unused budget'],
                 colors=['#009E73','#0072B2','#D55E00','#CCCCCC'])
    ax.set_ylim(0,100);ax.set_xticks(range(24));ax.set_xlabel('Global layer, zero based')
    ax.set_ylabel('Share of patch budget (%)');ax.legend(loc='upper left',ncol=4,fontsize=8)
    ax.set_title('heads TWO frames: emergent allocation, NOT task-quality evidence')
    fig.savefig(directory/'allocation.png',dpi=160);plt.close(fig)
    (directory/'README.md').write_text('Original-coordinate categorical masks, two-frame diagnostic only.\n'
        'Reuse counting is recovered from captured exact per-group slot counts; the initial JSON logging bug is not concealed or overwritten.\n'
        'All six quality metrics and paired Dense speedup are N/A for this diagnostic.128-frame quality evaluation is a separate directory.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True)
    run(p.parse_args().input_dir)
