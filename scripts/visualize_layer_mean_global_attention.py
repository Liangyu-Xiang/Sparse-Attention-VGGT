"""All 24 Dense VGGT global layers: mean of per-head softmax, not mean QK."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from torch.nn import functional as F

from scripts.overlay_vggt_query_attention import build_model
from scripts.visualize_vggt_global_attention import _save_input_frames
from saf3r.utils.data_utils import load_and_process_images_vggt

ROOT = Path(__file__).resolve().parents[1]


def run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'result.json').exists():
        raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = [ROOT/'workspace/benchmark_dataset/7scenes/7Scenes/chess/seq-01'/
             f'frame-{i:06d}.color.png' for i in (0,10)]
    _save_input_frames(paths, directory/'input_frames.png')
    images = load_and_process_images_vggt([str(p) for p in paths]).cuda()
    checkpoint = ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'
    model = build_model(checkpoint, 'cuda')
    pooled, records, handles = [], [], []

    def capture(layer):
        def hook(module, inputs, kwargs, output):
            x = inputs[0]
            pos = kwargs.get('pos', inputs[1] if len(inputs)>1 else None)
            b,n,c = x.shape
            q,k,v = module.qkv(x).reshape(b,n,3,module.num_heads,module.head_dim).permute(2,0,3,1,4).unbind(0)
            q,k = module.q_norm(q),module.k_norm(k)
            if module.rope is not None:
                q,k = module.rope(q,pos),module.rope(k,pos)
            # Match BF16 autocast operands entering the production Dense SDPA.
            q,k = q.to(v.dtype),k.to(v.dtype)
            with torch.autocast('cuda',enabled=False):
                average = torch.zeros(n,n,device=x.device,dtype=torch.float32)
                for head in range(module.num_heads):
                    weights = ((q[0,head].float()@k[0,head].float().T)*module.scale).softmax(-1)
                    average.add_(weights/module.num_heads)
                error = float((average.sum(-1)-1).abs().max())
                if error > 1e-5 or not torch.isfinite(average).all():
                    raise ValueError('Invalid attention probability matrix')
                # Rendering only: pool along both axes; saved .npy is full N x N.
                view = F.adaptive_avg_pool2d(average[None,None],(512,512))[0,0].cpu().numpy()
                matrix = average.cpu().numpy()
            np.save(directory/f'layer_{layer:02d}_full.npy',matrix)
            half = n//2
            specials = np.array([f*half+j for f in range(2) for j in range(5)])
            patches = np.setdiff1d(np.arange(n),specials)
            records.append(dict(layer=layer,shape=list(matrix.shape),row_sum_max_error=error,
                mean_special_key_mass=float(matrix[patches][:,specials].sum(1).mean()),
                frame_key_mass=[[float(matrix[f*half+5:(f+1)*half,g*half:(g+1)*half].sum(1).mean())
                                 for g in range(2)] for f in range(2)]))
            pooled.append(view)
            print('captured',layer,'shape',matrix.shape,'row_error',error,flush=True)
        return hook

    for i,block in enumerate(model.aggregator.global_blocks):
        handles.append(block.attn.register_forward_hook(capture(i),with_kwargs=True))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        captured_pose = model(images)['pose_enc'].float().cpu()
    for handle in handles: handle.remove()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        reference_pose = model(images)['pose_enc'].float().cpu()
    assert torch.equal(captured_pose,reference_pose), 'Instrumentation changed model output'
    assert len(pooled)==len(records)==24
    np.savez_compressed(directory/'rendered_512.npz',attention=np.stack(pooled))

    # One fixed logarithmic scale across all layers, no per-layer normalization.
    norm = LogNorm(vmin=1e-6,vmax=.1)
    def panel(ax,layer):
        im = ax.imshow(pooled[layer],origin='upper',extent=(0,2082,2082,0),
            cmap='magma',norm=norm,interpolation='nearest',rasterized=True)
        ax.axhline(1041,color='cyan',lw=.55,alpha=.8)
        ax.axvline(1041,color='cyan',lw=.55,alpha=.8)
        ax.set_xticks([0,1041,2082]); ax.set_yticks([0,1041,2082])
        ax.tick_params(labelsize=7)
        ax.set_title(f'L{layer}',fontsize=12)
        return im
    for start,end,name in [(0,24,'all_layers'),(0,10,'early_0_9'),(10,17,'middle_10_16'),(17,24,'late_17_23')]:
        count=end-start;cols=4 if count==24 else 5 if count==10 else 4
        rows=(count+cols-1)//cols
        fig,axes=plt.subplots(rows,cols,figsize=(cols*3.1,rows*3),squeeze=False,constrained_layout=True)
        for i,ax in enumerate(axes.flat):
            if start+i<end: im=panel(ax,start+i)
            else: ax.set_visible(False)
        fig.suptitle('Dense VGGT: mean of 16 head-wise softmax matrices\nChess frames 0 & 10 | rows=Query, columns=Key | cyan=frame boundary',fontsize=12)
        fig.colorbar(im,ax=list(axes.flat),shrink=.7,label='Attention probability (fixed log scale)')
        fig.savefig(directory/(name+'.png'),dpi=180)
        plt.close(fig)
    for layer in range(24):
        fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
        im=panel(ax,layer);ax.set_xlabel('Key token');ax.set_ylabel('Query token')
        fig.colorbar(im,ax=ax,label='Attention probability (fixed log scale)')
        fig.savefig(directory/f'layer_{layer:02d}.png',dpi=220)
        plt.close(fig)
    metadata=dict(model='unmodified Dense VGGT',checkpoint=str(checkpoint),
        image_paths=[str(p) for p in paths],image_shape=list(images.shape),
        definition='mean_h softmax((Q_h K_h^T)/sqrt(d)); BF16 SDPA operands, FP32 explicit scores/softmax',
        heads=16,layers=records,token_order='frame0 [5 special,1036 patch], frame1 [5 special,1036 patch]',
        matrix_shape=[2082,2082],rendering='512x512 adaptive average pooling only for display',
        color_scale=dict(type='log',vmin=1e-6,vmax=.1),
        instrumented_pose_bitwise_equal=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (directory/'result.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/global_attention/chess_gap10_layer_mean_all24')
    run(parser.parse_args().output_dir)
