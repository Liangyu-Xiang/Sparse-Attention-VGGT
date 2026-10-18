"""Small native GPU validation of the non-oracle corrected installer."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from scripts.overlay_vggt_query_attention import build_model
from scripts.evaluate_region_history_two_frames import dense_weights,mask_from_ids
from saf3r.utils.data_utils import load_and_process_images_vggt
from saf3r.patch.region_history_attention import prepare_qkv
from saf3r.patch.region_history_corrections import install_corrected_region_history_attention
from saf3r.patch.value_evolving_attention import token_layout

ROOT=Path(__file__).resolve().parents[1]


def run(out):
    if (out/'result.json').exists():raise ValueError('Use fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=json.loads((ROOT/'outputs/global_attention/chess_gap10_layer_mean_all24/result.json').read_text())
    images=load_and_process_images_vggt(meta['image_paths']).cuda()
    model=build_model(Path(meta['checkpoint']),'cuda')
    dense_qk={};handles=[]
    for layer in range(10,17):
        def capture(module,inputs,kwargs,output,l=layer):
            pos=kwargs.get('pos',inputs[1] if len(inputs)>1 else None)
            q,k,_=prepare_qkv(module,inputs[0],pos);dense_qk[l]=(q.cpu(),k.cpu())
        handles.append(model.aggregator.global_blocks[layer].attn.register_forward_hook(capture,with_kwargs=True))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
    for handle in handles:handle.remove()
    state=install_corrected_region_history_attention(model,retention=.1,transfer_ratio=.5,query_chunk=128,capture=True)
    patches,specials=token_layout(2,28*37,5,'cpu')
    previous=[None];checked=[]
    def check(module,inputs,output):
        layer=module.layer
        if 10<=layer<=16:
            ids=state.captured[layer]['indices']
            mask=mask_from_ids(ids,2082)
            history=previous[0];valid=history>=0
            incoming=patches[history.long().clamp_min(0)]
            assert bool(mask.gather(1,incoming)[valid].all())
            assert bool(mask[:,specials].all())
            assert bool((mask.sum(1)==int(state.retention*2082)).all())
            checked.append(layer)
        if 9<=layer<=16:previous[0]=state.history.cpu().clone()
    handles=[block.attn.register_forward_hook(check) for block in model.aggregator.global_blocks]
    rows=[];saved={};poses={};records={}
    for retention in [.1,.2]:
        for variant in ['A','B']:
            name=f'budget{round(retention*100)}_{variant}'
            state.retention,state.variant=retention,variant;previous[0]=None;checked.clear()
            torch.cuda.synchronize();started=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pose=model(images)['pose_enc'].float().cpu()
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            assert checked==list(range(10,17))
            assert torch.equal(state.captured[10]['q'],dense_qk[10][0])
            assert torch.equal(state.captured[10]['k'],dense_qk[10][1])
            poses[name]=pose.numpy();records[name]=list(state.records)
            for layer in range(10,17):
                data=state.captured[layer];ids=data['indices'];saved[f'{name}_L{layer}']=ids.numpy().astype(np.int32)
                q,k,v=[data[key].cuda() for key in ['q','k','v']]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    reference=F.scaled_dot_product_attention(q[None],k[None],v[None])[0].float()
                actual=data['output'].cuda().float()
                with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                    original_weights=dense_weights(dense_qk[layer][0].cuda(),dense_qk[layer][1].cuda())
                ranking=original_weights[:,patches.cuda()].argsort(dim=1,descending=True,stable=True)[:,:207]
                mask=mask_from_ids(ids.cuda(),2082)[:,patches.cuda()]
                rows.append(dict(setting=name,layer=layer,
                    local_output_error=float(torch.linalg.vector_norm(actual-reference)/torch.linalg.vector_norm(reference)),
                    original_dense_top10_recall=float(mask.gather(1,ranking).float().mean()),
                    diagnostic_seconds=seconds))
            print('validated',name,'hard history, budgets, Dense prefix',flush=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        repeat=model(images)['pose_enc'].float().cpu().numpy()
    assert np.array_equal(repeat,poses['budget20_B'])
    for handle in handles:handle.remove()
    np.savez_compressed(out/'masks.npz',**saved);np.savez_compressed(out/'poses.npz',**poses)
    sources=['saf3r/patch/region_history_attention.py','saf3r/patch/region_history_corrections.py']
    result=dict(rows=rows,records=records,input=meta['image_paths'],
        production_dense_oracle=False,all_incoming_history_preserved=True,all_specials_preserved=True,
        all_masks_unique_exact_budget=True,dense_prefix_bitwise_equal=True,repeated_pose_bitwise_equal=True,
        protocol='Native Dense0–9 / corrected sparse10–16 / Dense17–23; two frames; not speed/accuracy benchmark',
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE',str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args().output_dir)
