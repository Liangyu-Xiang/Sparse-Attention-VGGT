"""Small raw replay artifact; never use this capture forward for timing."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main(args):
    from scripts.benchmark_adaptive_block_value import source_manifest
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.patch.adaptive_block_value import install_adaptive_block_value
    from saf3r.utils.data_utils import load_and_process_images_vggt
    sys.path.insert(0,str(ROOT/'evaluation'))
    from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    reader=SevenScenes();reader.data_root=str(ROOT/'workspace/benchmark_dataset/7scenes')
    data=reader.get_data(args.scene)
    paths=[data.image_files[i] for i in args.indices]
    images=load_and_process_images_vggt(paths).cuda()
    model=build_model(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt','cuda')
    state=install_adaptive_block_value(model,capture=True,validation_queries=64)
    raw={};positions={};handles=[]
    def before(layer):
        def hook(module,inputs,kwargs):
            pos=kwargs.get('pos')
            positions[layer]=None if pos is None else pos.detach().cpu()
        return hook
    def qkv_hook(layer):
        def hook(module,inputs,output):
            raw[layer]=output.detach().cpu()
        return hook
    for layer,b in enumerate(model.aggregator.global_blocks):
        handles+=[b.attn.register_forward_pre_hook(before(layer),with_kwargs=True),
                  b.attn.original.qkv.register_forward_hook(qkv_hook(layer))]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=model(images)
    torch.cuda.synchronize()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for layer,snapshot in enumerate(state.snapshots):
        torch.save(dict(snapshot=snapshot,decision=state.records[layer],
                        projected_qkv_before_headnorm_rope=raw[layer],rope_positions=positions[layer]),
                   args.output_dir/f'layer_{layer:02d}.pt')
    torch.save(dict(pose_enc=prediction['pose_enc'].float().cpu()),args.output_dir/'prediction.pt')
    metadata=dict(scene=args.scene,frame_indices=args.indices,image_paths=paths,image_shape=list(images.shape),
        source_sha256=source_manifest(),layers=list(range(24)),
        note='Raw SMALL-context replay/debug artifact, not a 128-frame accuracy or timing result. QKV is pre-headnorm/RoPE; use saved positions and checkpoint headnorm/RoPE modules to reconstruct. Layer snapshots contain shared hierarchy, actual head/block masks and validated histories; dense fallback masks are None. Decisions include C/S, budgets, head matches and independent query errors.')
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for handle in handles:handle.remove()
    print(args.output_dir,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',default='chess');p.add_argument('--indices',nargs='+',type=int,default=[0,20])
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/adaptive_block_value/chess_0_20_replay')
    main(p.parse_args())
