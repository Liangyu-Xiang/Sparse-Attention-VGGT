"""One native sequence: Dense0–9 / corrected Attention10–16 / Dense17–23."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(args):
    import torch
    from scripts.overlay_vggt_query_attention import build_model
    from saf3r.patch.region_history_corrections import install_corrected_region_history_attention
    from saf3r.utils.data_utils import load_and_process_images_vggt
    from saf3r.models.official_vggt.utils.pose_enc import pose_encoding_to_extri_intri
    sys.path.insert(0,str(ROOT/'evaluation'))
    from benchmarks.da3.bench.datasets.sevenscenes import SevenScenes
    from benchmarks.da3.bench.utils import compute_pose
    from benchmarks.da3.geometry import as_homogeneous
    if args.warmup<0 or args.repeats<1:raise ValueError('Invalid repetition count')
    directory=args.output_dir/('dense' if args.dense else 'corrected')/args.scene
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():raise ValueError('Use fresh output directory')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    reader=SevenScenes();reader.data_root=str(ROOT/'workspace/benchmark_dataset/7scenes')
    data=reader.get_data(args.scene)
    positions=np.linspace(0,len(data.image_files)-1,args.frames,dtype=int)
    paths=[data.image_files[i] for i in positions]
    images=load_and_process_images_vggt(paths).cuda()
    model=build_model(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt','cuda')
    state=None if args.dense else install_corrected_region_history_attention(model,
        variant=args.variant,terminal='local_top1',retention=args.retention,
        transfer_ratio=args.transfer_ratio,query_chunk=args.query_chunk,capture=False)
    sources=['saf3r/patch/region_history_attention.py','saf3r/patch/region_history_corrections.py',
        'saf3r/patch/residual_history_av.py','saf3r/triton/residual_attention.py',
        'saf3r/patch/layer_shared_av.py','saf3r/patch/hierarchical_value_attention.py',
        'saf3r/patch/value_clustering_gpu.py','scripts/evaluate_corrected_region_sequence.py']
    hashes={}
    for name in sources:
        content=(ROOT/name).read_bytes();hashes[name]=hashlib.sha256(content).hexdigest()
        target=directory/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    config=vars(args).copy();config['output_dir']=str(config['output_dir'])
    (directory/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (directory/'input.json').write_text(json.dumps(dict(image_paths=paths,frame_indices=positions.tolist(),
        image_shape=list(images.shape)),indent=2)+'\n')
    events={};sample_layers=[];iteration_ref=[0]
    def before(module,inputs):
        start=torch.cuda.Event(enable_timing=True);start.record();events[module._evaluation_layer]=start
    def after(module,inputs,output):
        layer=module._evaluation_layer;end=torch.cuda.Event(enable_timing=True);end.record()
        sample_layers.append((layer,events[layer],end))
        if args.progress_layers:
            end.synchronize()
            record=dict(iteration=iteration_ref[0],layer=layer,seconds=events[layer].elapsed_time(end)/1000,
                peak_gib=torch.cuda.max_memory_allocated()/2**30)
            (directory/'layer_progress.json').write_text(json.dumps(record)+'\n')
            print('LAYER',json.dumps(record),flush=True)
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn._evaluation_layer=layer
        block.attn.register_forward_pre_hook(before);block.attn.register_forward_hook(after)
    samples=[];fingerprints=[]
    for iteration in range(args.warmup+args.repeats):
        iteration_ref[0]=iteration;sample_layers.clear()
        (directory/'status.json').write_text(json.dumps(dict(state='inference',iteration=iteration))+'\n')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(images)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        pose=prediction['pose_enc'].float().cpu();del prediction
        assert torch.isfinite(pose).all() and len(sample_layers)==24
        layer_seconds={layer:a.elapsed_time(b)/1000 for layer,a,b in sample_layers}
        if state is not None:
            assert [record['layer'] for record in state.records]==list(range(9,17))
            tokens=images.shape[0]*((images.shape[-2]//14)*(images.shape[-1]//14)+5)
            expected=int(args.retention*tokens)/tokens
            assert all(abs(record['density']-expected)<1e-7 for record in state.records if record['mode']=='sparse')
            assert not state.captured
        sample=dict(seconds=seconds,global_seconds=sum(layer_seconds.values()),
            middle_seconds=sum(layer_seconds[l] for l in range(10,17)),
            seed_layer9_seconds=layer_seconds[9],peak_gib=torch.cuda.max_memory_allocated()/2**30,
            layer_seconds=layer_seconds)
        if iteration>=args.warmup:
            samples.append(sample);fingerprints.append(hashlib.sha256(pose.numpy().tobytes()).hexdigest())
        print('FORWARD',json.dumps(sample),flush=True)
    extrinsics,intrinsics=pose_encoding_to_extri_intri(pose,images.shape[-2:])
    metrics=compute_pose(torch.from_numpy(as_homogeneous(extrinsics[0].numpy())),
        torch.from_numpy(np.asarray(data.extrinsics)[positions]))
    np.savez_compressed(directory/'poses.npz',pose_enc=pose.numpy(),extrinsics=extrinsics.numpy(),intrinsics=intrinsics.numpy())
    medians={'median_'+key:float(np.median([sample[key] for sample in samples]))
        for key in ['seconds','global_seconds','middle_seconds','seed_layer9_seconds','peak_gib']}
    result=dict(scene=args.scene,config=config,source_sha256=hashes,samples=samples,
        pose={key:float(value) for key,value in metrics.items()},
        input=dict(image_paths=paths,frame_indices=positions.tolist(),image_shape=list(images.shape),
            checkpoint=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt'),gpu=torch.cuda.get_device_name()),
        layers=[] if state is None else state.records,pose_fingerprints=fingerprints,
        repeated_pose_identical=len(set(fingerprints))==1,
        timing_protocol='Cold diagnostic single forward, includes compilation and layer synchronization' if args.warmup==0
            else 'Warmed timing; per-layer synchronization included' if args.progress_layers else 'Warmed timing',
        no_dense_oracle_in_middle=True,**medians)
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (directory/'status.json').write_text(json.dumps(dict(state='complete'))+'\n')
    print('METRICS',json.dumps(result['pose']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--scene',default='chess')
    parser.add_argument('--frames',type=int,default=128)
    parser.add_argument('--dense',action='store_true')
    parser.add_argument('--retention',type=float,default=.1)
    parser.add_argument('--transfer-ratio',type=float,default=.5)
    parser.add_argument('--variant',choices=['A','B'],default='B')
    parser.add_argument('--query-chunk',type=int,default=32)
    parser.add_argument('--warmup',type=int,default=0)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--progress-layers',action='store_true')
    run(parser.parse_args())
