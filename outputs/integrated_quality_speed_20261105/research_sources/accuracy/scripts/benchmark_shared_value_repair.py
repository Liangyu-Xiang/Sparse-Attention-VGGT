"""Alternating same-model full-forward timing; Dense speedup remains unavailable."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import numpy as np
import torch
from scripts.quality_metrics import quality_record,quality_markdown


def main():
    from addict import Dict
    import yaml
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from benchmarks.da3.bench.registries import MV_REGISTRY
    from benchmarks.da3.da3bench_utils import sample_frames
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=128)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--candidate-config',type=Path,default=ROOT/'configs/evaluation/reuse_shared_value_loss_256_repaired.json')
    p.add_argument('--sparse-statistics',action='store_true',help='Experimental algebraically equivalent cluster sums plus sparse corrections')
    args=p.parse_args()
    if args.sparse_statistics:
        from saf3r.triton.state_loss_sparse_statistics import enable
        enable()
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        (args.output/name).write_text(json.dumps(data,indent=2)+'\n')
    q=quality_record({},'Timing-only diagnostic; pose and recon_unposed not evaluated')
    q['speedup_unavailable']='Dense not rerun; paired comparison is original Shared Value versus repaired Shared Value only'
    for method in ('original','repaired'):
        (args.output/method).mkdir()
        (args.output/method/'quality.json').write_text(json.dumps(q,indent=2)+'\n')
        (args.output/method/'quality.md').write_text(quality_markdown(q))
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dataset=MV_REGISTRY.get('7scenes')()
    indices,scene=sample_frames(dataset.get_data('heads'),'heads',args.frames,sample_mode='uniform',seed=0)
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model'])
    cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda()
    original_attn=[b.attn for b in model.aggregator.global_blocks]
    variants={}
    for method,filename in [('original',ROOT/'configs/evaluation/reuse_shared_value_loss_256.json'),('repaired',args.candidate_config)]:
        for block,attn in zip(model.aggregator.global_blocks,original_attn):block.attn=attn
        if hasattr(model,'reuse_explore_state'):del model.reuse_explore_state
        config=ReuseExploreConfig(**json.loads(filename.read_text()))
        state=install_reuse_residual_exploration(model,config)
        state.reset_handle.remove()
        variants[method]=(state,[b.attn for b in model.aggregator.global_blocks])
    save('input.json',dict(image_paths=scene.image_files,frame_indices=indices,checkpoint=cfg.ckpt_path,
        shape=list(images.shape),hardware=torch.cuda.get_device_name(),warmup=args.warmup,repeats=args.repeats,
        scope='Alternating original/repaired full forwards of same model, includes reset/clustering/selection/history/heads; loading and switching excluded',
        protocol=dict(f1_threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample),saf3r='N/A',
        sparse_statistics=args.sparse_statistics,configs={m:asdict(s.config) for m,(s,_) in variants.items()}))
    samples={m:[] for m in variants};cold={m:[] for m in variants}
    for rep in range(args.warmup+args.repeats):
        order=['original','repaired'] if rep%2==0 else ['repaired','original']
        for method in order:
            state,attentions=variants[method]
            for block,attn in zip(model.aggregator.global_blocks,attentions):block.attn=attn
            def reset(module,inputs):
                x=inputs[0];state.reset(x.shape[1],x.shape[-2]//14,x.shape[-1]//14,x.device)
            handle=model.aggregator.register_forward_pre_hook(reset)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
            torch.cuda.synchronize()
            sample=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                actual_mean_density=float(np.mean([float(r['actual_unique_density']) for r in state.records])))
            handle.remove();del raw
            (cold if rep<args.warmup else samples)[method].append(sample)
            save('progress.json',dict(warmup=cold,samples=samples))
            print('TIMING',rep,method,sample,flush=True)
    medians={m:float(np.median([s['seconds'] for s in v])) for m,v in samples.items()}
    save('result.json',dict(warmup=cold,samples=samples,medians=medians,
        speedup_vs_original_shared=medians['original']/medians['repaired'],
        runtime_increase_fraction=medians['repaired']/medians['original']-1.,
        speedup_vs_dense=None,dense_reason=q['speedup_unavailable'],quality=q,
        memory_note='One shared set of model weights; both selector Graph caches coexist. Per-method standalone memory is in quality runs.'))
    print(medians,flush=True)


if __name__=='__main__':main()
