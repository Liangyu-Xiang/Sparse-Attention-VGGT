"""Full-forward repair-cache timing and output equivalence, without geometry GT."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--configs',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch,yaml
    from addict import Dict
    from saf3r.utils.model_utils import build_model
    from saf3r.utils.data_utils import load_images
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
    from scripts.quality_metrics import quality_record,quality_markdown
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    quality=quality_record({},'RGB-only full-forward cache/equivalence diagnostic; pose and recon_unposed not evaluated; no paired original Dense timing')
    save(a.output_dir/'quality.json',quality);(a.output_dir/'quality.md').write_text(quality_markdown(quality))
    save(a.output_dir/'status.json',{'stage':'loading'})
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');paths=json.loads(a.input.read_text())['image_paths'];images=load_images(model,'vggt',paths).cuda()
    save(a.output_dir/'input.json',dict(image_paths=paths,shape=list(images.shape),checkpoint_sha256=hashlib.sha256(Path(cfg.ckpt_path).read_bytes()).hexdigest()))
    originals=[b.attn for b in model.aggregator.global_blocks];reference=None;rows=[]
    for path in a.configs:
        config=json.loads(path.read_text());state=install_reuse_residual_exploration(model,ReuseExploreConfig(**config));samples=[]
        save(a.output_dir/'status.json',dict(stage='forward',variant=path.stem))
        for i in range(4):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if i:samples.append(dict(seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,graph_creations=state.graph_cache_creations,graph_evictions=state.graph_cache_evictions,graph_cache_bytes=getattr(state,'graph_cache_bytes',None)))
            if i==3:prediction={k:v.detach().cpu() for k,v in raw.items() if torch.is_tensor(v)}
            del raw
        compare=None
        if reference is not None:
            compare={k:dict(exact=bool(torch.equal(v,reference[k])),relative_l2=float((v.float()-reference[k].float()).norm()/reference[k].float().norm().clamp_min(1e-12)),max_abs=float((v.float()-reference[k].float()).abs().max())) for k,v in prediction.items()}
        else:reference=prediction
        row=dict(variant=path.stem,config=config,quality=quality,samples=samples,warmup=1,repeats=3,median_seconds=float(np.median([s['seconds'] for s in samples])),equivalence_to_first=compare,hardware=dict(gpu=torch.cuda.get_device_name(),uuid=str(torch.cuda.get_device_properties(0).uuid)),scope='Complete uninstrumented model forward, including reset/clustering/selection/history and cache construction/eviction')
        rows.append(row);save(a.output_dir/'summary.json',dict(rows=rows));print(path.stem,row['median_seconds'],compare,flush=True)
        state.reset_handle.remove()
        for block,original in zip(model.aggregator.global_blocks,originals):block.attn=original
        del model.reuse_explore_state;del state;gc.collect();torch.cuda.empty_cache()
    save(a.output_dir/'status.json',dict(stage='diagnostic_complete'))

if __name__=='__main__':main()
