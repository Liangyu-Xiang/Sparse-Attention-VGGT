"""Changed method vs its immediate reference; Dense explicitly not rerun."""
import argparse,json,statistics,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
    model=build_model(cfg,'cuda');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();c=ReuseExploreConfig(**json.loads(a.config.read_text()));s=install_reuse_residual_exploration(model,c)
    methods={'current':replace(c,execution_kernel='reference',fuse_mask_ops=False,budget_graph=False,window_graph=False,history_evidence_backend='grid_local',cache_probe_indices=False,fuse_state_kernel=False,reuse_window_graphs=False,fused_coarse=False),'candidate':c};times={m:[] for m in methods}
    for i in range(4):
        for m in (list(methods) if i%2==0 else list(reversed(methods))):
            s.config=methods[m];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            assert len(s.records)==24 and torch.isfinite(raw['depth']).all() and torch.isfinite(raw['pose_enc']).all()
            if i:times[m].append(dict(seconds=elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,graph_hits=s.graph_cache_hits,graph_misses=s.graph_cache_misses,graph_evictions=s.graph_cache_evictions,geometry_updates=s.graph_constant_updates))
            del raw
            (a.output_dir/'progress.json').write_text(json.dumps(dict(iteration=i,method=m,times=times),indent=2));print(m,i,elapsed,flush=True)
    result=dict(complete=True,times=times,median_seconds={m:statistics.median(t['seconds'] for t in ts) for m,ts in times.items()},quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Complete-forward performance diagnostic, geometry evaluation pending',dense_speedup=None,dense_reason='User explicitly requested no Dense rerun; historical Dense comparison not contemporaneous')
    (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
