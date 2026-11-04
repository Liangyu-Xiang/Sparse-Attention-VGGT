"""Stage-event diagnostic of one frozen candidate; not a speed/quality evaluation."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch,numpy as np
from scripts.quality_metrics import quality_record,quality_markdown

def main():
 from addict import Dict
 import yaml
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from benchmarks.da3.bench.registries import MV_REGISTRY
 from benchmarks.da3.da3bench_utils import sample_frames
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 q=quality_record({},'Stage-event profiling diagnostic only; pose/recon_unposed not evaluated')
 (a.output/'quality.json').write_text(json.dumps(q,indent=2));(a.output/'quality.md').write_text(quality_markdown(q))
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dataset=MV_REGISTRY.get('7scenes')();indices,scene=sample_frames(dataset.get_data('heads'),'heads',128,sample_mode='uniform',seed=0)
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',scene.image_files).cuda();state=install_reuse_residual_exploration(model,ReuseExploreConfig(**json.loads(a.config.read_text())))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for _ in range(2):raw=model(images);del raw
  torch.cuda.synchronize();state.profiling=True;torch.cuda.reset_peak_memory_stats();start=time.perf_counter();raw=model(images);torch.cuda.synchronize()
 rows=[dict(layer=l,stage=n,milliseconds=start.elapsed_time(end)) for l,n,start,end in state.profile_events]
 totals={n:sum(r['milliseconds'] for r in rows if r['stage']==n) for n in {r['stage'] for r in rows}}
 result=dict(stages=totals,layer_events=rows,wall_seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
  speedup_vs_dense=None,reason='Instrumented stage events, no matched warmed Dense full-forward comparison',images=scene.image_files,indices=indices,config=json.loads(a.config.read_text()),quality=q,
  threshold_m=dataset.eval_threshold,downsample_m=dataset.down_sample,saf3r_profile=None)
 (a.output/'profile.json').write_text(json.dumps(result,indent=2));print(totals,flush=True)
if __name__=='__main__':main()
