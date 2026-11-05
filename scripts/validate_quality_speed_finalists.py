"""Independent RGB validation, candidate selection frozen before test geometry."""
import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import Fidelity,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import numpy as np,torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record,quality_markdown
 plan=json.loads(a.plan.read_text());torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=plan['checkpoint']
 model=build_model(cfg,'cuda').eval()
 for name,entry in plan['validation'].items():
  images=load_images(model,'vggt',entry['image_paths']).cuda()
  for candidate in ['current',a.candidate]:
   row=next(r for r in plan['candidates'] if r['name']==candidate);out=a.output/candidate/name
   out.mkdir(parents=True,exist_ok=False);q=quality_record({},'Independent RGB-only candidate validation, no geometric truth; all six quality fields unavailable')
   save(out/'quality.json',q);(out/'quality.md').write_text(quality_markdown(q))
   state=install_reuse_residual_exploration(model,ReuseExploreConfig(**row['config']));samples=[];raw=None
   for rep in range(4):
    del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();samples.append(dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30));print('TIMING',candidate,name,rep,samples[-1],flush=True)
   del raw;raw=None
   with Fidelity(model,state) as fidelity,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
   torch.cuda.synchronize();score=fidelity.summary();del fidelity
   save(out/'result.json',dict(name=candidate,input=entry,config=row['config'],quality=q,timing=dict(warmup=1,repeats=3,median_seconds=float(np.median([r['seconds'] for r in samples[1:]])),samples=samples[1:],cold_samples=samples[:1]),fidelity=score,speedup_vs_dense=None,reason='RGB validation not Dense control comparison',test_gt_read=False,checkpoint_sha256=plan['checkpoint_sha256']))
   state.reset_handle.remove()
   for b in model.aggregator.global_blocks:b.attn=b.attn.original
   del model.reuse_explore_state
   del state,raw;gc.collect();torch.cuda.empty_cache()
  del images;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
