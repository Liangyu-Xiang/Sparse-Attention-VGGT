"""New method only;RGB timing,local Dense teachers and historical probability audit."""
import argparse,json,sys,time,statistics,ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import save,Fidelity


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--timing-from',type=Path);p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/streamed_history_20261106.json');p.add_argument('--sequence',choices=['scannet','tum128'],default='scannet');a=p.parse_args()
 import numpy as np,torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record
 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['saf3r','evaluation'] for p in (ROOT/folder).rglob('*.py')};plan=json.loads((ROOT/'outputs/quality_speed_search_20261106/phase5/plan.json').read_text());entry=plan['calibration']['scannet'] if a.sequence=='scannet' else plan['validation']['tum_walking128'];torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=plan['checkpoint'];model=build_model(cfg,'cuda').eval();images=load_images(model,'vggt',entry['image_paths']).cuda();config=json.loads(a.config.read_text());s=install_reuse_residual_exploration(model,ReuseExploreConfig(**config));samples=[];raw=None
 if a.timing_from:
  samples=[ast.literal_eval(line.split(' ',2)[2]) for line in a.timing_from.read_text().splitlines() if line.startswith('TIMING ')]
  assert len(samples)==4
 for rep in ([] if a.timing_from else range(4)):
  del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize();samples.append(dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30));print('TIMING',rep,samples[-1],flush=True)
 del raw;raw=None
 from scripts.quality_metrics import quality_record
 save(a.output,dict(quality=quality_record({},'Explicit RGB complete-forward timing only;pose and reconstruction not rerun'),sequence=a.sequence,input=entry,config=config,source_sha256=source_sha256,timing=dict(warmup=1,repeats=3,median_seconds=statistics.median(v['seconds'] for v in samples[1:]),samples=samples[1:],cold_samples=samples[:1]),speedup_vs_dense=None,speedup_reason='No paired Dense timing authorized'))
 print('COMPLETE',a.sequence,flush=True)
if __name__=='__main__':main()
