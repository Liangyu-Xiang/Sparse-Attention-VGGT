"""Pair bare AV / AV+export on every actual mask of one new-method trajectory.

Not a full Dense/old-method rerun. Instrumented inference has extra calls and is
NEVER used as normal end-to-end timing or geometric quality.
"""
import argparse,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import save
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from saf3r.patch import current_probe_reuse as backend
 from scripts.quality_metrics import quality_record
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');model=build_model(cfg,'cuda').eval();entry=json.loads(a.input.read_text());images=load_images(model,'vggt',entry['image_paths']).cuda();config=json.loads(a.config.read_text());s=install_reuse_residual_exploration(model,ReuseExploreConfig(**config))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();del raw
 original=backend.main_attention;rows=[]
 def paired(q,k,v,actual,specials,scale,bias,**kwargs):
  samples={'bare':[],'with_export':[]};bare_kwargs=dict(kwargs,history_bits=0,history_logz=None,history_split=False);selected=None
  for rep in range(3):
   for name in (['bare','with_export'] if rep%2==0 else ['with_export','bare']):
    first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record();out,evidence=original(q,k,v,actual,specials,scale,bias,**(bare_kwargs if name=='bare' else kwargs));last.record();last.synchronize();samples[name].append(first.elapsed_time(last)/1000)
    if name=='bare':reference=out
    else:selected=(out,evidence)
  error=float((reference.float()-selected[0].float()).square().sum().sqrt()/reference.float().square().sum().sqrt().clamp_min(1e-20));assert error<=.005
  base=statistics.median(samples['bare']);export=statistics.median(samples['with_export']);row=dict(layer=s.previous_layer+1,window=len([r for r in rows if r['layer']==s.previous_layer+1]),queries=q.shape[1],groups=len(actual),patch_cap=actual.shape[1],bare_av_seconds=base,av_with_export_seconds=export,incremental_export_seconds=export-base,relative_l2=error,samples=samples);rows.append(row);print('PAIR',row['layer'],row['window'],base,export,flush=True);return selected
 backend.main_attention=paired
 try:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 finally:backend.main_attention=original
 bare=sum(r['bare_av_seconds'] for r in rows);export=sum(r['av_with_export_seconds'] for r in rows)
 save(a.output,dict(quality=quality_record({},'Instrumented current-method QKV/mask-paired AV diagnostic;no geometric truth'),input=entry,config=config,rows=rows,bare_av_seconds=bare,av_with_export_seconds=export,incremental_export_seconds=export-bare,export_overhead_fraction=export/bare-1,maximum_relative_l2=max(r['relative_l2'] for r in rows),speedup_vs_dense=None,speedup_reason='No full Dense timing',scope='Sum of per-window paired GPU-event medians;AV plus export only,excludes coarse observation,missing-key normalization,cache finalization and transfer;instrumented full-forward is not a runtime measurement'))
if __name__=='__main__':main()
