"""Finite overall10% retention profiles;independent RGB teachers,no target GT."""
import argparse,json,sys,time,gc,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import save
from scripts.camera_weighted_history_audit import CameraAudit as Audit

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
 import numpy as np,torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record
 plan=json.loads(a.plan.read_text());source=json.loads((ROOT/'outputs/quality_speed_search_20261106/phase5/plan.json').read_text());torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=source['checkpoint'];model=build_model(cfg,'cuda').eval();images={k:load_images(model,'vggt',v['image_paths']).cuda() for k,v in source['calibration'].items()}
 for i,row in enumerate(plan['candidates']):
  if i%3!=a.shard:continue
  config=json.loads((ROOT/row['config']).read_text())
  for name,x in images.items():
   if (a.output/row['name']/name/'result.json').exists():continue
   s=install_reuse_residual_exploration(model,ReuseExploreConfig(**config));raw=None
   try:
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(x)
    torch.cuda.synchronize();del raw;raw=None
    with Audit(s) as audit,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(x)
    torch.cuda.synchronize();assert len(audit.fidelity)>0
    layers={}
    for v in audit.fidelity:layers.setdefault(v['layer'],[]).append(v['projected_error_vs_dense'])
    score=statistics.mean(statistics.mean(v) for v in layers.values());assert len(layers)==24
    weighted={}
    for record in audit.camera_rows:
     item=weighted.setdefault(record['layer'],dict(numerator=0.,denominator=0.,mass=record['patch_mass']))
     item['numerator']+=sum(w*e for w,e in zip(record['importance'],record['squared_error']));item['denominator']+=sum(w*e for w,e in zip(record['importance'],record['reference_energy']))
    assert len(weighted)==23
    camera_score=sum(v['mass']*v['numerator']/max(v['denominator'],1e-20) for v in weighted.values())/sum(v['mass'] for v in weighted.values())
    save(a.output/row['name']/name/'result.json',dict(name=row['name'],sequence=name,config=config,input=source['calibration'][name],quality=quality_record({},'Independent RGB-only sparse90 profile screening;local exact Attention teachers,no geometry'),score=score,camera_score=camera_score,camera_layers=weighted,camera_rows=audit.camera_rows,fidelity=audit.fidelity,probability=audit.rows,speedup_vs_dense=None,speedup_reason='Instrumented RGB proxy only,not formal timing',test_gt_read=False));print('CAMERA_SCREEN',row['name'],name,score,camera_score,flush=True)
   finally:
    s.reset_handle.remove()
    for block in model.aggregator.global_blocks:block.attn=block.attn.original
    del model.reuse_explore_state
    del s,raw;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
