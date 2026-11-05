"""RGB-only stage timing/state coverage diagnostics, separate from production timing."""
import argparse,dataclasses,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/rep_history_loss_20261105/profiles'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def cpu(v):
 import torch
 if torch.is_tensor(v):return v.item() if v.numel()==1 else v.detach().cpu().tolist()
 if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
 if isinstance(v,list):return [cpu(x) for x in v]
 return v
def main():
 import torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record,quality_markdown
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 quality=quality_record({},'Instrumented RGB-only CUDA stage/state diagnostic; no pose/recon_unposed or paired production Dense timing')
 save(OUT/'quality.json',quality);(OUT/'quality.md').write_text(quality_markdown(quality))
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(cfg,'cuda');inp=json.loads((ROOT/'outputs/global_state_loss_oracle_20261105/speed_heads128/layout_fixed/input.json').read_text());images=load_images(model,'vggt',inp['image_paths']).cuda();save(OUT/'input.json',inp)
 originals=[b.attn for b in model.aggregator.global_blocks]
 for name in ['baseline','A','B']:
  config=json.loads((ROOT/'outputs/rep_history_loss_20261105/calibration/configs'/f'{name}.json').read_text());config['rep_history_diagnostics']=True
  state=install_reuse_residual_exploration(model,ReuseExploreConfig(**config));state.profiling=True;samples=[]
  for i in range(3):
   torch.cuda.synchronize();begin=time.perf_counter()
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(images)
   torch.cuda.synchronize();seconds=time.perf_counter()-begin;del pred
   if i:
    times={};layer_times={}
    for layer,stage,a,b in state.profile_events:
     value=a.elapsed_time(b);times[stage]=times.get(stage,0.)+value;key=f'{layer}:{stage}';layer_times[key]=layer_times.get(key,0.)+value
    mask_names=['value_update','pooling','cluster_seed_prepare','coarse_observation','m5c_coarse_observation','reuse_and_budget','residual_fine_probe']
    samples.append(dict(instrumented_full_forward_seconds=seconds,stage_gpu_ms=times,layer_stage_gpu_ms=layer_times,selection_prior_budget_gpu_ms=times.get('reuse_and_budget',0.),mask_pipeline_gpu_ms=sum(times.get(n,0.) for n in mask_names),layers=cpu(state.records),graph_creations=state.graph_cache_creations,graph_evictions=state.graph_cache_evictions))
  save(OUT/f'{name}.json',dict(method=name,config=config,scope='heads128, GPU6;1 warmup+2 instrumented forwards; profile flag only adds statistics. Not production full-forward speed. Stage CUDA events include stream idle gaps due host submission. Mask pipeline includes Value update/clustering, shared pooling/statistics, coarse estimates, prior/gating/loss/budget/packing/weights; sparse attention/history execution separately reported',quality=quality,samples=samples,hardware=dict(gpu=torch.cuda.get_device_name(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid))))
  print('PROFILE_COMPLETE',name,[(s['mask_pipeline_gpu_ms'],s['selection_prior_budget_gpu_ms']) for s in samples],flush=True)
  state.reset_handle.remove()
  for b,o in zip(model.aggregator.global_blocks,originals):b.attn=o
  del model.reuse_explore_state;del state;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
