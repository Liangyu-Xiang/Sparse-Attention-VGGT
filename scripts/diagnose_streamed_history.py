"""New method only;RGB timing,local Dense teachers and historical probability audit."""
import argparse,json,sys,time,statistics,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import save,Fidelity

class Audit:
 def __init__(self,state):self.s=state;self.rows=[];self.generation=[]
 def __enter__(self):
  import torch
  import saf3r.patch.reuse_residual_exploration as mod
  from saf3r.patch import streamed_probability_history as hist
  self.prepare=mod.prepare_qkv;self.observe=hist.observe;self.prior=hist.prior;self.finalize=hist.finalize
  def prepare(original,x,pos):
   q,k,v=self.prepare(original,x,pos);q=q.to(v.dtype);k=k.to(v.dtype);s=self.s;self.groups=torch.linspace(0,s.groups-1,4,device=q.device).long().unique();positions=(self.groups*s.config.query_group_size+s.config.query_group_size//8).clamp_max(s.n-1)
   with torch.autocast('cuda',enabled=False):self.truth=(q[:,positions].float()@k.float().transpose(1,2)*original.scale).softmax(-1).mean(0)[:,s.patches]
   return q,k,v
  def observe(*args):self.obs=self.observe(*args);return self.obs
  def metrics(a,b):return dict(relative_l1=float((a-b).abs().sum()/b.sum().clamp_min(1e-20)),predicted_mass=float(a.sum()),true_mass=float(b.sum()))
  def prior(s,obs,start,g):
   result=self.prior(s,obs,start,g);pick=(self.groups>=start)&(self.groups<start+g);groups=self.groups[pick]
   if len(groups):
    predicted=result[groups-start];true=self.truth[pick];coarse=obs['base'][groups][:,s.grid]
    row=dict(layer=s.previous_layer+1,groups=groups.tolist(),all_patch_prediction=metrics(predicted,true),coarse_only=metrics(coarse,true),has_history=s.streamed_history_cache is not None)
    if s.streamed_history_cache is not None:
     old=s.streamed_history_cache;ids=s.history_ids[groups].long();valid=ids>=0;safe=ids.clamp_min(0);regions=s.grid[safe]
     previous=(old['conditional'][groups].float()*old['estimated_grid_probability'][groups].gather(1,regions))*valid
     expected=true.gather(1,safe)*valid;current=predicted.gather(1,safe)*valid
     row.update(history_migrated=metrics(current,expected),history_unchanged=metrics(previous,expected),history_current_coarse=metrics(coarse.gather(1,safe)*valid,expected))
    self.rows.append(row)
   return result
  def finalize(s,obs,start,ids,scores):
   cache=self.finalize(s,obs,start,ids,scores);g=len(ids);pick=(self.groups>=start)&(self.groups<start+g);groups=self.groups[pick]
   if len(groups):
    local=groups-start;keys=ids[local].long();valid=keys>=0;safe=keys.clamp_min(0);region=s.grid[safe]
    generated=cache['conditional'][local].float()*cache['estimated_grid_probability'][local].gather(1,region);expected=self.truth[pick].gather(1,safe)
    self.generation.append(dict(layer=s.previous_layer+1,groups=groups.tolist(),generated_dense_probability=metrics(generated*valid,expected*valid)))
   return cache
  mod.prepare_qkv=prepare;hist.observe=observe;hist.prior=prior;hist.finalize=finalize;return self
 def __exit__(self,*exc):
  import saf3r.patch.reuse_residual_exploration as mod
  from saf3r.patch import streamed_probability_history as hist
  mod.prepare_qkv=self.prepare;hist.observe=self.observe;hist.prior=self.prior;hist.finalize=self.finalize

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--timing-from',type=Path);p.add_argument('--sequence',choices=['scannet','tum128'],default='scannet');a=p.parse_args()
 import numpy as np,torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record
 plan=json.loads((ROOT/'outputs/quality_speed_search_20261106/phase5/plan.json').read_text());entry=plan['calibration']['scannet'] if a.sequence=='scannet' else plan['validation']['tum_walking128'];torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=plan['checkpoint'];model=build_model(cfg,'cuda').eval();images=load_images(model,'vggt',entry['image_paths']).cuda();config=json.loads((ROOT/'configs/evaluation/streamed_history_20261106.json').read_text());s=install_reuse_residual_exploration(model,ReuseExploreConfig(**config));samples=[];raw=None
 if a.timing_from:
  samples=[ast.literal_eval(line.split(' ',2)[2]) for line in a.timing_from.read_text().splitlines() if line.startswith('TIMING ')]
  assert len(samples)==4
 for rep in ([] if a.timing_from else range(4)):
  del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize();samples.append(dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30));print('TIMING',rep,samples[-1],flush=True)
 del raw;raw=None
 with Audit(s) as audit,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 del raw;raw=None;s._diagnostic_heldout=True
 with Fidelity(model,s) as fidelity,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 q=quality_record({},'Independent RGB-only current sparse-trajectory QKV probability/attention diagnostic;no geometric truth or full Dense rerun')
 save(a.output,dict(quality=q,sequence=a.sequence,input=entry,config=config,timing=dict(warmup=1,repeats=3,median_seconds=statistics.median(v['seconds'] for v in samples[1:]),samples=samples[1:],cold_samples=samples[:1]),probability_transfer=audit.rows,probability_generation=audit.generation,fidelity=fidelity.summary(),speedup_vs_dense=None,speedup_reason='No full Dense rerun;new method only',test_gt_read=False,layers=[{k:(v.item() if isinstance(v,torch.Tensor) else v) for k,v in row.items()} for row in s.records]));print('COMPLETE',a.sequence,flush=True)
if __name__=='__main__':main()
