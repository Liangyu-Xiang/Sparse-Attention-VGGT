"""Changed methods only; historical Dense timing is NOT a paired speedup."""
import argparse,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images

def build(config_path):
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(cfg,'cuda');data=json.loads(config_path.read_text())
 if 'mode' in data:
  from saf3r.patch.reuse_lightweight import LightweightConfig,install_lightweight_reuse
  config=LightweightConfig(**data);state=install_lightweight_reuse(model,config)
 else:
  from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
  config=ReuseExploreConfig(**data);state=install_reuse_residual_exploration(model,config)
 return model,state

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--profile',action='store_true');a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
 torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 model,state=build(a.config);resources={}
 from saf3r.triton import reuse_direct_sparse_attention as backend
 original_kernel=backend._attention
 class ResourceLaunch:
  def __getitem__(self,grid):
   launch=original_kernel[grid]
   def run(*args,**kwargs):
    compiled=launch(*args,**kwargs)
    key=str((args[12:18],kwargs.get('num_warps'),kwargs.get('num_stages')))
    if key not in resources:
     resources[key]=dict(registers_per_thread=getattr(compiled,'n_regs',None),spills=getattr(compiled,'n_spills',None),shared_bytes=getattr(compiled.metadata,'shared',None),warps=kwargs.get('num_warps'),stages=kwargs.get('num_stages'))
    return compiled
   return run
 backend._attention=ResourceLaunch()
 images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();samples=[];cold=None
 for i in range(4):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize();value=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
  assert len(state.records)==24 and torch.isfinite(raw['depth']).all() and torch.isfinite(raw['pose_enc']).all()
  del raw
  if i:samples.append(value)
  else:cold=value
  (a.output_dir/'progress.json').write_text(json.dumps(dict(iteration=i,samples=samples,cold=cold),indent=2));print(i,value,flush=True)
 result=dict(complete=True,median_seconds=statistics.median(x['seconds'] for x in samples),samples=samples,cold=cold,config=json.loads(a.config.read_text()),quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Performance-only full-forward diagnostic',dense_speedup=None,dense_reason='User prohibited unchanged Dense rerun; historical heads128 Dense 12.216022619977593 s is not paired',historical_dense_seconds=12.216022619977593)
 if a.profile:
  import contextlib
  direct_events=[]
  @contextlib.contextmanager
  def direct_scope(name):
   st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
   try:yield
   finally:en.record();direct_events.append((name,st,en))
  state.direct_scope=direct_scope
  state.profiling=True;state.profile_events=[];begin=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
  torch.cuda.synchronize();wall=time.perf_counter()-begin;groups={}
  for layer,name,start,end in state.profile_events:groups[name]=groups.get(name,0.)+start.elapsed_time(end)/1000
  result['profile']=dict(wall_seconds=wall,exclusive_seconds=groups,unattributed_seconds=wall-sum(groups.values()),scope='Disjoint top-level stages; direct_parts are nested inside execution_and_evidence,never added twice')
  direct_parts={}
  for name,st,en in direct_events:direct_parts[name]=direct_parts.get(name,0.)+st.elapsed_time(en)/1000
  result['profile']['direct_parts']=direct_parts
  result['profile']['direct_calls']=sum(name=='legacy_full_query_grid_and_probe_observation' for name,_,_ in direct_events)
 result['kernel_resources']=resources
 (a.output_dir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
