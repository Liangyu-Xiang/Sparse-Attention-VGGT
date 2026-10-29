"""Single-sequence CUDA-event diagnostic; quality and paired speedup unavailable."""
import sys,json,time,hashlib
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record
import saf3r.triton.reuse_explore_av as kernels
OUT=ROOT/'outputs/reuse_compact_heads128_profile_20261029'
REFERENCE=ROOT/'outputs/reuse_rep_revision_7scenes128_20261029/heads/rep_combined'
class TimedLaunch:
 def __init__(self,original,name,events):self.original,self.name,self.events=original,name,events
 def __getitem__(self,grid):
  launch=self.original[grid]
  def call(*args,**kwargs):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   result=launch(*args,**kwargs);b.record();self.events.append((self.name,a,b));return result
  return call

def run():
 if OUT.exists():raise RuntimeError('Fresh profile output required')
 OUT.mkdir(parents=True)
 write_quality(OUT,quality_record({},'Stage-profiling diagnostic only; pose and recon_unposed not evaluated; no matched instrumented Dense timing'))
 save(OUT/'status.json',dict(stage='loading'))
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 inputs=json.loads((REFERENCE/'input.json').read_text());cfg=Dict(json.loads((REFERENCE/'config.json').read_text())['model'])
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',inputs['image_paths']).cuda()
 assert list(images.shape)==inputs['shape']
 config=ReuseExploreConfig(**json.loads((ROOT/'configs/evaluation/reuse_rep_compact.json').read_text()))
 state=install_reuse_residual_exploration(model,config)
 raw=None
 save(OUT/'input.json',inputs)
 save(OUT/'status.json',dict(stage='warmup'))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();del raw
 state.profiling=True;launch_events=[];function_events=[]
 import saf3r.triton.reuse_compact as implementation
 def wrap_function(original,name):
  def call(*args,**kwargs):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
   result=original(*args,**kwargs);b.record()
   function_events.append((state.previous_layer+1,name,a,b));return result
  return call
 for name in ('prepare_geometry','top_history','losses','finalize','budget','pack','transfer','diagnostics'):
  setattr(implementation,name,wrap_function(getattr(implementation,name),name))
 kernels._attention=TimedLaunch(kernels._attention,'attention_online',launch_events)
 kernels._mass=TimedLaunch(kernels._mass,'evidence_second_qk',launch_events)
 save(OUT/'status.json',dict(stage='profiling'))
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();start.record()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 end.record();torch.cuda.synchronize();wall=time.perf_counter()-begin
 stage=defaultdict(float);per_layer=defaultdict(lambda:defaultdict(float))
 for layer,name,a,b in state.profile_events:
  seconds=a.elapsed_time(b)/1000;stage[name]+=seconds;per_layer[layer][name]+=seconds
 for name,a,b in launch_events:stage[name]+=a.elapsed_time(b)/1000
 allocation_parent=stage.pop('reuse_and_budget')
 function_seconds=defaultdict(float);function_calls=defaultdict(int)
 for layer,name,a,b in function_events:
  seconds=a.elapsed_time(b)/1000;function_seconds[name]+=seconds;function_calls[name]+=1;per_layer[layer][name]+=seconds
 geometry_seconds=function_seconds.pop('prepare_geometry',0.)
 stage['pooling']-=geometry_seconds
 stage['geometry_prepare']=geometry_seconds
 stage.update(function_seconds)
 stage['allocation_setup_and_gaps']=allocation_parent-sum(function_seconds.values())
 total_gpu=start.elapsed_time(end)/1000
 execution=stage.pop('execution_and_evidence')
 stage['execution_setup_and_gaps']=execution-stage['attention_online']-stage['evidence_second_qk']
 stage['other_model_operations']=total_gpu-sum(stage.values())
 reference=json.loads((REFERENCE/'forward.json').read_text())
 result=dict(diagnostic=True,dataset='7scenes',scene='heads',frames=128,input=inputs,method='compact combined reuse 2x2',warmup=1,profile_repeats=1,
  wall_seconds=wall,cuda_forward_interval_seconds=total_gpu,stages_seconds=dict(stage),stage_fraction={k:v/total_gpu for k,v in stage.items()},per_layer_seconds={str(k):dict(v) for k,v in per_layer.items()},
  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
  hardware=dict(gpu=torch.cuda.get_device_name(),physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__),
  uninstrumented_reference=dict(path=str(REFERENCE/'forward.json'),median_seconds=reference['timing']['median_seconds'],samples=reference['timing']['samples'],physical_gpu=0),
  allocation_parent_seconds=allocation_parent,function_calls=dict(function_calls),
  notes=['CUDA-event elapsed intervals include launch/queue gaps; not hardware active-kernel-only durations.','Original second-QK mass recovery is timed separately from attention using launch events; no per-layer synchronization.','All event instrumentation overhead remains in diagnostic total; this diagnostic is not a speedup comparison.','Other model operations is the measured total CUDA-event interval minus disjoint profiled stages: includes DINO, local attention, FFN, projections, heads, routing outside tagged intervals and launch gaps.','Reference wall timing is an earlier uninstrumented combined-method run on GPU0, NOT a contemporaneous speed comparison. Do not apply compact profile percentages to it as exact stage durations.'])
 save(OUT/'profile.json',result)
 lines=['# heads 128-frame compact combined reuse: detailed stage profiling','',f'Profile wall time {wall:.3f}s; CUDA forward interval {total_gpu:.3f}s. One warmup, one profiled forward on GPU7. Peak allocated {result["peak_allocated_gib"]:.3f} GiB.','', '| Stage | Seconds | Share of CUDA forward interval |','|---|---:|---:|']
 for name,seconds in sorted(stage.items(),key=lambda x:-x[1]):lines.append(f'| {name} | {seconds:.6f} | {100*seconds/total_gpu:.3f}% |')
 lines+=['',f'Historical uninstrumented combined reference on GPU0: warmed median {reference["timing"]["median_seconds"]:.3f}s (1 warmup, 3 measurements).','']+result['notes']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n');save(OUT/'status.json',dict(stage='diagnostic_complete'))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as exc:
  if OUT.exists():save(OUT/'status.json',dict(stage='failed',error=f'{type(exc).__name__}: {exc}'))
  raise
