"""Same-method before/after128 latency, output repeatability, and transparent timing."""
import argparse,ast,contextlib,hashlib,json,math,statistics,sys,time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['before','after'],required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
FORK=Path('/tmp/zdrop_history_states_20261102' if args.variant=='before' else '/tmp/zdrop_history_cleanup_20261102')
sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import numpy as np
import torch
from addict import Dict
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
import saf3r.patch.reuse_residual_exploration as impl


def output_cpu(raw):return {k:v.detach().float().cpu().clone() for k,v in raw.items() if torch.is_tensor(v) and k!='images'}
def difference(current,reference):
 result={}
 for k,a in current.items():
  b=reference[k];delta=a.double()-b.double()
  result[k]=dict(equal=bool(torch.equal(a,b)),relative_l2=float(delta.norm()/b.double().norm().clamp_min(1e-30)),mean_abs=float(delta.abs().mean()),max_abs=float(delta.abs().max()),shape=list(a.shape))
 return result

class Recorder:
 def __init__(self,state):self.state=state;self.events=[];self.stack=[];self.handles=[]
 @contextlib.contextmanager
 def scope(self,name,kind='function',layer=None):
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();begin=time.perf_counter();item=dict(name=name,kind=kind,layer=self.state.previous_layer+1 if layer is None else layer,parent=self.stack[-1]['name'] if self.stack else None)
  self.stack.append(item)
  try:yield
  finally:
   b.record();item.update(cpu_seconds=time.perf_counter()-begin,start=a,end=b);self.stack.pop();self.events.append(item)
 def wrap(self,module,name,label):
  original=getattr(module,name)
  def wrapped(*a,**kw):
   use=label() if callable(label) else label
   with self.scope(use):return original(*a,**kw)
  setattr(module,name,wrapped)
 def attach(self,module,label,layer=None):
  active=[]
  def before(m,a):ctx=self.scope(label,'module',layer);ctx.__enter__();active.append(ctx)
  def after(m,a,out):active.pop().__exit__(None,None,None)
  self.handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
 def install(self,model):
  # Existing stage contexts only record events; no intermediate synchronization.
  impl.stage=lambda state,name:self.scope(name,'stage')
  # Add a transparent scope around the existing inline RMS Drop statements.
  tree=ast.parse((FORK/'saf3r/patch/reuse_residual_exploration.py').read_text())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReuseResidualExploration')
  fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
  class TagDrop(ast.NodeTransformer):
   def visit_If(self,node):
    self.generic_visit(node)
    if isinstance(node.test,ast.Attribute) and isinstance(node.test.value,ast.Name) and node.test.value.id=='c' and node.test.attr=='direction_drop':
     tag=ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='stage',ctx=ast.Load()),args=[ast.Name(id='s',ctx=ast.Load()),ast.Constant(value='drop_vector_sum')],keywords=[]))],body=node.body);node.body=[ast.copy_location(tag,node)]
    return node
  fn=TagDrop().visit(fn);single=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));namespace={}
  exec(compile(single,'<timing-only-forward>','exec'),impl.__dict__,namespace)
  impl.ReuseResidualExploration.forward=namespace['forward']
  (args.out/'instrumented_forward.py').write_text(ast.unparse(single)+'\n')
  import saf3r.patch.adaptive_value_clustering as cluster
  for name,label in [('centers_from_labels','cluster_centroid_reduce'),('full_assignment','cluster_full_search'),('center_graph','cluster_neighbor_graph'),('local_assignment','cluster_local_search'),('repair_empty','cluster_empty_repair')]:self.wrap(cluster,name,label)
  compact=__import__('saf3r.triton.reuse_compact',fromlist=['x'])
  self.wrap(compact,'prepare_geometry','geometry_prepare')
  self.wrap(compact,'budget','budget_solve4')
  rule=impl.history_rule
  for name,label in [('gate','history_gate'),('states','state_losses'),('decode','state_decode'),('pack','support_pack'),('transfer','mass_transfer'),('diagnostics','state_diagnostics')]:self.wrap(rule,name,label)
  import saf3r.triton.zdrop_history_cleanup as fast_kernel
  import saf3r.triton.zdrop_history_states as hist_kernel
  self.wrap(fast_kernel,'rep_loss','rep_loss');self.wrap(hist_kernel,'history_loss','history_loss_and_owners')
  import saf3r.patch.reuse_m5c as m5c
  self.wrap(m5c,'build_prior','history_probability');self.wrap(m5c,'save_evidence','m5c_evidence_statistics')
  self.wrap(m5c,'selected_probe_logits',lambda:'previous_history_probe_qk' if any(x['name']=='history_probability' for x in self.stack) else 'current_history_probe_qk')
  import saf3r.triton.reuse_explore_av as attn
  recorder=self
  class TimedLaunch:
   def __init__(self,original,label):self.original,self.label=original,label
   def __getitem__(self,grid):
    launch=self.original[grid]
    def run(*a,**kw):
     with recorder.scope(self.label,'kernel'):return launch(*a,**kw)
    return run
  attn._attention=TimedLaunch(attn._attention,'sparse_attention_kernel');attn._mass=TimedLaunch(attn._mass,'attention_mass_second_qk')
  self.attach(model.aggregator.patch_embed,'patch_embed_dino',-1)
  for name in ['camera_head','depth_head','point_head','track_head']:
   module=getattr(model,name,None)
   if module is not None:self.attach(module,name,-1)
  for kind in ['frame','global']:
   for i,block in enumerate(getattr(model.aggregator,kind+'_blocks')):
    self.attach(block,kind+'_block',i)
    self.attach(block.attn,kind+'_attention',i);self.attach(block.mlp,kind+'_mlp',i)
    att=getattr(block.attn,'original',block.attn)
    for n in ['qkv','proj']:self.attach(getattr(att,n),kind+'_'+n,i)
 def serialize(self):
  rows=[]
  for item in self.events:
   d={k:v for k,v in item.items() if k not in ['start','end']};d['seconds']=item['start'].elapsed_time(item['end'])/1000;rows.append(d)
  return rows


def main():
 out=args.out
 if out.exists():raise ValueError('Fresh output required')
 out.mkdir(parents=True)
 write_quality(out,quality_record({},'128-frame same-method implementation/timing diagnostic only; no pose or recon_unposed evaluation, no Dense/SAF3R run'))
 save(out/'status.json',dict(stage='loading',variant=args.variant))
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 ref=ROOT/'outputs/zdrop_shared_rms_20261031/7scenes/heads/rep_combined_l2_direction_adaptive'
 inputs=json.loads((ref/'input.json').read_text());cfg=Dict(json.loads((ref/'config.json').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',inputs['image_paths']).cuda();assert list(images.shape)==inputs['shape'] and len(images)==128
 config=impl.ReuseExploreConfig(**json.loads((FORK/'configs/evaluation/reuse_zdrop_history_states.json').read_text()));state=impl.install_reuse_residual_exploration(model,config)
 save(out/'input.json',dict(inputs,checkpoint=cfg.ckpt_path));save(out/'config.json',asdict(config))
 tensor_dir=Path('/tmp/zdrop_cleanup_outputs_20261102');tensor_dir.mkdir(exist_ok=True)
 first=None;previous=None;samples=[];cold=[];jitter=[];comparisons=[];raw=None
 reference=torch.load(tensor_dir/'before_first.pt',map_location='cpu',weights_only=True) if args.variant=='after' else None
 for iteration in range(4):
  del raw;raw=None
  save(out/'status.json',dict(stage='warmup' if iteration==0 else 'timing',iteration=iteration,variant=args.variant))
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize();sample=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
  (cold if iteration==0 else samples).append(sample);print('FORWARD',args.variant,iteration,sample,flush=True)
  if iteration:
   cpu=output_cpu(raw)
   if first is None:
    first=cpu;torch.save(cpu,tensor_dir/(args.variant+'_first.pt'))
   else:jitter.append(difference(cpu,first))
   if reference is not None:comparisons.append(difference(cpu,reference))
  save(out/f'layers_{iteration}.json',state.records)
 forward=dict(variant=args.variant,frames=128,scene='heads',dataset='7scenes',timing=dict(warmup=1,repeats=3,median_seconds=statistics.median(s['seconds'] for s in samples),samples=samples,cold_samples=cold),config=asdict(config),hardware=dict(gpu=torch.cuda.get_device_name(),physical_gpu=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__),same_variant_repeat_jitter=jitter,before_after_output_differences=comparisons,algorithm_parameters_changed=False)
 save(out/'forward.json',forward)
 if args.variant=='after':
  save(out/'status.json',dict(stage='profiling',variant=args.variant));rec=Recorder(state);rec.install(model)
  del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin_ev=torch.cuda.Event(enable_timing=True);end_ev=torch.cuda.Event(enable_timing=True);begin_ev.record();start=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  end_ev.record();torch.cuda.synchronize();wall=time.perf_counter()-start
  prof=dict(wall_seconds=wall,cuda_interval_seconds=begin_ev.elapsed_time(end_ev)/1000,events=rec.serialize(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,profile_vs_uninstrumented_difference=difference(output_cpu(raw),first),notes=['CUDA intervals include CPU launch/queue idle gaps; not pure active-kernel time.','Nested inclusive intervals must not be added twice.','Timing instrumentation is transparent context wrappers/hooks only; no extra per-layer synchronization.','Profile total includes instrumentation overhead; uninstrumented median is reported separately.'])
  save(out/'profile.json',prof);save(out/'profile_layers.json',state.records)
 save(out/'status.json',dict(stage='diagnostic_complete',variant=args.variant))
 print('COMPLETE',args.variant,flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if args.out.exists():
   write_quality(args.out,quality_record({},f'Implementation/profile diagnostic failed: {type(exc).__name__}: {exc}; no quality evaluation'))
   save(args.out/'status.json',dict(stage='failed',error=f'{type(exc).__name__}: {exc}'))
  raise
