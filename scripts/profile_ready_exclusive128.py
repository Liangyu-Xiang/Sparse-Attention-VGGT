"""Timeline audit: unique scope IDs, exclusive main-stream time, separate DMA."""
import argparse,ast,contextlib,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];FORK=Path('/tmp/zdrop_ready_history_v5_20261102')
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import numpy as np
import torch
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
import saf3r.patch.reuse_residual_exploration as impl
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record
base_tree=ast.parse((ROOT/'scripts/profile_ready_history_v5_128.py').read_text())
base_class=next(n for n in base_tree.body if isinstance(n,ast.ClassDef) and n.name=='Recorder')
exec(compile(ast.fix_missing_locations(ast.Module(body=[base_class],type_ignores=[])),'<base-timing-scopes>','exec'),globals())
BaseRecorder=Recorder

class TimelineRecorder(BaseRecorder):
 def __init__(self,state):
  super().__init__(state);self.next_id=0;self.origin=torch.cuda.Event(enable_timing=True);self.main_stream=torch.cuda.current_stream().cuda_stream
 @contextlib.contextmanager
 def scope(self,name,kind='function',layer=None,lane=None,nbytes=0):
  stream=torch.cuda.current_stream();lane=lane or ('main' if stream.cuda_stream==self.main_stream else 'copy')
  a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record(stream)
  begin=time.perf_counter();item=dict(id=self.next_id,parent_id=self.stack[-1]['id'] if self.stack else None,name=name,kind=kind,lane=lane,stream=int(stream.cuda_stream),layer=self.state.previous_layer+1 if layer is None else layer,nbytes=int(nbytes),start_event=a,end_event=b)
  self.next_id+=1;self.stack.append(item)
  try:yield
  finally:
   assert self.stack[-1] is item;self.stack.pop();b.record(stream);item['host_seconds']=time.perf_counter()-begin;self.events.append(item)
 def patch_function(self,module,name,groups=None,dma=False):
  tree=ast.parse(Path(module.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
  if groups:
   original=fn.body;result=[];pending=[];last=None
   def flush():
    nonlocal pending
    if pending:
     if last is None:result.extend(pending)
     else:result.append(ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='__trace',ctx=ast.Load()),attr='scope',ctx=ast.Load()),args=[ast.Constant(last)],keywords=[]))],body=pending))
     pending=[]
   for node in original:
    label=next((label for lo,hi,label in groups if lo<=node.lineno<=hi),None)
    if label!=last:flush();last=label
    pending.append(node)
   flush();fn.body=result
  if dma:
   tracer=self;hits=[]
   class DMA(ast.NodeTransformer):
    def visit_Expr(self,node):
     if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='copy_':
      direction='dma_h2d' if name=='prefetch' else 'dma_d2h';hits.append(direction)
      ctx=ast.parse(f"__trace.scope('{direction}', 'transfer', lane='copy', nbytes=s.ready_stages[j][:g].numel()*s.ready_stages[j].element_size())",mode='eval').body
      return ast.copy_location(ast.With(items=[ast.withitem(context_expr=ctx)],body=[node]),node)
     return node
   fn=DMA().visit(fn);assert len(hits)==1,(name,hits)
  single=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));module.__dict__['__trace']=self;space={}
  exec(compile(single,'<transparent-fine-scopes>','exec'),module.__dict__,space);setattr(module,name,space[name])
  (args.out/f'instrumented_{name}.py').write_text(ast.unparse(single)+'\n')
 def install(self,model):
  import saf3r.patch.reuse_ready_history as ready
  self.patch_function(ready,'build_prior',[(87,90,'prior_grid_transfer'),(91,91,'prior_expand_all_patch_keys'),(92,92,'prior_history_edge_update'),(93,93,'prior_cluster_mass_reduce'),(94,94,'prior_prefetch_enqueue'),(95,96,'prior_mass_validation')])
  self.patch_function(ready,'save_evidence',[(103,106,'calib_support_count'),(107,110,'calib_normalizer'),(111,111,'calib_grid_reduce'),(112,113,'calib_mass_validation'),(114,115,'calib_conditionals_encode'),(116,116,'calib_writeback_enqueue'),(117,119,'calib_grid_cache_write')])
  self.patch_function(ready,'prefetch',dma=True);self.patch_function(ready,'cache_written',dma=True)
  # Patch the actual observe function, then refresh its imported alias.
  import saf3r.patch.reuse_m5c as coarse
  tree=ast.parse(Path(coarse.__file__).read_text());obs=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
  labels=[]
  for node in obs.body:
   text=ast.unparse(node)
   label='coarse_probe_prepare'
   if text.startswith('grid ='):label='coarse_grid_qk'
   elif text.startswith('special ='):label='coarse_special_qk'
   elif text.startswith(('logz =','coarse =')):label='coarse_normalize'
   elif isinstance(node,ast.Return):continue
   labels.append((node.lineno,node.end_lineno,label))
  self.patch_function(coarse,'observe',labels);ready.observe=coarse.observe
  super().install(model)
  for name,label in [('cache_read','cache_read_or_h2d_wait'),('cache_write','cache_staging_write_wait'),('prefetch','cache_prefetch_host_submit'),('cache_written','cache_writeback_host_submit'),('predict','cached_edge_predict_pipeline')]:self.wrap(ready,name,label)
  import saf3r.triton.reuse_ready_history as kernels
  recorder=self
  class TimedLaunch:
   def __init__(self,original,label):self.original,self.label=original,label
   def __getitem__(self,grid):
    launch=self.original[grid]
    def run(*a,**kw):
     with recorder.scope(self.label,'kernel'):return launch(*a,**kw)
    return run
  for name,label in [('_row_lse','kernel_selected_logsumexp'),('_calibrate','kernel_grid_exp_segment_reduce'),('_store','kernel_conditional_quantize_pack'),('_predict','kernel_cached_probability_update'),('_write','kernel_history_scatter')]:setattr(kernels,name,TimedLaunch(getattr(kernels,name),label))
  import saf3r.triton.zdrop_history_states as history
  history._history=TimedLaunch(history._history,'kernel_history_nearest_and_loss')
 def serialize(self):
  result=[]
  for item in self.events:
   record={k:v for k,v in item.items() if k not in ['start_event','end_event']}
   record['start_seconds']=self.origin.elapsed_time(item['start_event'])/1000
   record['end_seconds']=self.origin.elapsed_time(item['end_event'])/1000
   record['seconds']=record['end_seconds']-record['start_seconds'];result.append(record)
  return sorted(result,key=lambda x:x['id'])

def main():
 if args.out.exists():raise ValueError('Fresh output required')
 args.out.mkdir();write_quality(args.out,quality_record({},'Exclusive timeline / DMA diagnostic only, no pose or recon_unposed GT evaluation'))
 manifest=json.loads((ROOT/'outputs/ready_history_v5_heads128_20261102/manifest.json').read_text());hashes=manifest['source']['after']['sha256']
 def unchanged():return all(hashlib.sha256((FORK/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
 assert unchanged();save(args.out/'status.json',dict(stage='loading'))
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 prior=ROOT/'outputs/zdrop_shared_rms_20261031/7scenes/heads/rep_combined_l2_direction_adaptive'
 inputs=json.loads((prior/'input.json').read_text());cfg=Dict(json.loads((prior/'config.json').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',inputs['image_paths']).cuda();config=impl.ReuseExploreConfig(**json.loads((FORK/'configs/evaluation/reuse_zdrop_history_states.json').read_text()));state=impl.install_reuse_residual_exploration(model,config)
 save(args.out/'input.json',dict(inputs,checkpoint=cfg.ckpt_path));save(args.out/'status.json',dict(stage='warmup'))
 torch.cuda.synchronize();start=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();warm=time.perf_counter()-start;del raw
 trace=TimelineRecorder(state);trace.install(model);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();trace.origin.record();end=torch.cuda.Event(enable_timing=True)
 save(args.out/'status.json',dict(stage='profiling'));start=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 end.record();torch.cuda.synchronize();wall=time.perf_counter()-start
 result=dict(complete=True,scope='heads128 single sequence, one warmup/one fine profile',wall_seconds=wall,cuda_interval_seconds=trace.origin.elapsed_time(end)/1000,warmup_seconds=warm,events=trace.serialize(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,source_unchanged=unchanged(),quality_evaluated=False,dense_speedup=None,dense_reason='Unchanged Dense/SAF3R not rerun')
 assert result['source_unchanged'];save(args.out/'timeline.json',result);save(args.out/'layers.json',state.records)
 finite={k:bool(torch.isfinite(v).all()) for k,v in raw.items() if torch.is_tensor(v)};assert all(finite.values()),finite
 save(args.out/'status.json',dict(stage='diagnostic_complete'));print('COMPLETE',wall,'events',len(result['events']),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if args.out.exists():
   write_quality(args.out,quality_record({},f'Exclusive timeline diagnostic failed: {type(exc).__name__}: {exc}; no GT evaluation'))
   save(args.out/'status.json',dict(stage='failed',error=f'{type(exc).__name__}: {exc}'))
  raise
