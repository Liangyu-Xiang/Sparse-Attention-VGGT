"""Optional diagnostic instrumentation. Never installed by normal inference.

CUDA events describe outer GPU phases. Graph-internal phases use timestamp
marker kernels, because Python capture times do not measure replay. All nested
phases report inclusive AND exclusive values. Component trials never write back.
"""
import ast,contextlib,csv,hashlib,inspect,json,statistics,subprocess,time
from pathlib import Path
import torch,triton
import triton.language as tl

@triton.jit
def _stamp(DATA,COUNT,CONTEXT,SITE:tl.constexpr,ENTER:tl.constexpr,CAP:tl.constexpr):
 slot=tl.atomic_add(COUNT,1)
 now=tl.inline_asm_elementwise('mov.u64 $0, %globaltimer;',constraints='=l',args=[],dtype=tl.uint64,is_pure=False,pack=1)
 layer=tl.load(CONTEXT);window=tl.load(CONTEXT+1)
 tl.store(DATA+slot*5,now,slot<CAP);tl.store(DATA+slot*5+1,SITE,slot<CAP);tl.store(DATA+slot*5+2,ENTER,slot<CAP);tl.store(DATA+slot*5+3,layer,slot<CAP);tl.store(DATA+slot*5+4,window,slot<CAP)

def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,default=str)+'\n')
def csvout(path,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)

def category(name):
 if name.startswith('A/'):return 'A'
 if name.startswith('B/'):return 'B'
 return 'C'

def instrument(module,qualname,prefix,rec,recurse=False):
 owner=module
 parts=qualname.split('.')
 for name in parts[:-1]:owner=getattr(owner,name)
 original=getattr(owner,parts[-1]);source=inspect.getsource(original);source=inspect.cleandoc(source) if source.startswith(' ') else source
 import textwrap
 tree=ast.parse(textwrap.dedent(inspect.getsource(original)));fn=tree.body[0];fn.decorator_list=[]
 class Wrap(ast.NodeTransformer):
  def block(self,body):
   result=[]
   for node in body:
    if isinstance(node,(ast.Return,ast.Import,ast.ImportFrom,ast.Raise,ast.Pass)) or (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant)):result.append(node);continue
    if recurse:
     for key in ('body','orelse','finalbody'):
      if hasattr(node,key):setattr(node,key,self.block(getattr(node,key)))
    ast.fix_missing_locations(node)
    label=prefix+'/L'+str(node.lineno)+'/'+ast.unparse(node).split('\n')[0][:140]
    result.append(ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='_detail_scope',ctx=ast.Load()),args=[ast.Constant(label)],keywords=[]))],body=[node],type_comment=None))
   return result
 fn.body=Wrap().block(fn.body);ast.fix_missing_locations(tree)
 namespace=module.__dict__;namespace['_detail_scope']=rec.scope
 exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
 replacement=namespace[fn.name];setattr(owner,parts[-1],replacement)
 rec.sources.append(dict(function=module.__name__+'.'+qualname,source=inspect.getsource(original),instrumented=ast.unparse(tree)))

class Detail:
 def __init__(self,state,out):
  self.state=state;self.out=Path(out);self.capacity=1000000;self.data=torch.empty((self.capacity,5),device='cuda',dtype=torch.uint64);self.count=torch.zeros(1,device='cuda',dtype=torch.int32);self.context=torch.zeros(2,device='cuda',dtype=torch.int32);self.sites={};self.sources=[];self.events=[];self.enabled=False;self.resources={};self.snapshot=None;self.state_snapshot=None
 def site(self,name):
  if name not in self.sites:self.sites[name]=len(self.sites)
  return self.sites[name]
 @contextlib.contextmanager
 def scope(self,name):
  site=self.site(name);capture=torch.cuda.is_current_stream_capturing()
  if not self.enabled and not capture:yield;return
  event=None
  if not capture:
   st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();event=(name,st,en)
  with torch.profiler.record_function('detail/'+name):
   torch.cuda.nvtx.range_push(name)
   _stamp[(1,)](self.data,self.count,self.context,site,1,self.capacity,num_warps=1)
   try:yield
   finally:
    _stamp[(1,)](self.data,self.count,self.context,site,0,self.capacity,num_warps=1);torch.cuda.nvtx.range_pop()
  if event:en.record();self.events.append(event)
 def wrap(self,module,name,label):
  original=getattr(module,name)
  def run(*args,**kwargs):
   with self.scope(label):return original(*args,**kwargs)
  setattr(module,name,run)
 def install(self,model):
  import saf3r.patch.reuse_residual_exploration as impl
  import saf3r.patch.reuse_window_graph as window
  rec=self
  # Rebuild instrumentation-only graphs so replay contains the GPU markers.
  from collections import OrderedDict
  self.state._reused_window_graphs=OrderedDict()
  impl.stage=lambda state,name:rec.scope('C/stage/'+name)
  self.wrap(impl,'prepare_qkv','A/QKV_LayerNorm_RoPE')
  instrument(impl,'ReuseResidualExploration.forward','C/forward',self,recurse=True)
  original=impl.ReuseResidualExploration.forward
  def forward(att,*args,**kwargs):
   rec.context[0].fill_(att.layer);rec.context[1].fill_(-1)
   with rec.scope('C/global_sparse_wrapper'):return original(att,*args,**kwargs)
  impl.ReuseResidualExploration.forward=forward
  for block in model.aggregator.global_blocks:
   self.wrap(block.attn.original.proj,'forward','A/global_output_projection')
   self.wrap(block.attn.original.proj_drop,'forward','A/global_output_dropout')
  # Existing direct wrapper has exactly two disjoint scopes: AV and export.
  self.state.direct_scope=lambda name:rec.scope(('B/' if name=='direct_av' else 'C/C3/')+name)
  modules=[('saf3r.patch.reuse_probe_cache',['observe'],'C/C2'),('saf3r.patch.reuse_direct_coarse_observation',['observe'],'C/C2'),('saf3r.patch.reuse_probability_history',['build_prior','observe_execution','save_evidence'],'C/C5_C4'),('saf3r.triton.reuse_probability_history',['store','predict'],'C/C4'),('saf3r.patch.reuse_ready_history',['ensure_cache','cache_read','cache_write','cache_written','prefetch','trim','release'],'C/C4_cache'),('saf3r.patch.reuse_inference_ops',['gate','pack','transfer','diagnostics'],'C/C6_C8_C9_C10'),('saf3r.triton.reuse_fused_states',['states'],'C/C6'),('saf3r.triton.reuse_compact',['prepare_geometry','budget'],'C/C1_C7')]
  import importlib
  for mod,names,prefix in modules:
   module=importlib.import_module(mod)
   for name in names:instrument(module,name,prefix+'/'+name,self,recurse=(name=='budget'))
  # Resolve the actual imported observation aliases, not only module attributes.
  coarse=importlib.import_module('saf3r.patch.reuse_direct_coarse_observation');coarse._observe=importlib.import_module('saf3r.patch.reuse_probe_cache').observe
  importlib.import_module('saf3r.patch.reuse_probability_history').observe=coarse.observe
  # Actual imported budget alias must target the instrumented function.
  window.budget=importlib.import_module('saf3r.triton.reuse_compact').budget
  instrument(impl,'ReuseExploreState.prepare','C/C1_C2/prepare',self)
  instrument(window,'eager','C/graph_selection',self)
  instrument(window,'Window.update_constants','C/C10/graph_constants',self)
  instrument(window,'Window.__call__','C/C10/graph_replay',self)
  self.wrap(window.Window,'__init__','C/C10/graph_initialization')
  # Correct actual module aliases after AST instrumentation.
  call=window.Window.__call__
  def captured_window(instance,*args,**kwargs):
   result=call(instance,*args,**kwargs)
   if self.enabled and self.state_snapshot is None:self.state_snapshot=instance
   return result
  window.Window.__call__=captured_window
  observer=importlib.import_module('saf3r.patch.reuse_probability_history');observe=observer.observe
  def observed(s,q,k,start,g,scale):
   self.context[1].fill_(start)
   return observe(s,q,k,start,g,scale)
  observer.observe=observed
  backend=importlib.import_module('saf3r.triton.reuse_direct_sparse_attention');direct=backend.direct_sparse_attention
  def attention(*args,**kwargs):
   if self.enabled and self.snapshot is None and self.state.previous_layer+1==12:
    # References retained only for an independent same-input microbenchmark.
    self.snapshot=(args,kwargs)
   return direct(*args,**kwargs)
  backend.direct_sparse_attention=attention
  class Launch:
   def __init__(self,kernel,label):self.kernel=kernel;self.label=label
   def __getitem__(self,grid):
    launch=self.kernel[grid]
    def run(*args,**kwargs):
     compiled=launch(*args,**kwargs)
     if not torch.cuda.is_current_stream_capturing():
      key=self.label+str(tuple(x.shape for x in args if torch.is_tensor(x)))
      if key not in rec.resources:rec.resources[key]=dict(kernel=self.label,registers_per_thread=getattr(compiled,'n_regs',None),spills=getattr(compiled,'n_spills',None),shared_bytes=getattr(getattr(compiled,'metadata',None),'shared',None),occupancy=None,occupancy_reason='Hardware counters unavailable; compiler resources only')
     return compiled
    return run
  for mod,names in [('saf3r.triton.reuse_direct_sparse_attention',['_attention','_probe']),('saf3r.triton.reuse_fused_states',['_states'])]:
   module=importlib.import_module(mod)
   for name in names:setattr(module,name,Launch(getattr(module,name),mod+'.'+name))
  save(self.out/'instrumented_sources.json',self.sources)
 def reset(self):
  self.count.zero_();self.events=[];self.enabled=True
 def finish(self):
  self.enabled=False;torch.cuda.synchronize();n=int(self.count.item());assert n<self.capacity,n
  stamps=self.data[:n].cpu().tolist();names={v:k for k,v in self.sites.items()};stack=[];rows=[]
  for clock,site,enter,layer,window in stamps:
   if enter:stack.append(dict(name=names[site],site=site,begin=clock,children=0,layer=int(layer),window=int(window)))
   else:
    assert stack and stack[-1]['site']==site,(site,stack[-1] if stack else None)
    row=stack.pop();duration=(clock-row['begin'])/1e9;exclusive=duration-row.pop('children')
    if stack:stack[-1]['children']+=duration
    rows.append(dict(name=row['name'],category=category(row['name']),layer=row['layer'],window=row['window'],inclusive_seconds=duration,exclusive_seconds=max(0.,exclusive)))
  assert not stack
  csvout(self.out/'per_window.csv',rows)
  grouped={};layers={}
  for r in rows:
   k=r['name'];v=grouped.setdefault(k,dict(operation=k,category=r['category'],calls=0,gpu_inclusive_seconds=0.,gpu_exclusive_seconds=0.,includes_children=True,memory_bytes=None,memory_kind='N/A: no hardware counters'))
   v['calls']+=1;v['gpu_inclusive_seconds']+=r['inclusive_seconds'];v['gpu_exclusive_seconds']+=r['exclusive_seconds']
   k=(r['layer'],r['category']);layers[k]=layers.get(k,0.)+r['exclusive_seconds']
  for v in grouped.values():v['mean_call_seconds']=v['gpu_inclusive_seconds']/v['calls']
  csvout(self.out/'operations.csv',list(grouped.values()));csvout(self.out/'per_layer.csv',[dict(layer=k[0],category=k[1],exclusive_seconds=v) for k,v in layers.items()]);csvout(self.out/'kernel_resources.csv',list(self.resources.values()))
  events=[dict(name=name,cuda_event_inclusive_seconds=st.elapsed_time(en)/1000,includes_children=True) for name,st,en in self.events];save(self.out/'cuda_events.json',events)
  return rows,list(grouped.values())

def run_detail(model,state,images,out,baseline):
 out=Path(out)
 import os,platform,subprocess,shutil
 root=Path(__file__).resolve().parents[1]
 save(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,python=platform.python_version(),gpu=str(torch.cuda.get_device_properties(0)),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),nsys=shutil.which('nsys'),hardware_counters='Not measured by this pass; do not infer occupancy/DRAM from compiler resources'))
 save(out/'source_commit.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=root,text=True).strip(),worktree=str(root),tracked_diff=subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True).splitlines(),config=state.config.__dict__,algorithm_files_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'saf3r').rglob('*.py')}))
 rec=Detail(state,out);rec.install(model)
 # Compile marker sites and capture instrumented graphs outside warmed timings.
 rec.enabled=True;begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):warm=model(images)
 torch.cuda.synchronize();capture_wall=time.perf_counter()-begin;del warm
 rec.reset();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();wall=time.perf_counter()-begin
 peak=torch.cuda.max_memory_allocated()/2**30
 rows,operations=rec.finish();del raw
 # Trace is independent from the stage timing above. No CPU/GPU times are added.
 rec.enabled=False;torch.cuda.synchronize();begin=time.perf_counter()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],profile_memory=True,record_shapes=False,with_stack=False) as prof:
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize()
 tracewall=time.perf_counter()-begin;del raw
 import tempfile,shutil,os
 trace_path=Path(tempfile.gettempdir())/('direct_probability_trace_'+str(os.getpid())+'.json')
 prof.export_chrome_trace(str(trace_path))
 target=out/'profiler_trace.json'
 if target.exists() or target.is_symlink():target.unlink()
 target.symlink_to(trace_path)
 import gzip
 with trace_path.open('rb') as source,gzip.open(out/'profiler_trace.json.gz','wb',compresslevel=1) as compressed:shutil.copyfileobj(source,compressed,length=2**20)
 selfrows=[];launches=[];allocations=[]
 for e in prof.key_averages():
  selfrows.append(dict(operation=e.key,calls=e.count,gpu_self_seconds=e.self_device_time_total/1.e6,cpu_self_seconds=e.self_cpu_time_total/1.e6,cpu_total_seconds=e.cpu_time_total/1.e6,device_memory_net_bytes=e.self_device_memory_usage,includes_children=False))
 for e in prof.events():
  if e.name in ('cudaLaunchKernel','cudaGraphLaunch','cuLaunchKernelEx','cudaStreamSynchronize','cudaDeviceSynchronize','cudaMemcpyAsync','cudaMalloc','cudaFree'):
   launches.append(dict(operation=e.name,cpu_start_us=e.time_range.start,cpu_duration_us=e.time_range.elapsed_us(),thread=e.thread))
  if e.name=='[memory]':allocations.append(dict(operation=e.name,cpu_time_us=e.time_range.start,net_device_bytes=e.device_memory_usage,net_cpu_bytes=e.cpu_memory_usage))
 csvout(out/'cpu_operator_self_time.csv',sorted(selfrows,key=lambda r:r['cpu_self_seconds'],reverse=True));
 kernels={}
 for event in prof.events():
  if event.device_type==torch.autograd.DeviceType.CUDA:
   row=kernels.setdefault(event.name,dict(kernel=event.name,calls=0,gpu_kernel_self_seconds=0.,includes_children=False));row['calls']+=1;row['gpu_kernel_self_seconds']+=event.time_range.elapsed_us()/1.e6
 csvout(out/'kernel_self_time.csv',sorted(kernels.values(),key=lambda r:r['gpu_kernel_self_seconds'],reverse=True));csvout(out/'kernel_launches.csv',launches);csvout(out/'memory_allocations.csv',allocations)
 # Residual of complete GPU span belongs to shared model work, with explicit
 # caveat: stream-idle and CPU scheduling are in phase time, not kernel self.
 totals={c:sum(r['exclusive_seconds'] for r in rows if r['category']==c) for c in 'ABC'}
 covered=sum(totals.values());totals['A']+=max(0.,wall-covered)
 top=[dict(category=c,description={'A':'VGGT shared compute plus uninstrumented forward gaps','B':'Direct sparse QK/online-softmax/AV and output','C':'Sparse algorithm overhead including instrumentation'}[c],gpu_phase_seconds=totals[c],kernel_self_seconds=None,includes_children=False,wall_residual_in_A=c=='A') for c in 'ABC']
 csvout(out/'top_level.csv',top)
 graph=[r for r in operations if 'graph_' in r['operation'] or 'graph/' in r['operation']];csvout(out/'graph_overhead.csv',graph)
 save(out/'complete_forward.json',dict(complete=True,baseline_warmed_median_seconds=baseline['median_seconds'],instrumented_warmed_wall_seconds=wall,overhead_seconds=wall-baseline['median_seconds'],overhead_percent=100*(wall/baseline['median_seconds']-1),instrumented_initial_graph_compile_warmup_seconds=capture_wall,operator_trace_wall_seconds=tracewall,peak_allocated_gib=peak,instrumentation_bytes=rec.data.numel()*rec.data.element_size(),top_level=top,quality={m:None for m in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Profiling only; no new geometric quality evaluation',dense_speedup=None,dense_reason='Unchanged Dense not rerun'))
 save(out/'kernel_launch_summary.json',dict(cuda_launches=sum(r['operation']=='cudaLaunchKernel' for r in launches),graph_launches=sum(r['operation']=='cudaGraphLaunch' for r in launches),synchronizations=sum('Synchronize' in r['operation'] for r in launches)))
 from scripts.direct_detail_microbench import run as micros
 micros(rec,out)
 return dict(wall=wall,top_level=top,operations=operations)
