"""Same-boundary audit of VGGT common operators versus attention/policy plumbing.

Diagnostic only:baseline timings use unmodified forward functions. Instrumented
forward functions preserve arithmetic;no masks,retention or weights are changed.
All six geometric metrics are unavailable;existing complete controls are reused.
"""
import argparse,contextlib,hashlib,inspect,json,statistics,sys,textwrap,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images


def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def meta(value):
 if torch.is_tensor(value):return dict(shape=list(value.shape),stride=list(value.stride()),dtype=str(value.dtype),contiguous=value.is_contiguous(),storage_ptr=value.untyped_storage().data_ptr(),data_ptr=value.data_ptr())
 if isinstance(value,(list,tuple)):return [meta(x) for x in value[:4]]
 if isinstance(value,dict):return {k:meta(v) for k,v in list(value.items())[:6]}
 return type(value).__name__

class Recorder:
 def __init__(self):self.enabled=False;self.events=[];self.stack=[];self.notes={};self.layer=None;self.handles=[]
 def note(self,name,value):
  if self.enabled:self.notes.setdefault(name,value)
 @contextlib.contextmanager
 def scope(self,name,kind):
  if not self.enabled:yield;return
  node=dict(name=name,kind=kind,parent=self.stack[-1] if self.stack else None,layer=self.layer,start=torch.cuda.Event(enable_timing=True),end=torch.cuda.Event(enable_timing=True),cpu_start=time.perf_counter());idx=len(self.events);self.events.append(node);node['start'].record();self.stack.append(idx)
  try:yield
  finally:self.stack.pop();node['end'].record();node['cpu_seconds']=time.perf_counter()-node.pop('cpu_start')
 def module(self,module,name,kind='common',layer=None,global_wrapper=False,rope=False):
  contexts=[];previous=[]
  def pre(m,args):
   if rope and self.layer is None:contexts.append(None);return
   if global_wrapper:previous.append(self.layer);self.layer=layer
   label=name if not rope else f'global_rope.L{self.layer}'
   cm=self.scope(label,kind);cm.__enter__();contexts.append(cm);self.note(label+'.input',meta(args))
  def post(m,args,result):
   cm=contexts.pop()
   if cm is None:return
   label=name if not rope else f'global_rope.L{self.layer}'
   self.note(label+'.output',meta(result));cm.__exit__(None,None,None)
   if global_wrapper:self.layer=previous.pop()
  self.handles.extend([module.register_forward_pre_hook(pre),module.register_forward_hook(post)])
 def finish(self,wall):
  torch.cuda.synchronize();rows=[]
  for i,n in enumerate(self.events):
   value={k:v for k,v in n.items() if k not in ('start','end')};value.update(index=i,seconds=n['start'].elapsed_time(n['end'])/1000);rows.append(value)
  for row in rows:
   children=[x for x in rows if x['parent']==row['index']];row['self_seconds']=max(0.,row['seconds']-sum(x['seconds'] for x in children))
  groups={}
  for row in rows:
   key=row['name'].split('.L')[0];g=groups.setdefault(key,dict(name=key,kind=row['kind'],calls=0,inclusive_seconds=0.,self_seconds=0.,cpu_seconds=0.));g['calls']+=1
   for k in ('self_seconds','cpu_seconds'):g[k]+=row[k]
   g['inclusive_seconds']+=row['seconds']
  common=[]
  for row in rows:
   if row['kind']!='common':continue
   parent=row['parent'];inside=False
   while parent is not None:
    if rows[parent]['kind']=='common':inside=True;break
    parent=rows[parent]['parent']
   if not inside:common.append(row)
  return dict(wall_seconds=wall,events=rows,groups=list(groups.values()),common_module_inclusive_seconds=sum(x['seconds'] for x in common),common_global_residual_self_seconds=sum(x['self_seconds'] for x in rows if x['kind']=='common_residual'),root_gpu_seconds=rows[0]['seconds'])


def install_audit(model,state,method,audit):
 audit.module(model.aggregator.patch_embed,'image_backbone')
 for i,b in enumerate(model.aggregator.frame_blocks):audit.module(b,f'frame_block.L{i}')
 seen_rope=set()
 for i,b in enumerate(model.aggregator.global_blocks):
  audit.module(b,f'global_block.L{i}','common_residual');audit.module(b.mlp,f'global_ffn.L{i}');audit.module(b.norm1,f'global_norm1.L{i}');audit.module(b.norm2,f'global_norm2.L{i}')
  wrapper=b.attn;audit.module(wrapper,f'global_wrapper.L{i}','container',i,global_wrapper=True)
  o=wrapper.original if method=='ours256' else wrapper.orig_module
  for name in ['qkv','q_norm','k_norm','proj','proj_drop']:audit.module(getattr(o,name),f'global_{name}.L{i}')
  if o.rope is not None and id(o.rope) not in seen_rope:audit.module(o.rope,'global_rope',rope=True);seen_rope.add(id(o.rope))
 for name in ['camera_head','depth_head','point_head']:
  if getattr(model,name,None) is not None:audit.module(getattr(model,name),name)
 if method=='ours256':
  import saf3r.patch.reuse_residual_exploration as implementation
  original=implementation.ReuseResidualExploration.forward;source=textwrap.dedent(inspect.getsource(original))
  needle='    q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()'
  assert needle in source
  source=source.replace(needle,"    with audit.scope('ours.qkv_prepare_total','pipeline'):\n        q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()\n    audit.note(f'ours.prepared_qkv.L{self.layer}',meta((q,k,v)))")
  source=source.replace('        output[:,first:end]=result',"        with audit.scope('ours.window_output_copy','mechanics'):\n            output[:,first:end]=result")
  first=source.index('    if c.fuse_mask_ops and self.layer:\n');last=source.index('    if order is not None:\n',first);source=source[:first]+"    with audit.scope('ours.post_attention_checks_stats','bookkeeping'):\n"+textwrap.indent(source[first:last],'    ')+source[last:]
  needle='    return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))';assert needle in source
  source=source.replace(needle,"    with audit.scope('ours.output_layout','mechanics'):\n        merged=output.transpose(0,1).reshape(1,s.n,-1)\n    return o.proj_drop(o.proj(merged))")
  namespace=dict(original.__globals__,audit=audit,meta=meta)
  def stage(s,name):return audit.scope('ours.'+name,'attention_pipeline' if name=='execution_and_evidence' else 'algorithm')
  namespace['stage']=stage;exec(compile(source,'<diagnostic_ours_forward>','exec'),namespace);save_source=source
  for b in model.aggregator.global_blocks:b.attn.forward=types.MethodType(namespace['forward'],b.attn)
  state.direct_scope=lambda name:audit.scope('ours.'+name,'attention' if name=='sampled_direct_av' else 'attention_information')
  from saf3r.patch import reuse_sampled_history, reuse_window_graph
  old=reuse_sampled_history.build_prior
  def prior(*args,**kwargs):
   with audit.scope('ours.history_prior','algorithm'):return old(*args,**kwargs)
  reuse_sampled_history.build_prior=prior
  old_select=reuse_window_graph.select
  def select(*args,**kwargs):
   with audit.scope('ours.state_loss_budget_pack_transfer','algorithm'):return old_select(*args,**kwargs)
  reuse_window_graph.select=select
 else:
  import torch.nn.functional as F
  def forward(self,x,pos=None):
   with audit.scope('saf3r.qkv_prepare_total','pipeline'):
    B,N,C=x.shape;qkv=self.orig_module.qkv(x).reshape(B,N,3,self.orig_module.num_heads,self.orig_module.head_dim).permute(2,0,3,1,4);q,k,v=qkv.unbind(0)
    q,k=self.orig_module.q_norm(q),self.orig_module.k_norm(k)
    if self.orig_module.rope is not None:q=self.orig_module.rope(q,pos);k=self.orig_module.rope(k,pos)
   audit.note(f'saf3r.prepared_qkv.L{audit.layer}',meta((q,k,v)))
   x=self.compute_multihead_attention(q,k,v)
   with audit.scope('saf3r.output_layout','mechanics'):x=x.transpose(1,2).reshape(B,N,C)
   x=self.orig_module.proj(x);x=self.orig_module.proj_drop(x);return x
  def compute(self,q,k,v):
   outputs=[]
   for h in range(self.num_heads):
    qh,kh,vh=q[:,h:h+1],k[:,h:h+1],v[:,h:h+1]
    with audit.scope('saf3r.attention_and_routing','attention_pipeline'):
     result=F.scaled_dot_product_attention(qh,kh,vh) if self.head_cfg[h]['mode']=='full' else self.sparse_attention(qh,kh,vh,self.head_cfg[h],head_idx=h)
    outputs.append(result)
   with audit.scope('saf3r.output_head_concat','mechanics'):return torch.cat(outputs,dim=1)
  for b in model.aggregator.global_blocks:b.attn.forward=types.MethodType(forward,b.attn);b.attn.compute_multihead_attention=types.MethodType(compute,b.attn)
  save_source=inspect.getsource(forward)+'\n'+inspect.getsource(compute)
 return save_source


def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=['ours256','saf3r'],required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.ckpt_path=str(ROOT/'checkpoints/vggt/model_tracker_fixed_e20.pt');cfg.sparse_config_path=str(ROOT/cfg.sparse_config_path)
 state=None
 if a.method=='ours256':
  cfg.pop('patch_module');cfg.pop('sparse_config_path');model=build_model(cfg,'cuda')
  from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
  state=install_reuse_residual_exploration(model,ReuseExploreConfig(**json.loads((ROOT/'configs/evaluation/reuse_sampled_history_256.json').read_text())))
 else:model=build_model(cfg,'cuda')
 inp=json.loads((ROOT/'outputs/inference_efficiency_20261103/v8/quality/7scenes/heads/efficiency/input.json').read_text());images=load_images(model,'vggt',inp['image_paths']).cuda();audit=Recorder();topk_calls={}
 if a.method=='saf3r':
  modules=[]
  import saf3r.patch.utils as utils
  modules.append((utils,'compute_qk_topk_indices_batched'))
  try:
   import saf3r.triton.fused_qk_topk as fused
   modules.append((fused,'compute_qk_topk_indices_fused'))
  except (AttributeError,ImportError):pass
  for module,name in modules:
   original=getattr(module,name)
   def count(*args,_fn=original,_name=name,**kwargs):
    topk_calls[_name]=topk_calls.get(_name,0)+1
    with audit.scope('saf3r.mask_topk','algorithm'):return _fn(*args,**kwargs)
   setattr(module,name,count)
 def reset():
  if a.method=='saf3r':
   from saf3r.patch import update_args_per_forward
   update_args_per_forward(model,'vggt',*images.shape[-2:])
 samples=[]
 for i in range(4):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();before=dict(topk_calls);reset()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize();entry=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,topk_calls={k:v-before.get(k,0) for k,v in topk_calls.items()})
  assert torch.isfinite(raw['pose_enc']).all() and torch.isfinite(raw['depth']).all();del raw
  if i:samples.append(entry)
  else:cold=entry
  save(a.output_dir/'progress.json',dict(iteration=i,samples=samples,cold=cold));print(a.method,i,entry,flush=True)
 cached=None
 if a.method=='saf3r':
  before=dict(topk_calls);torch.cuda.synchronize();begin=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize();cached=dict(seconds=time.perf_counter()-begin,topk_calls={k:v-before.get(k,0) for k,v in topk_calls.items()},scope='ONE diagnostic forward without resetting input-dependent TopK,not an independent-inference benchmark');del raw;print('cached_route',cached,flush=True)
 diagnostic_source=install_audit(model,state,a.method,audit);(a.output_dir/'instrumented_forward.txt').write_text(diagnostic_source)
 old_empty=torch.cuda.empty_cache
 def empty():
  if not audit.enabled:return old_empty()
  before=torch.cuda.memory_reserved();t=time.perf_counter()
  with audit.scope('shared.empty_cache','infrastructure'):old_empty()
  audit.notes.setdefault('empty_cache_calls',[]).append(dict(cpu_seconds=time.perf_counter()-t,reserved_before=before,reserved_after=torch.cuda.memory_reserved()))
 torch.cuda.empty_cache=empty
 for attr in ['__bool__','__int__','item']:
  original=getattr(torch.Tensor,attr)
  def sync(t,*args,_fn=original,_attr=attr,**kwargs):
   if not audit.enabled or not t.is_cuda:return _fn(t,*args,**kwargs)
   with audit.scope('host_sync.'+_attr,'sync'):return _fn(t,*args,**kwargs)
  setattr(torch.Tensor,attr,sync)
 torch.cuda.synchronize();audit.enabled=True;before=dict(topk_calls);begin=time.perf_counter()
 with audit.scope('model','root'):
  reset()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();wall=time.perf_counter()-begin;audit.enabled=False;del raw
 prof=audit.finish(wall)
 result=dict(complete=True,method=a.method,input=inp,model_flags=dict(save_intermediates=model.aggregator.save_intermediates,point_head=model.point_head is not None,depth_head=model.depth_head is not None,camera_head=model.camera_head is not None),timing=dict(median_seconds=statistics.median(x['seconds'] for x in samples),samples=samples,cold=cold),cached_routing_diagnostic=cached,profile=prof,notes=audit.notes,profile_topk_calls={k:v-before.get(k,0) for k,v in topk_calls.items()},hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda),quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Same-boundary runtime/code audit only;no new pose or reconstruction evaluation,existing complete quality results reused',dense_speedup=None,dense_reason='No Dense rerun requested;paired comparison is ours256 versus SAF3R,not Dense')
 save(a.output_dir/'result.json',result)
 from scripts.quality_metrics import quality_record,quality_markdown
 (a.output_dir/'quality.md').write_text(quality_markdown(quality_record({},result['quality_reason'])))
 print('COMPLETE',a.method,result['timing']['median_seconds'],prof['wall_seconds'],prof['common_module_inclusive_seconds'],flush=True)
if __name__=='__main__':main()
