import contextlib
import torch
import saf3r.patch.reuse_residual_exploration as impl
class Recorder:
 def __init__(self,state):self.state=state;self.rows=[];self.stack=[];self.ident=0
 @contextlib.contextmanager
 def scope(self,name,layer=None):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record()
  row=dict(id=self.ident,parent=self.stack[-1]['id'] if self.stack else None,name=name,layer=self.state.previous_layer+1 if layer is None else layer,start=st,end=en)
  self.ident+=1;self.stack.append(row)
  try:yield
  finally:en.record();self.stack.pop();self.rows.append(row)
 def wrap(self,module,name,label):
  original=getattr(module,name)
  def wrapped(*args,**kwargs):
   with self.scope(label):return original(*args,**kwargs)
  setattr(module,name,wrapped)
 def install(self):
  impl.stage=lambda state,name:self.scope(name)
  self.state.grid_local_scope=self.scope
  self.wrap(impl,'prepare_qkv','qkv_rope_prepare')
  original=impl.ReuseResidualExploration.forward;rec=self
  def forward(att,*args,**kw):
   with rec.scope('global_attention',att.layer):return original(att,*args,**kw)
  impl.ReuseResidualExploration.forward=forward
  import saf3r.patch.adaptive_value_clustering as cluster
  for name in ['centers_from_labels','center_graph','local_assignment','full_assignment','repair_empty']:self.wrap(cluster,name,'cluster_'+name)
  import saf3r.patch.reuse_headmean_history as history
  self.wrap(history,'build_prior','consume_history');self.wrap(history,'save_evidence','produce_history')
  import saf3r.patch.reuse_grid_local_history as local
  self.wrap(local,'save_evidence','produce_history')
  rule=impl.history_rule
  for name in ['gate','states','pack','transfer','diagnostics']:self.wrap(rule,name,'state_'+name)
  import saf3r.triton.reuse_single_qk_grid as backend
  self.wrap(backend,'weighted_attention','attention_wrapper')
  class Launch:
   def __init__(self,kernel,label):self.kernel=kernel;self.label=label
   def __getitem__(self,grid):
    call=self.kernel[grid]
    def run(*args,**kwargs):
     with rec.scope(self.label):return call(*args,**kwargs)
    return run
  backend._attention=Launch(backend._attention,'attention_core_fused_grid')
  backend._reduce=Launch(backend._reduce,'attention_grid_normalize_reduce')
  import saf3r.triton.reuse_ready_history as evidence
  import saf3r.triton.reuse_headmean_history as mean_kernel
  evidence._row_lse=Launch(evidence._row_lse,'kernel_step1_selected_logsumexp')
  evidence._calibrate=Launch(evidence._calibrate,'kernel_step2_grid_exp_segment_sum')
  mean_kernel._store_mean=Launch(mean_kernel._store_mean,'kernel_step34_headmean_fp16_u14_store')
  import saf3r.triton.reuse_grid_local_history as local_kernel
  for name in ['_normalizers','_compat','_uniform','_store_local']:setattr(local_kernel,name,Launch(getattr(local_kernel,name),'kernel_local'+name))
 def finish(self):
  torch.cuda.synchronize()
  by={r['id']:r for r in self.rows};children={i:[] for i in by}
  for r in self.rows:
   if r['parent'] is not None:children[r['parent']].append(r)
  origin=min(self.rows,key=lambda r:r['id'])['start']
  for r in self.rows:r['begin_s']=origin.elapsed_time(r['start'])/1000;r['end_s']=origin.elapsed_time(r['end'])/1000
  result=[]
  for r in sorted(self.rows,key=lambda x:x['id']):
   sub=sorted(children[r['id']],key=lambda x:x['begin_s'])
   last=r['begin_s']
   for c in sub:
    assert c['begin_s']>=last-1e-7 and c['end_s']<=r['end_s']+1e-7
    last=c['end_s']
   inclusive=r['end_s']-r['begin_s'];exclusive=inclusive-sum(c['end_s']-c['begin_s'] for c in sub)
   result.append(dict(id=r['id'],parent=r['parent'],name=r['name'],layer=r['layer'],begin_seconds=r['begin_s'],end_seconds=r['end_s'],inclusive_seconds=inclusive,exclusive_seconds=exclusive))
  grouped={}
  for r in result:
   q=grouped.setdefault(r['name'],dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
   q['calls']+=1;q['inclusive_seconds']+=r['inclusive_seconds'];q['exclusive_seconds']+=r['exclusive_seconds']
  roots=[r for r in result if r['parent'] is None]
  total=sum(r['inclusive_seconds'] for r in roots);unique=sum(r['exclusive_seconds'] for r in result)
  assert len(roots)==24 and abs(total-unique)<1e-7
  return result,grouped,total,unique
