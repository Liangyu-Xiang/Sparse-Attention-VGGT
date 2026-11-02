"""One changed-method diagnostic with exclusive, non-overlapping GPU scopes."""
import argparse,contextlib,hashlib,json,sys,time,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
FORK=ROOT/'outputs/no_mass_compensation_quality_v2_20261102/source';sys.path[:0]=[str(FORK),str(FORK/'evaluation')]
import torch,yaml
from addict import Dict
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
import saf3r.patch.reuse_residual_exploration as impl
from scripts.evaluate_attention_first_residual_value import save,write_quality
from scripts.quality_metrics import quality_record
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
  original=impl.ReuseResidualExploration.forward;rec=self
  def forward(att,*args,**kw):
   with rec.scope('global_attention',att.layer):return original(att,*args,**kw)
  impl.ReuseResidualExploration.forward=forward
  import saf3r.patch.adaptive_value_clustering as cluster
  for name in ['centers_from_labels','center_graph','local_assignment','full_assignment','repair_empty']:self.wrap(cluster,name,'cluster_'+name)
  import saf3r.patch.reuse_headmean_history as history
  self.wrap(history,'build_prior','consume_history');self.wrap(history,'save_evidence','produce_history')
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
   result.append(dict(id=r['id'],parent=r['parent'],name=r['name'],layer=r['layer'],inclusive_seconds=inclusive,exclusive_seconds=exclusive))
  grouped={}
  for r in result:
   q=grouped.setdefault(r['name'],dict(calls=0,inclusive_seconds=0.,exclusive_seconds=0.))
   q['calls']+=1;q['inclusive_seconds']+=r['inclusive_seconds'];q['exclusive_seconds']+=r['exclusive_seconds']
  roots=[r for r in result if r['parent'] is None]
  total=sum(r['inclusive_seconds'] for r in roots);unique=sum(r['exclusive_seconds'] for r in result)
  assert len(roots)==24 and abs(total-unique)<1e-7
  return result,grouped,total,unique

def main():
 out=a.output_dir
 if out.exists():raise ValueError('Fresh diagnostic output required')
 out.mkdir();write_quality(out,quality_record({},'Exclusive timing diagnostic only; no GT quality run'))
 save(out/'status.json',dict(stage='loading'))
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((FORK/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(FORK/'checkpoints/vggt/model_tracker_fixed_e20.pt')
 inputs=json.loads((ROOT/'outputs/ready_history_v5_heads128_20261102/after/input.json').read_text()) if (ROOT/'outputs/ready_history_v5_heads128_20261102/after/input.json').exists() else json.loads((ROOT/'outputs/ready_history_quality_20261102/7scenes/heads/ready_history_v5/input.json').read_text())
 model=build_model(cfg,'cuda');images=load_images(model,'vggt',inputs['image_paths']).cuda()
 state=impl.install_reuse_residual_exploration(model,impl.ReuseExploreConfig(**json.loads((FORK/'configs/evaluation/reuse_zdrop_history_states.json').read_text())))
 save(out/'status.json',dict(stage='forward'))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 del raw;torch.cuda.synchronize();rec=Recorder(state);rec.install();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();wall=time.perf_counter()-begin
 assert torch.isfinite(raw['pose_enc']).all() and torch.isfinite(raw['depth']).all()
 rows,groups,total,unique=rec.finish()
 save(out/'profile.json',dict(wall_seconds=wall,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,scope='All24 global Attention modules; fused core includes probes and grid sums. QKV/RoPE/projection included only in root residual.',exclusive_sum_seconds=unique,global_attention_seconds=total,duplicate_seconds=0.,sum_error_seconds=unique-total,groups=groups,events=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},reason='Exclusive timing diagnostic; no GT evaluation',dense_speedup=None,dense_reason='Unchanged controls not rerun'))
 with (out/'scopes.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(out/'status.json',dict(stage='diagnostic_complete'));print(json.dumps(dict(wall_seconds=wall,groups=groups),indent=2))
if __name__=='__main__':main()
