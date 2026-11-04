"""Matched full-forward timings and disjoint common/additional-operation profile."""
import argparse,contextlib,json,statistics,sys,time
from pathlib import Path
import torch,yaml
import torch.nn.functional as F
from addict import Dict

p=argparse.ArgumentParser();p.add_argument('--method',choices=['dense','v8','p0','beta'],required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
a.output_dir.mkdir(parents=True,exist_ok=False);root=a.source;sys.path[:0]=[str(root),str(root/'scripts'),str(root/'evaluation')]
from saf3r.utils.model_utils import build_model
from saf3r.utils.data_utils import load_images
cfg=Dict(yaml.safe_load((root/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=str(root/'checkpoints/vggt/model_tracker_fixed_e20.pt')
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=build_model(cfg,'cuda');state=None
if a.method!='dense':
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 name={'v8':'reuse_efficiency_v8','p0':'reuse_v8_p0','beta':'reuse_beta_gridmass'}[a.method]
 config=json.loads((root/f'configs/evaluation/{name}.json').read_text());state=install_reuse_residual_exploration(model,ReuseExploreConfig(**config))
else:config=None
images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();samples=[]
for i in range(4):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 torch.cuda.synchronize();value=dict(seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
 assert torch.isfinite(raw['depth']).all() and torch.isfinite(raw['pose_enc']).all();del raw
 if i:samples.append(value)
 else:cold=value
 (a.output_dir/'progress.json').write_text(json.dumps(dict(iteration=i,samples=samples,cold=cold),indent=2));print(i,value,flush=True)
rows=[];stack=[]
@contextlib.contextmanager
def scope(name):
 st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);row=dict(name=name,id=len(rows),parent=stack[-1]['id'] if stack else None,start=st,end=en);rows.append(row);stack.append(row);st.record()
 try:yield
 finally:en.record();stack.pop()
def wrap(obj,name,label):
 original=getattr(obj,name)
 def fn(*args,**kwargs):
  with scope(label):return original(*args,**kwargs)
 setattr(obj,name,fn)
def wrapmod(mod,label):wrap(mod,'forward',label)
wrapmod(model.aggregator.patch_embed,'image_backbone')
for block in model.aggregator.frame_blocks:wrapmod(block.attn,'frame_attention')
for block in [*model.aggregator.frame_blocks,*model.aggregator.global_blocks]:wrapmod(block.mlp,'ffn');wrapmod(block.norm1,'block_norm');wrapmod(block.norm2,'block_norm')
for name in ['camera_head','depth_head','point_head','track_head']:
 if getattr(model,name,None) is not None:wrapmod(getattr(model,name),name)
for layer,block in enumerate(model.aggregator.global_blocks):
 wrapmod(block.attn,'global_attention')
 orig=block.attn if a.method=='dense' else block.attn.original
 wrapmod(orig.qkv,'global_qkv');wrapmod(orig.q_norm,'global_qk_norm');wrapmod(orig.k_norm,'global_qk_norm');wrapmod(orig.proj,'global_output_projection')
 if orig.rope is not None:wrapmod(orig.rope,'global_rope')
orig_sdpa=F.scaled_dot_product_attention
def sdpa(*args,**kwargs):
 name='dense_global_core' if any(x['name']=='global_attention' for x in stack) else 'other_sdpa_core'
 with scope(name):return orig_sdpa(*args,**kwargs)
F.scaled_dot_product_attention=sdpa
if state:
 import saf3r.patch.reuse_residual_exploration as impl
 impl.stage=lambda state,name:scope('sparse_stage_'+name)
 import importlib
 backend=importlib.import_module('saf3r.triton.'+('reuse_serial_sparseprob' if a.method=='beta' else 'reuse_serial_tuned'))
 wrap(backend,'weighted_attention','sparse_execution_wrapper')
 class Launch:
  def __init__(self,k,label):self.k=k;self.label=label
  def __getitem__(self,grid):
   fn=self.k[grid]
   def run(*args,**kwargs):
    with scope(self.label):return fn(*args,**kwargs)
   return run
 for name,label in [('_attention','sparse_core_fused_evidence'),('_reduce','sparse_grid_reduce')]:
  if hasattr(backend,name):setattr(backend,name,Launch(getattr(backend,name),label))
torch.cuda.synchronize();begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
torch.cuda.synchronize();wall=time.perf_counter()-begin
items=[];groups={}
for row in rows:
 inclusive=row['start'].elapsed_time(row['end'])/1000
 exclusive=inclusive-sum(c['start'].elapsed_time(c['end'])/1000 for c in rows if c['parent']==row['id'])
 items.append(dict(id=row['id'],parent=row['parent'],operation=row['name'],inclusive_seconds=inclusive,exclusive_seconds=exclusive))
 entry=groups.setdefault(row['name'],dict(inclusive_seconds=0.,exclusive_seconds=0.,calls=0));entry['inclusive_seconds']+=inclusive;entry['exclusive_seconds']+=exclusive;entry['calls']+=1
outside=wall-sum(v['exclusive_seconds'] for v in groups.values())
result=dict(method=a.method,source=str(root),config=config,input=json.loads(a.input.read_text()),hardware=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__),complete=True,median_seconds=statistics.median(x['seconds'] for x in samples),samples=samples,cold=cold,profile=dict(wall_seconds=wall,groups=groups,outside_instrumented_seconds=outside,events=items),quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Matched timing/profiling diagnostic; complete geometry results are separate',dense_speedup=None,dense_reason='Computed after all paired runs finish')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(method=a.method,median_seconds=result['median_seconds'],groups=groups,outside_seconds=outside),indent=2),flush=True)
