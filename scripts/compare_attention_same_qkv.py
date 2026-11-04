"""First execution window of every layer; same Q/K/V Dense SDPA vs sparse executor."""
import argparse,json,statistics,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();root=a.source;sys.path[:0]=[str(root),str(root/'scripts'),str(root/'evaluation')];a.output_dir.mkdir(exist_ok=False,parents=True)
from benchmark_beta_gridmass import build
from saf3r.utils.data_utils import load_images
from saf3r.triton import reuse_serial_sparseprob as backend
model,state=build(root/'configs/evaluation/reuse_beta_gridmass.json');torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();rows=[];seen=set();original=backend.weighted_attention;core=backend._attention;events=[]
class Core:
 def __getitem__(self,grid):
  fn=core[grid]
  def call(*args,**kwargs):
   st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();r=fn(*args,**kwargs);en.record();events.append((st,en));return r
  return call
backend._attention=Core()
def timed(fn):
 st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();out=fn();en.record();en.synchronize();return st.elapsed_time(en)/1000

def execute(q,k,v,indices,bias,specials,scale,key_grid,regions,**kwargs):
 layer=state.previous_layer+1
 if layer not in seen:
  seen.add(layer);dense=lambda:F.scaled_dot_product_attention(q[None],k[None],v[None],scale=scale)
  sparse=lambda:original(q,k,v,indices,bias,specials,scale,key_grid,regions,**kwargs)
  dense();sparse();torch.cuda.synchronize();ds=[];ss=[];cs=[]
  for i in range(3):
   ds.append(timed(dense));events.clear();ss.append(timed(sparse));cs.append(sum(x.elapsed_time(y)/1000 for x,y in events))
  rows.append(dict(layer=layer,query_rows=q.shape[1],key_rows=k.shape[1],heads=q.shape[0],retained_patch_keys_mean=float((indices>=0).sum(1).float().mean()),dense_core_seconds=statistics.median(ds),sparse_core_fused_evidence_seconds=statistics.median(cs),sparse_executor_seconds=statistics.median(ss),dense_over_sparse_core=statistics.median(ds)/statistics.median(cs),dense_over_sparse_executor=statistics.median(ds)/statistics.median(ss)))
  (a.output_dir/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
 return original(q,k,v,indices,bias,specials,scale,key_grid,regions,**kwargs)
backend.weighted_attention=execute
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
torch.cuda.synchronize();result=dict(complete=True,sampling='First64-group window (2048queries) in every global layer;beta trajectory,same actual Q/K/V,1warmup+3samples/operator;not dataset/full-forward speedup',rows=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Same-QKV operator timing diagnostic; no geometry evaluation',dense_speedup=None,dense_reason='Operator-only speed ratios; full-forward paired timings separate',peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
