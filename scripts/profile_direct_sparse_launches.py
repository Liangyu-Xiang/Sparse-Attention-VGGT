"""Actual CUDA launch counts/compiler resources on real heads128 first window."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
import torch.nn.functional as F
from torch.profiler import profile,ProfilerActivity
from torch.nn.attention import sdpa_kernel,SDPBackend
from benchmark_v8_p0 import build
from saf3r.utils.data_utils import load_images
from saf3r.triton import reuse_serial_tuned as old,reuse_direct_sparse_attention as direct
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,state=build(ROOT/'configs/evaluation/reuse_efficiency_v8.json');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();original=old.weighted_attention;resources={};rows=[];done=False;mode=''
class Capture:
 def __init__(self,kernel,name):self.kernel=kernel;self.name=name
 def __getitem__(self,grid):
  launch=self.kernel[grid]
  def run(*args,**kwargs):
   kernel=launch(*args,**kwargs);resources[mode+':'+self.name]=dict(registers_per_thread=getattr(kernel,'n_regs',None),spills=getattr(kernel,'n_spills',None),shared_bytes=getattr(kernel.metadata,'shared',None),warps=kwargs.get('num_warps'),stages=kwargs.get('num_stages'));return kernel
  return run
for name in ['_attention','_probe','_inline_mean']:setattr(direct,name,Capture(getattr(direct,name),name))
def run(q,k,v,ids,bias,specials,scale,key_grid,regions,**kwargs):
 global done,mode
 result=original(q,k,v,ids,bias,specials,scale,key_grid,regions,**kwargs)
 if done:return result
 done=True;positions=(torch.arange(len(ids),device=q.device)[:,None]*32+torch.arange(4,device=q.device)[None]*8+4).clamp_max(q.shape[1]-1).int()
 def dense():
  with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):return F.scaled_dot_product_attention(q[None],k[None],v[None],scale=scale)
 methods={'dense_flash':dense,'grid_sparse':lambda:original(q,k,v,ids,bias,specials,scale,key_grid,regions,**kwargs),'direct_pure':lambda:direct.direct_sparse_attention(q,k,v,ids,specials,scale,None,positions),'direct_compensated':lambda:direct.direct_sparse_attention(q,k,v,ids,specials,scale,bias,positions),'direct_A':lambda:direct.direct_sparse_attention(q,k,v,ids,specials,scale,bias,positions,True,history_export='inline'),'direct_B':lambda:direct.direct_sparse_attention(q,k,v,ids,specials,scale,bias,positions,True,history_export='lse_probe')}
 for mode,fn in methods.items():
  fn();torch.cuda.synchronize()
  with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:fn();torch.cuda.synchronize()
  names=[e.name for e in prof.events() if str(e.device_type).endswith('CUDA')];counts={}
  for name in names:counts[name]=counts.get(name,0)+1
  rows.append(dict(method=mode,cuda_events=len(names),event_counts=counts))
 return result
old.weighted_attention=run
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
result=dict(complete=True,scope='Actual heads128 L0 first64-group window,all16heads,2048queries,exact same QKV/support/bias;CUDA event list distinguishes data kernels and memsets',launches=rows,compiler_resources=resources,occupancy=None,occupancy_reason='Separate Nsight Compute attempt ERR_NVGPUCTRPERM;no measured hardware occupancy',quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='CUDA launch/resource diagnostic;no geometry evaluation',dense_speedup=None,dense_reason='Not full-forward performance timing')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
