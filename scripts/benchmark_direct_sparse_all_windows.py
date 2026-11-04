"""Matched actual Q/K/V and support over all windows; old v8 trajectory unchanged."""
import argparse,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from benchmark_v8_p0 import build
from saf3r.utils.data_utils import load_images
from saf3r.triton import reuse_serial_tuned as old
from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,state=build(ROOT/'configs/evaluation/reuse_efficiency_v8.json');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
torch.cuda.synchronize();original=old.weighted_attention;rows=[];warmed=set();layer_windows={};configs=[(64,4,1),(128,4,2),(256,4,2),(128,8,2),(256,8,2)];tuning=[]
def timed(fn,repeats=3):
 values=[];out=None
 for i in range(repeats):
  st=torch.cuda.Event(enable_timing=True);en=torch.cuda.Event(enable_timing=True);st.record();out=fn();en.record();en.synchronize();values.append(st.elapsed_time(en)/1000)
 return statistics.median(values),out

def run(q,k,v,ids,bias,specials,scale,key_grid,regions,**kwargs):
 layer=state.previous_layer+1;window=layer_windows.get(layer,0);layer_windows[layer]=window+1;shape=(q.shape,ids.shape)
 result=original(q,k,v,ids,bias,specials,scale,key_grid,regions,**kwargs)
 positions=(torch.arange(len(ids),device=q.device)[:,None]*32+torch.arange(4,device=q.device)[None]*8+4).clamp_max(q.shape[1]-1).int()
 def direct(mode='compensated',export=None,config=(128,4,2)):
  return direct_sparse_attention(q,k,v,ids,specials,scale,bias if mode=='compensated' else None,positions,export is not None,history_export=export or 'lse_probe',block_keys=config[0],num_warps=config[1],num_stages=config[2])
 def dense():
  with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):return F.scaled_dot_product_attention(q[None],k[None],v[None],scale=scale)
 fns={'dense_flash':dense,'grid_compensated':lambda:original(q,k,v,ids,bias,specials,scale,key_grid,regions,**kwargs),'direct_pure':lambda:direct('pure'),'direct_compensated':lambda:direct(),'direct_history_A':lambda:direct(export='inline'),'direct_history_B':lambda:direct(export='lse_probe')}
 for name,fn in fns.items():
  key=(name,str(shape))
  if key not in warmed:fn();torch.cuda.synchronize();warmed.add(key)
 row=dict(layer=layer,window=window,groups=len(ids),query_rows=q.shape[1],key_rows=k.shape[1],heads=q.shape[0],retained_mean=float((ids>=0).sum(1).float().mean()))
 outputs={}
 for name,fn in fns.items():
  seconds,out=timed(fn);row[name+'_seconds']=seconds
  if name.startswith('direct'):outputs[name]=out
  else:del out
 out=outputs['direct_compensated'][0];rel=float((out.float()-result[0].float()).norm()/result[0].float().norm().clamp_min(1e-30));assert rel<.01,(layer,window,rel);row['output_relative_l2']=rel
 ea=outputs['direct_history_A'][1];eb=outputs['direct_history_B'][1];pa=torch.cat([ea['special_probability'],ea['patch_probability']],-1);pb=torch.cat([eb['special_probability'],eb['patch_probability']],-1);pe=float((pa-pb).abs().max());assert pe<2e-4,(layer,window,pe)
 row.update(probability_A_B_max_abs=pe,A_normalization_max_abs=float((pa.sum(-1)-1).abs().max()),B_normalization_max_abs=float((pb.sum(-1)-1).abs().max()),A_scratch_bytes=ea['scratch_bytes'],B_scratch_bytes=eb['scratch_bytes'])
 if window==0 and layer in [0,8,16,23]:
  for config in configs:
   direct(config=config);seconds,_=timed(lambda:direct(config=config));tuning.append(dict(layer=layer,block_keys=config[0],warps=config[1],stages=config[2],seconds=seconds))
 rows.append(row);(a.output_dir/'progress.json').write_text(json.dumps(dict(windows=len(rows),last=row),indent=2)+'\n');return result
old.weighted_attention=run;torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):model(images)
torch.cuda.synchronize();layers=[]
for layer in range(24):
 group=[x for x in rows if x['layer']==layer];layers.append(dict(layer=layer,windows=len(group),query_rows=sum(x['query_rows'] for x in group),**{name+'_seconds':sum(x[name+'_seconds'] for x in group) for name in ['dense_flash','grid_compensated','direct_pure','direct_compensated','direct_history_A','direct_history_B']}))
result=dict(complete=True,sampling='heads128,original v8 trajectory,EVERY window/all24 global layers;one warmup model forward;three samples per operator/window;fixed128/4/2 configuration for complete-window results;tuning firstwindow layers0/8/16/23 separate',input=json.loads(a.input.read_text()),windows=rows,layers=layers,tuning=tuning,totals={name+'_seconds':sum(x[name+'_seconds'] for x in rows) for name in ['dense_flash','grid_compensated','direct_pure','direct_compensated','direct_history_A','direct_history_B']},diagnostic_wall_seconds=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Matched operator/tuning diagnostic with extra executions;no GT geometry evaluation',dense_speedup=None,dense_reason='Operator-only timings;not end-to-end VGGT speedup')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,windows=len(rows),totals=result['totals']),indent=2),flush=True)
