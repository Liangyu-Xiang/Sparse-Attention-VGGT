"""All24 first windows,identical-input original AV/history and actual FP32 queries."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'evaluation')]
import torch
from benchmark_direct_sparse_forward import build
from saf3r.utils.data_utils import load_images
from saf3r.patch import reuse_direct_history as history,reuse_ready_history as cache
from saf3r.triton import reuse_direct_sparse_attention as direct,reuse_serial_tuned as old,reuse_grid_local_history as local
from saf3r.triton.reuse_ready_history import decode
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(exist_ok=False,parents=True)
torch.manual_seed(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model,state=build(ROOT/'configs/evaluation/reuse_v8_direct_sparse.json');images=load_images(model,'vggt',json.loads(a.input.read_text())['image_paths']).cuda();original=direct.direct_sparse_attention;observe=history.observe_execution;save=history.save_evidence;seen=set();current={};rows=[]
def execute(*args,**kwargs):
 out=original(*args,**kwargs);current['out']=out[0];return out
def observe_execution(s,q,k,start,ids,actual,bias,obs,scale,evidence):
 summary=observe(s,q,k,start,ids,actual,bias,obs,scale,evidence)
 if start==0:
  layer=s.previous_layer+1;qw=q[:,:min(s.n,len(ids)*32)];layout={};ref,rs,raw=old.weighted_attention(qw,k,current['v'],actual,bias,s.specials,scale,s.grid[ids.long().clamp_min(0)],s.regions,evidence_layout=layout,emit_evidence=layer<23)
  rel=float((current['out'].float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-30));se=float((summary-rs).abs().max());assert rel<.01 and se<3e-6,(layer,rel,se)
  first_ids=actual[0,actual[0]>=0].long();keys=torch.cat([s.specials,first_ids]);b=torch.cat([torch.zeros(len(s.specials),device=q.device),bias[0,actual[0]>=0]])
  with torch.autocast('cuda',enabled=False):
   pq=qw[:,:min(32,qw.shape[1])].float();z=pq@k[:,keys].float().transpose(-1,-2)*scale+b;fp32=z.softmax(-1)@current['v'][:,keys].float()
   errors=(current['out'][:,:len(fp32[0])].float()-fp32).norm(dim=-1)/fp32.norm(dim=-1).clamp_min(1e-30)
  maximum=float(errors.max());assert maximum<.01,(layer,maximum)
  row=dict(layer=layer,groups=len(ids),av_relative_l2_vs_old=rel,fp32_max_head_query_relative_l2=maximum,fp32_mean_head_query_relative_l2=float(errors.mean()),legacy_grid_summary_max_abs=se)
  if raw is not None:
   den,_=local.normalizers(raw,obs['grid_logits'],layout['selected_grid_order'],layout['selected_grid_offsets'],s.grid_sizes,int(s.grid_sizes.max()));de=float((den-obs['direct_den']).abs().max());assert de<1e-5,(layer,de);row['legacy_grid_den_max_abs']=de;current['old_den']=den;current['old_raw']=raw
  rows.append(row);(a.output_dir/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
 return summary
def save_evidence(s,q,k,start,ids,actual,obs,scale):
 save(s,q,k,start,ids,actual,obs,scale)
 if start==0:
  raw=current.pop('old_raw');den=current.pop('old_den');logz=local.compatibility(den,obs['special_logits']);refcache=torch.empty_like(cache.peek_cache(s,0,len(ids)));uniform=torch.empty_like(s.ready_uniform[:len(ids)])
  local.store(raw,den,logz,ids,s.grid,obs['grid_logits'],refcache,uniform,1e-30)
  a=decode(refcache,s.budget);b=decode(cache.peek_cache(s,0,len(ids)),s.budget);err=float((a-b).abs().max());assert torch.allclose(a,b,atol=2e-7,rtol=.002),(s.previous_layer+1,err)
  rows[-1]['legacy_conditional_cache_max_abs']=err
# Keep actual V in the AV hook; no alternate output feeds the network.
def av(q,k,v,*args,**kwargs):current['v']=v;return execute(q,k,v,*args,**kwargs)
direct.direct_sparse_attention=av;history.observe_execution=observe_execution;history.save_evidence=save_evidence;begin=time.perf_counter()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
torch.cuda.synchronize();assert len(rows)==24 and len(state.records)==24 and torch.isfinite(raw['pose_enc']).all() and torch.isfinite(raw['depth']).all()
result=dict(complete=True,checks=rows,sampling='heads128,first64-group window in all24 layers;FP32 reference for every Head/query of first32-query group/layer;old AV and legacy Grid/cache compared on whole first window',diagnostic_wall_seconds=time.perf_counter()-begin,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Same-input AV/history/FP32 diagnostic with extra reference work;geometry evaluation separate',dense_speedup=None,dense_reason='Diagnostic wall includes oracle computations;not a benchmark')
(a.output_dir/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
