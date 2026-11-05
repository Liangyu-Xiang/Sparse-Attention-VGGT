"""Finite RGB-only search with local exact Attention teachers, never test GT."""
import argparse,gc,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def candidates(base):
 schedules={'middle':base['layer_retentions'],'flat':[.16]*24,'front':[.20]*6+[.16]*12+[.12]*6,'lowmiddle':[.08]*6+[.16]*12+[.08]*6,'lowflat':[.12]*24}
 rows=[dict(name='current',config=base)]
 for fraction in (.04,.06,.08,.10,.12,.16):
  for feature in ('head_mean','head_sketch'):
   for profile,rates in schedules.items():
    cfg=dict(base,cluster_fraction=fraction,shared_value_features=feature,layer_retentions=rates)
    if cfg==base:continue
    rows.append(dict(name=f'c{round(fraction*100):02}_{feature}_{profile}',config=cfg))
 return rows
class Fidelity:
 def __init__(self,model,state):self.model=model;self.s=state;self.rows=[];self.handles=[]
 def __enter__(self):
  import torch
  import saf3r.patch.reuse_residual_exploration as mod
  import saf3r.triton.reuse_direct_sampled_probability as direct
  self.orig_prepare=mod.prepare_qkv;self.new_backend=bool(getattr(self.s.config,'current_probability_probes',0))
  if self.new_backend:
   import saf3r.patch.current_probe_reuse as direct
   self.orig_execute=direct.main_attention
  else:self.orig_execute=direct.direct_sparse_attention_sampled
  def prepare(original,x,pos):
   q,k,v=self.orig_prepare(original,x,pos);self.q=q.to(v.dtype).contiguous();self.k=k.to(v.dtype).contiguous();self.v=v;self.module=original;self.starts=0;self.samples=[]
   # Four deterministic patch queries in each of eight evenly spaced groups.
   s=self.s;groups=torch.linspace(0,s.groups-1,8,device=q.device).long().unique()
   offset=s.config.query_group_size//16 if getattr(s,'_diagnostic_heldout',False) else 0
   ids=(groups[:,None]*s.config.query_group_size+torch.tensor([1,3,5,7],device=q.device)*(s.config.query_group_size//8)-offset).flatten().clamp_max(s.n-1)
   special=torch.zeros(s.n,device=q.device,dtype=torch.bool);special[s.specials]=True
   self.ids=ids[~special[ids]].unique()
   return q,k,v
  def execute(q,k,v,actual,specials,scale,bias,**kwargs):
   result,evidence=self.orig_execute(q,k,v,actual,specials,scale,bias,**kwargs)
   first=self.starts;end=first+q.shape[1];ids=self.ids[(self.ids>=first)&(self.ids<end)]
   if len(ids):
    group=(ids-first)//self.s.config.query_group_size
    support=torch.cat((actual[group].long(),specials[None].expand(len(ids),-1)),1)
    weights=torch.cat((bias[group].exp(),torch.ones((len(ids),len(specials)),device=q.device)),1)
    self.samples.append((ids,support,weights,result[:,ids-first].clone()))
   self.starts=end;return result,evidence
  mod.prepare_qkv=prepare
  if self.new_backend:direct.main_attention=execute
  else:direct.direct_sparse_attention_sampled=execute
  for layer,b in enumerate(self.model.aggregator.global_blocks):
   def hook(module,args,layer=layer):
    with torch.autocast('cuda',enabled=False):self.measure(layer)
   self.handles.append(b.attn.original.proj.register_forward_pre_hook(hook))
  return self
 def measure(self,layer):
  import torch
  q,k,v=self.q,self.k,self.v
  ids=torch.cat([r[0] for r in self.samples]);support=torch.cat([r[1] for r in self.samples]);weights=torch.cat([r[2] for r in self.samples]);sparse=torch.cat([r[3] for r in self.samples],1)
  logits=torch.bmm(q[:,ids].float(),k.float().transpose(1,2))*self.module.scale
  p=logits.softmax(-1);vf=v.float();target=torch.bmm(p,vf)
  def error(out):
   # Project head errors through the actual Wo, without adding its bias.
   x=(out-target).transpose(0,1).reshape(len(ids),-1)
   y=target.transpose(0,1).reshape(len(ids),-1)
   x=torch.nn.functional.linear(x.float(),self.module.proj.weight.float());y=torch.nn.functional.linear(y.float(),self.module.proj.weight.float())
   return float(x.square().sum()/y.square().sum().clamp_min(1e-12))
  safe=support.clamp_min(0);valid=support>=0
  selected=p.gather(2,safe[None].expand(len(q),-1,-1))*valid[None]
  keep=torch.zeros((len(ids),k.shape[1]),device=q.device).scatter_add_(1,safe,valid.float())>0
  bank=torch.zeros_like(keep,dtype=torch.float32).scatter_add_(1,safe,weights*valid)
  up=p*keep[None];unit=torch.bmm(up,vf)/up.sum(2).clamp_min(1e-20)[:,:,None]
  wp=p*bank[None];mass=wp.sum(2);moment=torch.bmm(wp,vf)
  reconstructed=moment/mass.clamp_min(1e-20)[:,:,None]
  # Oracle diagnostic: add 16 omitted high-probability keys; not a deployable
  # policy, not equal budget, never used inside the candidate's forward.
  rank=p.mean(0).masked_fill(keep,-1.)
  extra=rank.topk(16,1).indices;ep=p.gather(2,extra[None].expand(len(q),-1,-1));ev=vf[:,extra]
  added=(moment+(ep[:,:,:,None]*ev).sum(2))/(mass+ep.sum(2)).clamp_min(1e-20)[:,:,None]
  first_frame=(self.s.patches<self.s.height*self.s.width+5)
  row=dict(layer=layer,queries=len(ids),projected_relative_mse=error(sparse.float()),unit_weight_relative_mse=error(unit),weighted_reconstruction_relative_mse=error(reconstructed),add16_high_probability_relative_mse=error(added),selected_dense_probability=float(selected.sum(-1).mean()),first_frame_dense_mass=float(p[:,:,:self.s.height*self.s.width+5].sum(-1).mean()))
  self.rows.append(row);print('FIDELITY',layer,row['projected_relative_mse'],flush=True)
 def __exit__(self,*exc):
  import saf3r.patch.reuse_residual_exploration as mod
  import saf3r.triton.reuse_direct_sampled_probability as direct
  mod.prepare_qkv=self.orig_prepare
  if self.new_backend:
   import saf3r.patch.current_probe_reuse as backend
   backend.main_attention=self.orig_execute
  else:direct.direct_sparse_attention_sampled=self.orig_execute
  for h in self.handles:h.remove()
 def summary(self):
  assert len(self.rows)==24
  return dict(score=sum(r['projected_relative_mse'] for r in self.rows)/24,layers=self.rows,definition='Mean24-layer Wo-projected local exact Attention relative MSE on32 deterministic patch queries, sparse trajectory; not a full Dense model or geometric quality')
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=3);p.add_argument('--heldout-probes',action='store_true');a=p.parse_args()
 import numpy as np,torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record,quality_markdown
 plan=json.loads(a.plan.read_text());torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=plan['checkpoint']
 model=build_model(cfg,'cuda');model.eval();datasets={}
 for name,entry in plan['calibration'].items():datasets[name]=load_images(model,'vggt',entry['image_paths']).cuda()
 for index,row in enumerate(plan['candidates']):
  if row['name']!='current' and index%a.shards!=a.shard:continue
  for name,images in datasets.items():
   out=a.output/f'shard{a.shard}'/row['name']/name
   if (out/'result.json').exists():continue
   out.mkdir(parents=True,exist_ok=True)
   q=quality_record({},'RGB-only hyperparameter screening; no GT loaded; local exact sampled Attention diagnostic, not geometric quality evaluation')
   save(out/'quality.json',q);(out/'quality.md').write_text(quality_markdown(q))
   state=install_reuse_residual_exploration(model,ReuseExploreConfig(**row['config']));state._diagnostic_heldout=a.heldout_probes;samples=[];raw=None
   try:
    for rep in range(3):
     del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
     torch.cuda.synchronize();samples.append(dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30));print('TIMING',row['name'],name,rep,samples[-1],flush=True)
    del raw;raw=None
    with Fidelity(model,state) as fidelity,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
    torch.cuda.synchronize();scores=fidelity.summary();scores['heldout_queries']=a.heldout_probes;del fidelity
    save(out/'result.json',dict(name=row['name'],calibration=name,config=row['config'],quality=q,input=plan['calibration'][name],timing=dict(warmup=1,repeats=2,median_seconds=float(np.median([r['seconds'] for r in samples[1:]])),samples=samples[1:],cold_samples=samples[:1]),fidelity=scores,speedup_vs_dense=None,speedup_reason='No full Dense control rerun; diagnostic only',checkpoint_sha256=plan['checkpoint_sha256'],test_gt_read=False))
   except Exception as exc:
    import traceback
    reason=f'{type(exc).__name__}: {exc}'
    save(out/'failure.json',dict(complete=False,error=reason,traceback=traceback.format_exc(),quality=quality_record({},reason),speedup_vs_dense=None))
    raise
   finally:
    state.reset_handle.remove()
    for b in model.aggregator.global_blocks:b.attn=b.attn.original
    del model.reuse_explore_state
    del state,raw;gc.collect();torch.cuda.empty_cache()
   print('COMPLETE',row['name'],name,flush=True)
if __name__=='__main__':main()
