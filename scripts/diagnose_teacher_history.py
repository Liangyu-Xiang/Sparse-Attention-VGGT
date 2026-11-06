"""New method only;RGB timing,local Dense teachers and historical probability audit."""
import argparse,json,sys,time,statistics,ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import save,Fidelity

class Audit:
 def __init__(self,state):self.s=state;self.rows=[];self.generation=[];self.fidelity=[]
 def __enter__(self):
  import torch
  from saf3r.patch import reuse_residual_exploration as mod,reuse_window_graph as window
  if self.s.config.teacher_history_attention_backend=='probability_tiles':
   from saf3r.triton import teacher_history_probability_tiles as av
  else:
   from saf3r.triton import teacher_history_attention as av
  from saf3r.triton import teacher_history_probability as probability
  self.update=probability.update;self.prepare=mod.prepare_qkv;self.select=window.select;self.attention=av.attention;self.handles=[]
  def prepare(original,x,pos):
   q,k,v=self.prepare(original,x,pos);self.q=q.to(v.dtype);self.k=k.to(v.dtype);self.v=v;self.original=original;self.start=0;self.samples=[];s=self.s
   self.groups=torch.linspace(0,s.groups-1,4,device=q.device).long().unique();self.positions=(self.groups[:,None]*256+torch.tensor([32,96,160,224],device=q.device)).clamp_max(s.n-1)
   self.heldout=(self.positions-16).clamp_min(0)
   with torch.autocast('cuda',enabled=False):
    raw=self.q[:,self.positions.flatten()].float()@self.k.float().transpose(1,2)*original.scale;self.truth=raw.softmax(-1).mean(0).reshape(len(self.groups),4,s.n).mean(1)[:,s.patches]
    raw=self.q[:,self.heldout.flatten()].float()@self.k.float().transpose(1,2)*original.scale;self.dense=(raw.softmax(-1)@v.float()).reshape(len(q),len(self.groups),4,v.shape[-1])
   return q,k,v
  def metrics(a,b):return dict(relative_l1=float((a-b).abs().sum()/b.sum().clamp_min(1e-20)),relative_l2=float((a-b).square().sum().sqrt()/b.square().sum().sqrt().clamp_min(1e-20)),cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=1).mean()))
  def select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,**kwargs):
   result=self.select(s,layer,prior,pi,old_ids,allowed,z,reps,labels,population,geometry,**kwargs);start=self.start;g=len(prior);pick=(self.groups>=start)&(self.groups<start+g);groups=self.groups[pick];local=groups-start
   if len(groups):
    from saf3r.patch.current_probe_states import select as direct
    true=self.truth[pick];pred=prior[local];history=old_ids[local].long();valid=history>=0;safe=history.clamp_min(0)
    ref,_,w,_=direct(true,torch.empty(len(groups),len(reps),device=z.device),old_ids[local].contiguous(),allowed[local],z,reps,labels,population,geometry,s.budget,s.config.dual_iterations)
    n=pred.shape[1];ids=result[0][local];a=torch.zeros(len(groups),n+1,device=z.device,dtype=torch.bool).scatter_(1,ids.long().masked_fill(ids<0,n),ids>=0)[:,:n];b=torch.zeros(len(groups),n+1,device=z.device,dtype=torch.bool).scatter_(1,ref.long().masked_fill(ref<0,n),ref>=0)[:,:n]
    coarse=pred.reshape(len(groups),-1) # Metrics are on actual all-Key predicted probability.
    self.rows.append(dict(layer=layer,groups=groups.tolist(),all_patch_prediction=metrics(pred,true),history_prediction=metrics(pred.gather(1,safe)*valid,true.gather(1,safe)*valid),mask_jaccard=float(((a&b).sum(1)/(a|b).sum(1).clamp_min(1)).mean()),captured_mass_vs_reference=float(((true*a).sum(1)/(true*b).sum(1).clamp_min(1e-20)).mean())))
    self.reference=(pick,local,ref,w)
   else:self.reference=None
   self.start+=g;return result
  def attention(q,k,v,actual,specials,weights,scale,probes=4,capture=True,warps=4,stages=1):
   result,e=self.attention(q,k,v,actual,specials,weights,scale,probes,capture,warps,stages)
   if self.reference is not None:
    pick,local,ref,rw=self.reference;first=(self.start-len(actual))*256;positions=self.heldout[pick];patches=self.s.patches
    def manual(ids,ww):
     keys=torch.cat((specials[None].expand(len(ids),-1),patches[ids.clamp_min(0)].masked_fill(ids<0,-1)),1);weight=torch.cat((torch.ones(len(ids),len(specials),device=q.device),ww),1)
     with torch.autocast('cuda',enabled=False):
      raw=torch.einsum('hgpd,hgad->hgpa',self.q[:,positions].float(),k[:,keys.clamp_min(0)].float())*scale+weight.clamp_min(1e-30).log()[None,:,None];raw.masked_fill_((keys<0)[None,:,None],-torch.inf);return torch.einsum('hgpa,hgad->hgpd',raw.softmax(-1),v[:,keys.clamp_min(0)].float())
    target=manual(ref,rw);got=result[:,positions-first].float();dense=self.dense[:,pick];h,g,p,d=got.shape
    def err(a,b):
     with torch.autocast('cuda',enabled=False):
      aa=(a-b).permute(1,2,0,3).reshape(g*p,h*d);bb=b.permute(1,2,0,3).reshape(g*p,h*d);return float(torch.nn.functional.linear(aa,self.original.proj.weight.float()).square().sum()/torch.nn.functional.linear(bb,self.original.proj.weight.float()).square().sum().clamp_min(1e-20))
    self.fidelity.append(dict(layer=self.s.previous_layer+1,projected_error_vs_teacher_mask=err(got,target),projected_error_vs_dense=err(got,dense),teacher_mask_error_vs_dense=err(target,dense)))
   return result,e
  def update(prior,target_mass,ids,evidence,destination,specials,block,eta):
   self.update(prior,target_mass,ids,evidence,destination,specials,block,eta)
   start=self.start-len(ids);pick=(self.groups>=start)&(self.groups<self.start);local=self.groups[pick]-start
   if len(local):self.generation.append(dict(layer=self.s.previous_layer+1,all_patch_prediction=metrics(destination[local],self.truth[pick])))
  mod.prepare_qkv=prepare;window.select=select;av.attention=attention;probability.update=update
  return self
 def __exit__(self,*exc):
  from saf3r.patch import reuse_residual_exploration as mod,reuse_window_graph as window
  if self.s.config.teacher_history_attention_backend=='probability_tiles':
   from saf3r.triton import teacher_history_probability_tiles as av
  else:
   from saf3r.triton import teacher_history_attention as av
  from saf3r.triton import teacher_history_probability as probability
  mod.prepare_qkv=self.prepare;window.select=self.select;av.attention=self.attention;probability.update=self.update

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--timing-from',type=Path);p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/streamed_history_20261106.json');p.add_argument('--sequence',choices=['scannet','tum128'],default='scannet');a=p.parse_args()
 import numpy as np,torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record
 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['saf3r','evaluation'] for p in (ROOT/folder).rglob('*.py')};plan=json.loads((ROOT/'outputs/quality_speed_search_20261106/phase5/plan.json').read_text());entry=plan['calibration']['scannet'] if a.sequence=='scannet' else plan['validation']['tum_walking128'];torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=plan['checkpoint'];model=build_model(cfg,'cuda').eval();images=load_images(model,'vggt',entry['image_paths']).cuda();config=json.loads(a.config.read_text());s=install_reuse_residual_exploration(model,ReuseExploreConfig(**config));samples=[];raw=None
 if a.timing_from:
  samples=[ast.literal_eval(line.split(' ',2)[2]) for line in a.timing_from.read_text().splitlines() if line.startswith('TIMING ')]
  assert len(samples)==4
 for rep in ([] if a.timing_from else range(4)):
  del raw;raw=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  torch.cuda.synchronize();samples.append(dict(seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30));print('TIMING',rep,samples[-1],flush=True)
 del raw;raw=None
 with Audit(s) as audit,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
 del raw;raw=None;s._diagnostic_heldout=True
 # Dense reference is sampled locally by Audit;no full control rerun.
 q=quality_record({},'Independent RGB-only current sparse-trajectory QKV probability/attention diagnostic;no geometric truth or full Dense rerun')
 assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in source_sha256.items())
 save(a.output,dict(source_sha256=source_sha256,quality=q,sequence=a.sequence,input=entry,config=config,timing=dict(warmup=1,repeats=3,median_seconds=statistics.median(v['seconds'] for v in samples[1:]),samples=samples[1:],cold_samples=samples[:1]),probability_transfer=audit.rows,probability_generation=audit.generation,fidelity=audit.fidelity,speedup_vs_dense=None,speedup_reason='No full Dense rerun;new method only',test_gt_read=False,layers=[{k:(v.item() if isinstance(v,torch.Tensor) else v) for k,v in row.items()} for row in s.records]));print('COMPLETE',a.sequence,flush=True)
if __name__=='__main__':main()
