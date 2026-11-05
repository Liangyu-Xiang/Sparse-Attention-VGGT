"""Offline same-budget single-Key counterfactuals on independent RGB only.

Candidate pool is8 omitted high-probability keys per sampled query group, not
all-key global oracle. Exact sampled probabilities are unavailable at inference.
These constrained diagnostic swaps are never used to choose final test configs.
"""
import argparse,gc,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from scripts.search_quality_speed import Fidelity,save
class Impact(Fidelity):
 def __init__(self,*args):super().__init__(*args);self.keys=[];self.priors={};self.histories={};self.traces={}
 def __enter__(self):
  super().__enter__()
  import saf3r.patch.reuse_sampled_history as history
  self.prior_original=history.build_prior
  def prior(s,start,old,obs,return_uniform=False):
   r=self.prior_original(s,start,old,obs,return_uniform=return_uniform)
   for group in (self.ids//s.config.query_group_size).unique().tolist():
    if start<=group<start+len(old):self.priors[group]=r[0][group-start].clone();self.histories[group]=old[group-start].clone()
   return r
  history.build_prior=prior
  if self.new_backend:
   import saf3r.triton.current_probe_probability as probe
   self.current_probability_original=probe.probability;self.observed_positions=None
   def observed(q,k,patches,group_size,probes,scale):
    result=self.current_probability_original(q,k,patches,group_size,probes,scale);self.observed_positions=result[1]
    for group in (self.ids//group_size).unique().tolist():
     self.priors[group]=result[0][group].clone()
     if self.s.history_ids is not None:self.histories[group]=self.s.history_ids[group].clone()
    return result
   probe.probability=observed
  return self
 def __exit__(self,*args):
  import saf3r.patch.reuse_sampled_history as history
  history.build_prior=self.prior_original
  if self.new_backend:
   import saf3r.triton.current_probe_probability as probe
   probe.probability=self.current_probability_original
  super().__exit__(*args)
 def measure(self,layer):
  super().measure(layer)
  import torch
  q,k,v=self.q,self.k,self.v;module=self.module;s=self.s
  ids=torch.cat([r[0] for r in self.samples]);support=torch.cat([r[1] for r in self.samples]);weights=torch.cat([r[2] for r in self.samples])
  p=(torch.bmm(q[:,ids].float(),k.float().transpose(1,2))*module.scale).softmax(-1);vf=v.float();dense=torch.bmm(p,vf)
  valid=support>=0;safe=support.clamp_min(0);bank=torch.zeros((len(ids),s.n),device=q.device).scatter_add_(1,safe,weights*valid)
  weighted=p*bank[None];mass=weighted.sum(-1);moment=torch.bmm(weighted,vf)
  mapping=torch.full((s.n,),-1,device=q.device,dtype=torch.long);mapping[s.patches]=torch.arange(len(s.patches),device=q.device)
  def error(out,target):
   # out [H,Q,C,D],target [H,Q,D];project actual multihead output.
   h,n,c,d=out.shape;x=out.permute(2,1,0,3).reshape(c*n,h*d);y=target.permute(1,0,2).reshape(n,h*d)
   y=torch.nn.functional.linear(y,module.proj.weight.float());x=torch.nn.functional.linear(x,module.proj.weight.float()).reshape(c,n,-1)
   return (x-y[None]).square().sum((1,2))/y.square().sum().clamp_min(1e-20)
  for group in (ids//s.config.query_group_size).unique().tolist():
   take=(ids//s.config.query_group_size)==group;first=int(take.nonzero()[0]);prob=p[:,take];rbank=bank[first];score=prob.mean((0,1));top=score.masked_fill(rbank>0,-1).topk(8).indices
   eligible=(rbank>0)&(mapping>=0);weak=score.mul(rbank).masked_fill(~eligible,torch.inf).argmin();weak_w=rbank[weak]
   mm=moment[:,take];zz=mass[:,take];target=dense[:,take];base=error((mm/zz[:,:,None])[:,:,None,:],target)[0]
   query_ids=ids[take];probe_id=min(group*s.config.query_group_size+s.config.query_group_size//8,s.n-1)
   slot=(query_ids==probe_id).nonzero().flatten()
   if len(slot):probe_probability=prob[:,slot[0],top].mean(0)
   else:probe_probability=(torch.bmm(q[:,probe_id:probe_id+1].float(),k.float().transpose(1,2))*module.scale).softmax(-1)[:,0,top].mean(0)
   if self.new_backend:
    qpos=self.observed_positions[group]
    observed_mean=(torch.bmm(q[:,qpos].float(),k.float().transpose(1,2))*module.scale).softmax(-1)[:,:,top].mean((0,1))
   else:observed_mean=probe_probability
   import numpy as np
   self.traces[f'L{layer}_G{group}']=np.packbits((rbank>0).cpu().numpy())
   pe=prob[:,:,top];ve=vf[:,top]
   po=prob[:,:,weak];vo=vf[:,weak]
   swapped=(mm[:,:,None,:]-weak_w*po[:,:,None,None]*vo[:,None,None,:]+pe[:,:,:,None]*ve[:,None,:,:])/(zz[:,:,None]-weak_w*po[:,:,None]+pe).clamp_min(1e-20)[:,:,:,None]
   swap_error=error(swapped,target)
   patch=mapping[top];labels=s.fine.labels[patch];rep=s.patches[s.fine.reps[labels]];rw=rbank[rep];present=rw>0
   pr=prob[:,:,rep];vr=vf[:,rep]
   own=(mm[:,:,None,:]-rw[None,None,:,None]*pr[:,:,:,None]*vr[:,None,:,:]+rw[None,None,:,None]*pe[:,:,:,None]*ve[:,None,:,:])/(zz[:,:,None]-rw[None,None,:]*pr+rw[None,None,:]*pe).clamp_min(1e-20)[:,:,:,None]
   own_error=error(own,target)
   distance=(ve-vr).square().sum(-1).mean(0);mean_distance=(ve.mean(0)-vr.mean(0)).square().sum(-1)
   estimated=self.priors[group][patch] if group in self.priors else torch.full((len(top),),torch.nan,device=q.device)
   observed=torch.isin(patch,self.histories[group]) if group in self.histories else torch.zeros(len(top),device=q.device,dtype=torch.bool)
   arrays=torch.stack((top.float(),score[top],labels.float(),rep.float(),present.float(),estimated,observed.float(),swap_error,own_error,mean_distance,distance,(prob[:,:,top].clamp_min(1e-30).log()-prob[:,:,rep].clamp_min(1e-30).log()).abs().mean((0,1)),probe_probability,observed_mean)).cpu().T.tolist()
   for x in arrays:
    key,probability,cluster,rep_id,has_rep,prior,history,swap,replace,md,fd,logitdiff,probe_probability,observed_mean=x;key=int(key);frame=key//(s.height*s.width+5);pixel=key%(s.height*s.width+5)-5
    self.keys.append(dict(layer=layer,query_group=group,key_id=key,key_frame=frame,key_y=pixel//s.width,key_x=pixel%s.width,cluster=int(cluster),rep_id=int(rep_id),omitted_reason='approximate_cluster' if has_rep else 'dropped_cluster',true_dense_probability_group_mean=probability,true_dense_probability_probe0=probe_probability,true_dense_probability_observed_mean=observed_mean,prior_target=('Current observed4-clamped-probe/head-mean,compare observed_mean;includes duplicate tail probes' if self.new_backend and self.s.config.current_probability_probes==4 else 'Probe0 head-mean,compare probe0'),estimated_prior_probability=prior if prior==prior else None,was_history=bool(history),base_relative_mse=float(base),fixed_budget_weakest_swap_relative_mse=swap,fixed_budget_weakest_swap_gain=float(base)-swap,weakest_key=int(weak),weakest_weight=float(weak_w),own_rep_swap_relative_mse=replace if has_rep else None,own_rep_swap_gain=float(base)-replace if has_rep else None,head_mean_value_distance_sq=md,per_head_value_distance_sq=fd,mean_abs_per_head_logit_difference=logitdiff))
  self.priors={};self.histories={}

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--only-selected',action='store_true');a=p.parse_args()
 import torch,yaml
 from addict import Dict
 from saf3r.utils.model_utils import build_model
 from saf3r.utils.data_utils import load_images
 from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
 from scripts.quality_metrics import quality_record,quality_markdown
 plan=json.loads(a.plan.read_text());selected=json.loads(a.selection.read_text());rows=[plan['candidates'][0]]+[dict(name=r['name'],config=json.loads((ROOT/r['config']).read_text())) for r in selected['candidates'][:1]]
 if a.only_selected:rows=rows[1:]
 torch.set_num_threads(2);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cfg=Dict(yaml.safe_load((ROOT/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']);cfg.pop('patch_module');cfg.pop('sparse_config_path');cfg.ckpt_path=plan['checkpoint'];model=build_model(cfg,'cuda').eval();images=load_images(model,'vggt',plan['calibration']['scannet']['image_paths']).cuda()
 for row in rows:
  state=install_reuse_residual_exploration(model,ReuseExploreConfig(**row['config']));out=a.output/row['name'];out.mkdir(parents=True,exist_ok=False);quality=quality_record({},'Offline same-budget single-Key sampled Attention counterfactual on independent ScanNet RGB; no geometric truth or complete Dense model')
  with Impact(model,state) as impact,torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(images)
  import numpy as np
  torch.cuda.synchronize();np.savez_compressed(out/'selected_key_bits.npz',**impact.traces);save(out/'result.json',dict(quality=quality,input=plan['calibration']['scannet'],config=row['config'],layers=impact.rows,keys=impact.keys,scope=__doc__,speedup_vs_dense=None));(out/'quality.md').write_text(quality_markdown(quality));del impact
  state.reset_handle.remove()
  for b in model.aggregator.global_blocks:b.attn=b.attn.original
  del model.reuse_explore_state
  del state,raw;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
