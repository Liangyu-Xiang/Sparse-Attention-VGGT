"""Execution-only acceleration hooks for the frozen hierarchical mechanism."""
import torch
from . import reuse_m5c,hierarchical_reuse
from saf3r.triton import hierarchical_reuse_fast,reuse_compact,reuse_fine_fast
_installed=False

def install():
 global _installed
 if _installed:return
 _installed=True
 hierarchical_reuse.allocate=hierarchical_reuse_fast.allocate
 original_prepare=reuse_compact.prepare_geometry
 def prepare(z,labels,reps,population):
  geometry=original_prepare(z,labels,reps,population)
  if geometry is not None:geometry['fast_buckets']=reuse_fine_fast.buckets(population,geometry['width'])
  return geometry
 reuse_compact.prepare_geometry=prepare;reuse_compact.losses=reuse_fine_fast.losses
 reuse_m5c.save_evidence=save_evidence;reuse_m5c.build_prior=build_prior
 reuse_m5c.host_bytes=lambda cache:sum(t.numel()*t.element_size() for window in cache.values() for t in window.values() if t.device.type=='cpu')
 from .reuse_residual_exploration import ReuseResidualExploration
 original_forward=ReuseResidualExploration.forward
 def forward(self,*args,**kwargs):
  result=original_forward(self,*args,**kwargs)
  self.state.records[-1].update(execution_backend='hierarchical_fast_v1',history_cache_device='gpu_statistics_gpu_previous_qk',history_gpu_statistics_bytes=sum(t.numel()*t.element_size() for w in self.state.m5c_previous.values() for t in w.values()),history_cpu_bytes=0)
  return result
 ReuseResidualExploration.forward=forward

def save_evidence(s,q,k,start,ids,actual,obs,scale):
 from saf3r.triton.history_probe_logits import selected_probe_logits
 valid=ids>=0;regions=s.grid[ids.long().clamp_min(0)]
 count=torch.zeros((len(ids),s.regions),device=q.device).scatter_add_(1,regions,valid.float());remaining=s.grid_sizes[None]-count
 if bool((remaining<0).any()):raise ValueError('History count exceeds region population')
 from saf3r.triton.reuse_grouped_attention_fast import consume_probes
 raw=consume_probes(actual)
 reused=raw is not None
 if raw is None:raw=selected_probe_logits(q,k,actual,obs['positions'],scale)
 missing=(obs['grid_logits']+remaining.clamp_min(1).log()[:,None,None,:]).masked_fill(remaining[:,None,None,:]==0,-torch.inf)
 logz=torch.logaddexp(torch.logaddexp(missing.logsumexp(-1),raw.logsumexp(-1)),obs['special_logits'].logsumexp(-1))[:,:,:,None]
 probability=(raw-logz).exp();unobserved=(obs['grid_logits']-logz).exp()
 corrected=(unobserved*remaining[:,None,None,:]).scatter_add_(-1,regions[:,None,None,:].expand_as(probability),probability)
 special=(obs['special_logits']-logz).exp();s.m5c_seed_errors.append((corrected.sum(-1)+special.sum(-1)-1).abs().max())
 cache=dict(history_dense_log_normalizer=logz,previous_coarse_grid_probability=obs['coarse_grid_probability'],previous_corrected_grid_probability=corrected,previous_corrected_special_probability=special,unobserved_uniform_probability=unobserved)
 s.m5c_next[start]={key:value.detach() for key,value in cache.items()};s.m5c_probe_qk+=0 if reused else int(valid.sum())*4


def build_prior(s,start,old_ids,obs):
 from .history_dense_probability import EPS
 from saf3r.triton.reuse_m5c_fast import predict
 old=s.m5c_previous.pop(start);device=old_ids.device;safe=old_ids.long().clamp_min(0);valid=old_ids>=0
 current=obs['coarse_grid_probability'];previous=old['previous_coarse_grid_probability']
 ratio=(current+EPS)/(previous+EPS);confidence=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
 corrected=old['previous_corrected_grid_probability'];uniform_old=old['unobserved_uniform_probability']/corrected.clamp_min(EPS)
 base=current*(confidence*uniform_old+(1-confidence)/s.grid_sizes)
 prior=reuse_m5c.mean_probes(base,obs['weights'])[:,s.grid].clone()
 actual=s.patches[safe].masked_fill(~valid,-1).int().contiguous()
 predict(s.m5c_previous_q,s.m5c_previous_k,old_ids,actual,obs,old,s.grid,s.grid_sizes,prior,s.m5c_previous_scale,EPS)
 s.m5c_reconstruction_qk+=int(valid.sum())*4
 pi=torch.zeros((len(prior),s.fine.base_count),device=device).scatter_add_(1,s.fine.labels[None].expand(len(prior),-1),prior)
 expected=reuse_m5c.mean_probes(current,obs['weights']).sum(-1);s.m5c_prior_errors.append((prior.sum(-1)-expected).abs().max())
 return prior,pi
