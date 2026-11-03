"""One confidence per query group over the full patch-key domain.

Dense probability, not selected-support-normalized probability, is cached.
Pooled spatial anchors approximate the missing partition function; they do not
partition the calibration coefficient or the formal Attention execution.
"""
import torch
from .reuse_m5c import observe
from saf3r.triton.reuse_global_history import store
EPS=1.e-12

def mean_probes(value,obs):
 return (value.mean(1)*obs['weights'][...,None]).sum(1)

def current(s,obs):
 anchor=mean_probes(obs['coarse_grid_probability'],obs);eta=anchor.sum(1)
 return eta,anchor/eta[:,None].clamp_min(EPS)

def ensure_cache(s,q):
 s.global_edges=torch.empty((s.groups,s.budget),device=q.device)
 s.global_eta=torch.empty(s.groups,device=q.device);s.global_background=torch.empty_like(s.global_eta)
 s.global_anchor=torch.empty((s.groups,s.regions),device=q.device)
 s.ready_edges=s.global_edges

def components(s,start,old_ids,obs):
 if s.ready_written.get(start)!=s.previous_layer:raise RuntimeError('Stale global history')
 end=start+len(old_ids);eta,anchor=current(s,obs)
 confidence=(1.-.5*(anchor-s.global_anchor[start:end]).abs().sum(1)).clamp(0,1)
 oldeta=s.global_eta[start:end].clamp_min(EPS)
 uniform=(1-confidence)/len(s.patches)
 background=eta*(confidence*s.global_background[start:end]/oldeta+uniform)
 historical=eta[:,None]*(confidence[:,None]*s.global_edges[start:end]/oldeta[:,None]+uniform[:,None])
 historical=historical.masked_fill(old_ids<0,0.)
 return background,historical,dict(eta=eta,anchor=anchor,confidence=confidence)

def build_prior(s,start,old_ids,obs):
 background,historical,aux=components(s,start,old_ids,obs)
 g=len(old_ids);n=len(s.patches);valid=old_ids>=0
 padded=background[:,None].expand(g,n+1).clone()
 padded.scatter_(1,old_ids.long().masked_fill(~valid,n),historical.masked_fill(~valid,0.))
 prior=padded[:,:n].contiguous()
 pi=torch.zeros((g,s.fine.base_count),device=prior.device).scatter_add_(1,s.fine.labels[None].expand(g,-1),prior)
 s.m5c_prior_errors.append((prior.sum(1)-aux['eta']).abs().max())
 return prior,pi

def produce(s,start,ids,obs,raw,selected_lse=None,special_lse=None,denominator='partition'):
 valid=ids>=0;eta,anchor=current(s,obs);g=len(ids)
 selected_lse=raw.logsumexp(-1) if selected_lse is None else selected_lse
 special_lse=obs['special_logits'].logsumexp(-1) if special_lse is None else special_lse
 if denominator=='partition':
  count=torch.zeros((g,s.regions),device=ids.device).scatter_add_(1,s.grid[ids.long().clamp_min(0)],valid.float())
  remaining=s.grid_sizes[None]-count
  s.m5c_count_errors.append((remaining<0).any())
  missing=(obs['grid_logits']+remaining.clamp_min(1).log()[:,None,None,:]).masked_fill(remaining[:,None,None,:]<=0,-torch.inf).logsumexp(-1)
 elif denominator=='global':
  # Independent approximation: missing members have global anchor mean mass.
  fraction=1.-valid.sum(1).float()/len(s.patches)
  missing=(obs['grid_logits']+s.grid_sizes.log()).logsumexp(-1)+fraction.clamp_min(EPS).log()[:,None,None]
  missing=missing.masked_fill(fraction[:,None,None]<=0,-torch.inf)
 else:raise ValueError('Unknown denominator estimator')
 logz=torch.logaddexp(torch.logaddexp(selected_lse,missing),special_lse)
 probabilities=store(raw,logz,obs['weights'],ids)
 special=(special_lse-logz).exp()
 patch_eta=1.-(special.mean(1)*obs['weights']).sum(1)
 residual=patch_eta-probabilities.sum(1)
 s.m5c_seed_errors.append((-residual).clamp_min(0).max())
 end=start+g;s.global_edges[start:end].copy_(probabilities);s.global_eta[start:end].copy_(patch_eta)
 s.global_background[start:end].copy_(residual.clamp_min(0)/(len(s.patches)-valid.sum(1)).clamp_min(1))
 s.global_anchor[start:end].copy_(anchor);s.ready_written[start]=s.previous_layer+1
 return probabilities,patch_eta

def save_evidence(s,q,k,start,ids,actual,obs,scale):
 return produce(s,start,ids,obs,obs['ready_logits'])

def release(s):
 s.ready_edges=None
 for name in ['global_edges','global_eta','global_background','global_anchor']:setattr(s,name,None)

def trim(s):pass

def host_bytes(s):return 0

def device_bytes(s):
 return sum(t.numel()*t.element_size() for t in [getattr(s,k,None) for k in ['global_edges','global_eta','global_background','global_anchor']] if t is not None)
