"""Sparse probability history, bias removal and coarse missing-member completion.

This intentionally changes v8: ratio of head-mean scores replaces the mean of
per-head Grid conditionals. No legacy all-query QK/denominator/cache is called.
"""
import math
import torch
from . import reuse_ready_history as cache
from .reuse_direct_coarse_observation import observe
from .reuse_m5c import mean_probes
from .history_dense_probability import EPS
from saf3r.triton import reuse_probability_history as kernels


def ensure_cache(s,q):
    # A log cache has signed values; positive-u14 probability packing is invalid.
    s.ready_cache_plain=True;s.ready_cache_dtype=torch.float16
    cache.ensure_cache(s,q[:1])


def build_prior(s,start,old_ids,obs):
    if s.ready_written.get(start)!=s.previous_layer:raise RuntimeError('Stale sparse-probability history')
    g=len(old_ids);end=start+g
    current=obs['coarse_grid_probability'].mean(1,keepdim=True)
    previous=s.ready_coarse[start:end]
    ratio=(current+EPS)/(previous+EPS)
    beta=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
    # Complete unselected members before forming the current cluster statistics.
    base=current*(beta*s.ready_uniform[start:end]+(1-beta)/s.grid_sizes)
    prior=mean_probes(base,obs['weights'])[:,s.grid].contiguous()
    kernels.predict(cache.cache_read(s,start,g),old_ids,s.grid,s.grid_sizes,current,previous,obs['weights'],prior,EPS)
    pi=torch.zeros((g,s.fine.base_count),device=prior.device).scatter_add_(1,s.fine.labels[None].expand(g,-1),prior)
    expected=mean_probes(current,obs['weights']).sum(-1)
    s.m5c_prior_errors.append((prior.sum(-1)-expected).abs().max())
    cache.prefetch(s,end,min(s.ready_window,s.groups-end))
    return prior,pi


def observe_execution(s,q,k,start,ids,actual,bias,obs,scale,evidence):
    # Probe means replace the expensive all-32-query Grid summary. This summary
    # is used only for residual candidate exploration, not the calibrated prior.
    scores=evidence['debiased_log_patch_probability'].contiguous()
    evidence['debiased_log_patch_probability']=scores
    probability=(scores+bias[:,None,:]).exp().masked_fill(ids[:,None,:]<0,0.)
    regions=s.grid[ids.long().clamp_min(0)]
    per_grid=torch.zeros((len(ids),4,s.regions),device=q.device)
    per_grid.scatter_add_(2,regions[:,None,:].expand(-1,4,-1),probability)
    original=torch.zeros_like(per_grid)
    original.scatter_add_(2,regions[:,None,:].expand(-1,4,-1),scores.exp().masked_fill(ids[:,None,:]<0,0.))
    mass=torch.stack(((per_grid*obs['weights'][:,:,None]).sum(1),(original*obs['weights'][:,:,None]).sum(1)),1)
    obs['probability_evidence']=evidence
    return mass


def save_evidence(s,q,k,start,ids,actual,obs,scale):
    end=start+len(ids);evidence=obs.pop('probability_evidence')
    local=obs['positions']-start*32
    # Same compensated sparse denominator as each head's selected probabilities.
    # This is a small H x 4 x Grid operation, with no additional Grid QK.
    lse=evidence['lse'][:,local].permute(1,0,2)
    missing_scores=torch.logsumexp(obs['grid_logits']-lse[:,:,:,None],dim=1)-math.log(q.shape[0])
    uniform,counts,den=kernels.store(evidence['debiased_log_patch_probability'],missing_scores,ids,s.grid,s.grid_sizes,cache.cache_write(s,start,len(ids)))
    cache.cache_written(s,start,len(ids))
    s.ready_uniform[start:end,0].copy_(uniform)
    s.ready_coarse[start:end].copy_(obs['coarse_grid_probability'].mean(1,keepdim=True))
    s.ready_written[start]=s.previous_layer+1
    s.m5c_count_errors.append((counts>s.grid_sizes).any())
    # Full normalization checked in the numerical diagnostic; FP16 log-cache
    # quantization is explicitly included in next-layer prior error statistics.
    s.m5c_seed_errors.append(torch.where(torch.isfinite(den),0.,float('inf')).amax())


release=cache.release
trim=cache.trim
host_bytes=cache.host_bytes
device_bytes=cache.device_bytes
