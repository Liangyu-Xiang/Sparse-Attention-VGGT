"""New estimator: actual sparse P -> beta -> current Grid mass -> missing keys.

The state loss, allocator, retained-key budget and formal Attention are unchanged.
This is a modeling change, not an equivalent execution optimization.
"""
import torch
from contextlib import nullcontext
from . import reuse_headmean_history as reference
from . import reuse_ready_history as cache
from .reuse_m5c import observe,mean_probes
from .history_dense_probability import EPS
from saf3r.triton import reuse_sparse_beta_history as kernels


def calibrate(current,previous,mass,counts,sizes):
    """All probabilities are [G,4,R]; counts is [G,R]."""
    ratio=(current+EPS)/(previous+EPS)
    beta=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
    missing=sizes[None]-counts
    full=missing==0
    factor=torch.minimum(beta,current/mass.clamp_min(EPS))
    factor=torch.where(full[:,None,:],current/mass.clamp_min(EPS),factor)
    positive=mass>EPS
    factor=torch.where(positive,factor,0.)
    fallback=torch.where(full[:,None,:]&~positive,current/sizes[None,None,:],0.)
    selected=factor*mass+fallback*counts[:,None,:]
    uniform=torch.where(missing[:,None,:]>0,
        (current-selected).clamp_min(0)/missing[:,None,:].clamp_min(1),0.)
    return factor.contiguous(),fallback.contiguous(),uniform,beta


def ensure_cache(s,q):
    reference.ensure_cache(s,q)
    s.beta_selected_counts=torch.empty((s.groups,s.regions),device=q.device,dtype=torch.int32)


def build_prior(s,start,old_ids,obs):
    if s.ready_written.get(start)!=s.previous_layer:
        raise RuntimeError('Stale actual-sparse-probability history')
    scope=getattr(s,'grid_local_scope',lambda name:nullcontext())
    end=start+len(old_ids)
    with scope('beta_grid_calibration'):
        current=obs['coarse_grid_probability'].mean(1)
        previous=s.ready_coarse[start:end,0]
        mass=s.ready_uniform[start:end,0]
        counts=s.beta_selected_counts[start:end]
        factor,fallback,uniform,beta=calibrate(current,previous,mass,counts,s.grid_sizes)
    with scope('beta_dense_prior_materialization'):
        prior=mean_probes(uniform[:,None],obs['weights'])[:,s.grid].contiguous()
        kernels.apply(cache.cache_read(s,start,len(old_ids)),old_ids,s.grid,factor,fallback,obs['weights'],prior)
        pi=torch.zeros((len(old_ids),s.fine.base_count),device=prior.device).scatter_add_(1,s.fine.labels[None].expand(len(old_ids),-1),prior)
        expected=mean_probes(current[:,None],obs['weights']).sum(-1)
        s.m5c_prior_errors.append((prior.sum(-1)-expected).abs().max())
    return prior,pi


def save_evidence(s,q,k,start,ids,actual,obs,scale):
    scope=getattr(s,'grid_local_scope',lambda name:nullcontext())
    end=start+len(ids)
    with scope('actual_sparse_probability_export'):
        mass=s.ready_uniform[start:end]
        mass.zero_()
        kernels.export(obs['ready_logits'],obs['sparse_log_weights'],obs['sparse_lse'],
                       ids,s.grid,cache.cache_write(s,start,len(ids)),mass)
        cache.cache_written(s,start,len(ids))
    with scope('beta_counts_and_coarse_cache'):
        counts=obs['selected_grid_offsets'][:,1:]-obs['selected_grid_offsets'][:,:-1]
        s.beta_selected_counts[start:end].copy_(counts)
        s.m5c_count_errors.append((counts>s.grid_sizes).any())
        s.m5c_seed_errors.append((mass.sum(-1)-1).clamp_min(0).max())
        s.ready_coarse[start:end].copy_(obs['coarse_grid_probability'].mean(1,keepdim=True))
        s.ready_written[start]=s.previous_layer+1


def release(s):
    reference.release(s)
    s.beta_selected_counts=None


trim=reference.trim
host_bytes=reference.host_bytes
def device_bytes(s):
    value=reference.device_bytes(s)
    counts=getattr(s,'beta_selected_counts',None)
    return value+(0 if counts is None else counts.numel()*counts.element_size())
