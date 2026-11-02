"""Head-mean cross-layer conditionals; no head identity is transferred."""
import torch
from . import reuse_ready_history as cache
from .reuse_m5c import observe,mean_probes
from .history_dense_probability import EPS
from saf3r.triton.reuse_ready_history import calibrate,logsumexp,predict
from saf3r.triton.reuse_headmean_history import store_mean


def ensure_cache(s,q):
    cache.ensure_cache(s,q[:1])


def build_prior(s,start,old_ids,obs):
    if s.ready_written.get(start)!=s.previous_layer:raise RuntimeError('Stale head-mean history')
    g=len(old_ids)
    current=obs['coarse_grid_probability'].mean(1,keepdim=True)
    previous=s.ready_coarse[start:start+g]
    ratio=(current+EPS)/(previous+EPS)
    confidence=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
    base=current*(confidence*s.ready_uniform[start:start+g]+(1-confidence)/s.grid_sizes)
    prior=mean_probes(base,obs['weights'])[:,s.grid].contiguous()
    predict(cache.cache_read(s,start,g),old_ids,s.grid,s.grid_sizes,current,previous,obs['weights'],prior,EPS)
    pi=torch.zeros((g,s.fine.base_count),device=prior.device).scatter_add_(1,s.fine.labels[None].expand(g,-1),prior)
    expected=mean_probes(current,obs['weights']).sum(-1)
    s.m5c_prior_errors.append((prior.sum(-1)-expected).abs().max())
    return prior,pi


def save_evidence(s,q,k,start,ids,actual,obs,scale):
    valid=ids>=0;regions=s.grid[ids.long().clamp_min(0)]
    count=torch.zeros((len(ids),s.regions),device=ids.device).scatter_add_(1,regions,valid.float())
    remaining=s.grid_sizes[None]-count
    s.m5c_count_errors.append((remaining<0).any())
    raw=obs['ready_logits']
    missing=(obs['grid_logits']+remaining.clamp_min(1).log()[:,None,None,:]).masked_fill(remaining[:,None,None,:]==0,-torch.inf)
    logz=torch.logaddexp(torch.logaddexp(missing.logsumexp(-1),logsumexp(raw)),obs['special_logits'].logsumexp(-1))[:,:,:,None]
    unobserved=(obs['grid_logits']-logz).exp()
    corrected=calibrate(raw,logz,ids,s.grid,unobserved*remaining[:,None,None,:],regions)
    special=(obs['special_logits']-logz).exp()
    s.m5c_seed_errors.append((corrected.sum(-1)+special.sum(-1)-1).abs().max())
    end=start+len(ids)
    # Average conditional probabilities AFTER each current-layer head has its
    # own normalization. No logit averaging and no cross-layer head pairing.
    store_mean(raw,logz,ids,s.grid,corrected,cache.cache_write(s,start,len(ids)),EPS)
    cache.cache_written(s,start,len(ids))
    s.ready_uniform[start:end].copy_((unobserved/corrected.clamp_min(EPS)).mean(1,keepdim=True))
    s.ready_coarse[start:end].copy_(obs['coarse_grid_probability'].mean(1,keepdim=True))
    s.ready_written[start]=s.previous_layer+1

release=cache.release
trim=cache.trim
host_bytes=cache.host_bytes
device_bytes=cache.device_bytes
