"""M5c pipeline: same estimator, with statistics synchronized once per layer.

Keep compact FP32 calibration statistics on the device. Old windows are popped
before new ones are saved, so one layer's cache is live, not two full layers.
"""
import torch
from . import reuse_m5c as reference
from .history_dense_probability import EPS
from saf3r.triton.history_probe_logits import selected_probe_logits
from saf3r.triton.reuse_m5c_equivalent import predict,correct

observe=reference.observe
mean_probes=reference.mean_probes


def build_prior(s,start,old_ids,obs):
    old=s.m5c_previous.pop(start)
    current=obs['coarse_grid_probability']
    ratio=(current+EPS)/(old['previous_coarse_grid_probability']+EPS)
    confidence=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
    corrected=old['previous_corrected_grid_probability']
    uniform_old=old['unobserved_uniform_probability']/corrected.clamp_min(EPS)
    base=current*(confidence*uniform_old+(1-confidence)/s.grid_sizes)
    prior=mean_probes(base,obs['weights'])[:,s.grid].contiguous()
    safe=old_ids.long().clamp_min(0);valid=old_ids>=0
    actual=s.patches[safe].masked_fill(~valid,-1).int().contiguous()
    predict(s.m5c_previous_q,s.m5c_previous_k,old_ids,actual,obs,old,
            s.grid,s.grid_sizes,prior,s.m5c_previous_scale,EPS)
    s.m5c_reconstruction_qk+=valid.sum()*4
    pi=torch.zeros((len(prior),s.fine.base_count),device=prior.device).scatter_add_(
        1,s.fine.labels[None].expand(len(prior),-1),prior)
    expected=mean_probes(current,obs['weights']).sum(-1)
    s.m5c_prior_errors.append((prior.sum(-1)-expected).abs().max())
    return prior,pi


def save_evidence(s,q,k,start,ids,actual,obs,scale):
    valid=ids>=0;regions=s.grid[ids.long().clamp_min(0)]
    count=torch.zeros((len(ids),s.regions),device=q.device).scatter_add_(1,regions,valid.float())
    remaining=s.grid_sizes[None]-count
    s.m5c_count_errors.append((remaining<0).any())
    raw=selected_probe_logits(q,k,actual,obs['positions'],scale)
    missing=(obs['grid_logits']+remaining.clamp_min(1).log()[:,None,None,:]).masked_fill(remaining[:,None,None,:]==0,-torch.inf)
    logz=torch.logaddexp(torch.logaddexp(missing.logsumexp(-1),raw.logsumexp(-1)),
                        obs['special_logits'].logsumexp(-1))[:,:,:,None]
    unobserved=(obs['grid_logits']-logz).exp()
    corrected=correct(raw,logz,ids,s.grid,unobserved*remaining[:,None,None,:])
    special=(obs['special_logits']-logz).exp()
    s.m5c_seed_errors.append((corrected.sum(-1)+special.sum(-1)-1).abs().max())
    # Special probability is checked above but never read from the next cache.
    s.m5c_next[start]=dict(history_dense_log_normalizer=logz.detach(),
        previous_coarse_grid_probability=obs['coarse_grid_probability'].detach(),
        previous_corrected_grid_probability=corrected.detach(),
        unobserved_uniform_probability=unobserved.detach())
    s.m5c_probe_qk+=valid.sum()*4


def host_bytes(cache):
    return sum(t.numel()*t.element_size() for w in cache.values() for t in w.values() if t.device.type=='cpu')


def device_bytes(cache):
    return sum(t.numel()*t.element_size() for w in cache.values() for t in w.values() if t.device.type=='cuda')
