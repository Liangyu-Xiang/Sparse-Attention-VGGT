"""One actual sparse Probe per32/256 Queries,compact sampled-only history.

Unobserved Query Groups use their OWN coarse-uniform prior. Keys and probabilities
are never copied from a neighboring Query Group with a different sparse support.
Coarse/residual observers retain four Probes;the history uses Probe0 consistently.
"""
import math
import torch
import triton
from .reuse_direct_coarse_observation import observe
from .history_dense_probability import EPS
from saf3r.triton import reuse_probability_history as completion
from saf3r.triton.reuse_sampled_history import predict


def stride(s):return s.config.history_query_interval//s.config.query_group_size


def ensure_cache(s,q):
    step=stride(s);groups=triton.cdiv(s.groups,step)
    # Only sampled groups and ONE Probe are stored,without a dense four-Probe bank.
    if groups*s.budget*2>8*2**30:raise MemoryError('Sampled history exceeds the explicit8GiB device-cache limit')
    s.ready_edges=torch.empty((groups,1,1,s.budget),device=q.device,dtype=torch.float16)
    s.ready_uniform=torch.empty((groups,s.regions),device=q.device,dtype=torch.float32)
    s.ready_coarse=torch.empty_like(s.ready_uniform)


def build_prior(s,start,old_ids,obs,return_uniform=False):
    if s.ready_written.get(start)!=s.previous_layer:raise RuntimeError('Stale sampled history')
    step=stride(s);g=len(old_ids);begin=start//step;count=triton.cdiv(g,step)
    current=obs['coarse_grid_probability'][:,:,0,:].mean(1).contiguous()
    # No history in the other groups:uniform conditional within each coarse Grid.
    base=current/s.grid_sizes
    cur=current[::step].contiguous();previous=s.ready_coarse[begin:begin+count]
    ratio=(cur+EPS)/(previous+EPS);beta=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
    base[::step]=cur*(beta*s.ready_uniform[begin:begin+count]+(1-beta)/s.grid_sizes)
    prior=base[:,s.grid].contiguous()
    predict(s.ready_edges[begin:begin+count],old_ids[::step].contiguous(),s.grid,s.grid_sizes,cur,previous,prior,step,EPS)
    pi=torch.zeros((g,s.fine.base_count),device=prior.device).scatter_add_(1,s.fine.labels[None].expand(g,-1),prior)
    s.m5c_prior_errors.append((prior.sum(-1)-current.sum(-1)).abs().max())
    return (prior,pi,base) if return_uniform else (prior,pi)


def observe_execution(s,q,k,start,ids,actual,bias,obs,scale,evidence):
    step=stride(s);g=len(ids);regions=s.grid[ids.long().clamp_min(0)]
    counts=torch.zeros((g,s.regions),device=q.device).scatter_add_(1,regions,(ids>=0).float())
    # Residual-candidate summary for groups lacking actual Attention probabilities.
    total=obs['patch_probability'].clone();unit=total*counts/s.grid_sizes
    scores=evidence['debiased_log_patch_probability'].contiguous()
    evidence['debiased_log_patch_probability']=scores
    sampled_ids=ids[::step].contiguous();sampled_regions=regions[::step,None,:].contiguous()
    probability=(scores+bias[::step,None,:]).exp().masked_fill(sampled_ids[:,None,:]<0,0.)
    own=scores.exp().masked_fill(sampled_ids[:,None,:]<0,0.)
    sampled_total=torch.zeros((len(sampled_ids),1,s.regions),device=q.device).scatter_add_(2,sampled_regions,probability)
    sampled_unit=torch.zeros_like(sampled_total).scatter_add_(2,sampled_regions,own)
    total[::step]=sampled_total[:,0];unit[::step]=sampled_unit[:,0]
    obs['sampled_evidence']=evidence
    return torch.stack((total,unit),1)


def save_evidence(s,q,k,start,ids,actual,obs,scale):
    step=stride(s);begin=start//step;evidence=obs.pop('sampled_evidence')
    selected=ids[::step].contiguous();count=len(selected)
    positions=obs['positions'][::step,:1]-start*s.config.query_group_size
    lse=evidence['lse'][:,positions].permute(1,0,2)
    missing=torch.logsumexp(obs['grid_logits'][::step,:,:1,:]-lse[:,:,:,None],1)-math.log(len(q))
    target=s.ready_edges[begin:begin+count]
    uniform,counts,den=completion.store(evidence['debiased_log_patch_probability'],missing,selected,s.grid,s.grid_sizes,target)
    s.ready_uniform[begin:begin+count].copy_(uniform[:,0])
    s.ready_coarse[begin:begin+count].copy_(obs['coarse_grid_probability'][::step,:,0,:].mean(1))
    s.ready_written[start]=s.previous_layer+1
    s.m5c_count_errors.append((counts>s.grid_sizes).any())
    # Measure the actual stored log-cache normalization,including FP16 rounding.
    sums=torch.zeros((count,1,s.regions),device=q.device);zero=torch.zeros_like(sums)
    completion._reduce[(triton.cdiv(selected.shape[1],256),1,count)](target,selected,s.grid,zero,sums,selected.shape[1],1,s.regions,256,False,num_warps=4)
    norm=sums[:,0]+uniform[:,0]*(s.grid_sizes[None]-counts)
    s.m5c_seed_errors.append((norm-1).abs().max())


def release(s):s.ready_edges=None;s.ready_uniform=None;s.ready_coarse=None;s.ready_written={}
def trim(s):pass
def host_bytes(s):return 0
def device_bytes(s):return sum(t.numel()*t.element_size() for t in (s.ready_edges,s.ready_uniform,s.ready_coarse) if t is not None)
