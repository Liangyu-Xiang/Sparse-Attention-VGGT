"""Original v8 D2+M5c outside the Grid-free AV path; no actual-P-as-prior change."""
import torch
from contextlib import nullcontext
from . import reuse_headmean_history as old
from . import reuse_ready_history as cache
from .reuse_probe_cache import observe as original_observe
from saf3r.triton import reuse_direct_legacy_observation as kernels
from saf3r.triton import reuse_grid_local_history as local

ensure_cache=old.ensure_cache
build_prior=old.build_prior
release=old.release
trim=old.trim
host_bytes=old.host_bytes
device_bytes=old.device_bytes

def observe(s,q,k,start,g,scale):
    obs=original_observe(s,q,k,start,g,scale)
    # Same pooled FP32 Key/probe logits; two required normalization domains.
    patch=(obs['grid_logits']+s.grid_log_sizes).softmax(-1).mean((1,2))
    obs['patch_probability']=patch
    return obs

def observe_execution(s,q,k,start,ids,actual,bias,obs,scale,evidence):
    scope=getattr(s,'direct_scope',lambda name:nullcontext());first=start*32;end=min(s.n,(start+len(ids))*32);qw=q[:,first:end]
    positions=(obs['positions']-first).int().contiguous();grid=s.grid[ids.long().clamp_min(0)].contiguous()
    with scope('legacy_full_query_grid_and_probe_observation'):
        summary,den,counts=kernels.observe(qw,k,actual,grid,bias,positions,evidence['lse'],obs['grid_logits'].contiguous(),s.grid_sizes,scale)
    obs.update(direct_den=den,direct_positions=positions,direct_grid=grid,direct_counts=counts)
    return summary

def save_evidence(s,q,k,start,ids,actual,obs,scale):
    scope=getattr(s,'direct_scope',lambda name:nullcontext());end=start+len(ids);first=start*32;last=min(s.n,end*32);den=obs['direct_den']
    with scope('legacy_grid_conditional_cache'):
        logz=local.compatibility(den,obs['special_logits'])
        import math,triton
        h=den.shape[1];r=den.shape[-1]
        local._uniform[(len(ids),4,triton.cdiv(r,8))](obs['grid_logits'].contiguous(),den,logz,s.ready_uniform[start:end],h,r,triton.next_power_of_2(h),8,math.log(1e-30),True,num_warps=4)
        kernels.store(q[:,first:last],k,actual,obs['direct_grid'],obs['direct_positions'],den,logz,cache.cache_write(s,start,len(ids)),scale)
        cache.cache_written(s,start,len(ids));s.ready_coarse[start:end].copy_(obs['coarse_grid_probability'].mean(1,keepdim=True))
        s.ready_written[start]=s.previous_layer+1
        s.m5c_count_errors.append((obs['direct_counts']>s.grid_sizes).any())
        normalization=(den-logz[:,:,:,None]).exp().sum(-1)+(obs['special_logits']-logz[:,:,:,None]).exp().sum(-1)
        s.m5c_seed_errors.append((normalization-1).abs().max())
