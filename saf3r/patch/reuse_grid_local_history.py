"""Grid-local replacement for evidence production; original consumer unchanged."""
import math
from contextlib import nullcontext
from . import reuse_headmean_history as reference
from . import reuse_ready_history as cache
from .reuse_m5c import observe
from .history_dense_probability import EPS
from saf3r.triton import reuse_grid_local_history as kernels

build_prior=reference.build_prior
release=reference.release
trim=reference.trim
host_bytes=reference.host_bytes
device_bytes=reference.device_bytes


def ensure_cache(s,q):
    s.ready_cache_plain=getattr(s.config,'history_cache_format','packed')=='fp16'
    reference.ensure_cache(s,q)


def save_evidence(s,q,k,start,ids,actual,obs,scale):
    # Q/K/actual/scale are API-compatible only; NO QK is computed here.
    scope=getattr(s,'grid_local_scope',lambda name:nullcontext())
    with scope('local_normalizers'):
        base=obs['grid_logits'].contiguous()
        order=obs['selected_grid_order'];offsets=obs['selected_grid_offsets']
        gy=min(s.config.grid_y,s.height);gx=min(s.config.grid_x,s.width)
        maximum=math.ceil(s.height/gy)*math.ceil(s.width/gx)
        den,error=kernels.normalizers(obs['ready_logits'],base,order,offsets,s.grid_sizes,maximum)
    with scope('local_eps_compatibility'):
        # Preserve old EPS clipping with only small grid/special statistics.
        logz=kernels.compatibility(den,obs['special_logits'])
    with scope('local_probability_headmean_cache'):
        end=start+len(ids)
        kernels.store(obs['ready_logits'],den,logz,ids,s.grid,base,cache.cache_write(s,start,len(ids)),s.ready_uniform[start:end],EPS)
        cache.cache_written(s,start,len(ids))
    with scope('local_checks_and_coarse_store'):
        s.m5c_count_errors.append(((offsets[:,1:]-offsets[:,:-1])>s.grid_sizes).any())
        s.m5c_seed_errors.append(error.max())
        s.ready_coarse[start:end].copy_(obs['coarse_grid_probability'].mean(1,keepdim=True))
        s.ready_written[start]=s.previous_layer+1
