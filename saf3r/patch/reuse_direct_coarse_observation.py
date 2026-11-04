"""One FP32 pooled-Key QK supplies both original coarse normalization domains.

Pooled Keys are built once per layer by ReuseExploreState.prepare. M5c includes
Specials; residual demand normalizes Patch Grids only,as in the original method.
"""
from .reuse_probe_cache import observe as _observe

def observe(s,q,k,start,g,scale):
    obs=_observe(s,q,k,start,g,scale)
    obs['patch_probability']=(obs['grid_logits']+s.grid_log_sizes).softmax(-1).mean((1,2))
    return obs
