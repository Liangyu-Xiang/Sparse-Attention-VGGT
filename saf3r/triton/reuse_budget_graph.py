"""Replay the unchanged16-round budget solver with fresh inputs on every call."""
import torch
from .reuse_compact import budget as reference
_CACHE={}

def budget(errors,costs,upper,cap,iterations):
    key=(errors.device,errors.dtype,tuple(errors.shape),cap,iterations)
    if key not in _CACHE:
        e=torch.empty_like(errors);c=torch.empty_like(costs);u=torch.empty_like(upper)
        e.copy_(errors);c.copy_(costs);u.copy_(upper)
        # Compile and warm all original kernels before entering capture.
        reference(e,c,u,cap,iterations)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):selected,count=reference(e,c,u,cap,iterations)
        _CACHE[key]=(e,c,u,graph,selected,count)
    e,c,u,graph,selected,count=_CACHE[key]
    e.copy_(errors);c.copy_(costs);u.copy_(upper)
    graph.replay()
    return selected,count
