"""Cache only shape-derived probe indices/weights; recompute Q/K every layer."""
import torch

def prepare(s):
    offsets=[4,12,20,28] if getattr(s.config,'probe_count',4)==4 else [16]
    positions=(torch.arange(s.groups,device=s.grid.device)[:,None]*32+torch.tensor(offsets,device=s.grid.device)).clamp_max(s.n-1)
    unique=torch.ones_like(positions,dtype=torch.bool);unique[:,1:]=positions[:,1:]!=positions[:,:-1]
    s.probe_positions=positions;s.probe_weights=unique.float()/unique.sum(-1,keepdim=True)
    s.grid_log_sizes=s.grid_sizes.log()

def observe(s,q,k,start,g,scale):
    positions=s.probe_positions[start:start+g];weights=s.probe_weights[start:start+g]
    probe_q=q[:,positions];pq=probe_q.permute(1,0,2,3).float()
    grid=torch.einsum('ghqd,hrd->ghqr',pq,s.history_pooled_fp32)*scale
    special=torch.einsum('ghqd,htd->ghqt',pq,k[:,s.specials].float())*scale
    if getattr(getattr(s,'config',None),'fused_coarse',False) and grid.shape[2]==4:
        from saf3r.triton.reuse_coarse_observation import m5c
        coarse,logz=m5c(grid,special,s.grid_sizes,s.grid_log_sizes)
    else:
        logz=torch.logaddexp((grid+s.grid_log_sizes).logsumexp(-1),special.logsumexp(-1))[:,:,:,None]
        coarse=(grid-logz).exp()*s.grid_sizes
    return dict(positions=positions,weights=weights,grid_logits=grid,special_logits=special,coarse_grid_probability=coarse,coarse_logz=logz,probe_q=probe_q)
