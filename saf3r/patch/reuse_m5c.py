"""Production D2+M5c evidence, full-Key normalization, windowed FP32 statistics and lazy exact old-QK reconstruction.

No Dense oracle or current-QK scan of the previous history bank is used.
"""
import torch
from saf3r.triton.history_probe_logits import selected_probe_logits
from .history_dense_probability import EPS

def mean_probes(value,weights):
 return (value*weights[:,None,:,None]).sum(2).mean(1)

def observe(s,q,k,start,g,scale):
 positions=(torch.arange(start,start+g,device=q.device)[:,None]*32+torch.tensor([4,12,20,28],device=q.device)).clamp_max(s.n-1)
 unique=torch.ones_like(positions,dtype=torch.bool);unique[:,1:]=positions[:,1:]!=positions[:,:-1]
 weights=unique.float()/unique.sum(-1,keepdim=True)
 pq=q[:,positions].permute(1,0,2,3).float()
 grid=torch.einsum('ghqd,hrd->ghqr',pq,s.history_pooled_fp32)*scale
 special=torch.einsum('ghqd,htd->ghqt',pq,k[:,s.specials].float())*scale
 logz=torch.logaddexp((grid+s.grid_sizes.log()).logsumexp(-1),special.logsumexp(-1))[:,:,:,None]
 coarse=(grid-logz).exp()*s.grid_sizes
 return dict(positions=positions,weights=weights,grid_logits=grid,special_logits=special,coarse_grid_probability=coarse,coarse_logz=logz)

def build_prior(s,start,old_ids,obs):
 old=s.m5c_previous.pop(start)
 device=old_ids.device;safe=old_ids.long().clamp_min(0);valid=old_ids>=0;regions=s.grid[safe]
 current=obs['coarse_grid_probability']
 previous=old['previous_coarse_grid_probability'].to(device)
 ratio=(current+EPS)/(previous+EPS)
 confidence=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
 corrected=old['previous_corrected_grid_probability'].to(device)
 uniform_old=old['unobserved_uniform_probability'].to(device)/corrected.clamp_min(EPS)
 base=current*(confidence*uniform_old+(1-confidence)/s.grid_sizes)
 prior=mean_probes(base,obs['weights'])[:,s.grid].clone()
 index=regions[:,None,None,:].expand(-1,current.shape[1],4,-1)
 actual=s.patches[safe].masked_fill(~valid,-1).int().contiguous()
 previous_raw=selected_probe_logits(s.m5c_previous_q,s.m5c_previous_k,actual,obs['positions'],s.m5c_previous_scale)
 previous_probability=(previous_raw-old['history_dense_log_normalizer'].to(device)).exp()
 s.m5c_reconstruction_qk+=int(valid.sum())*4
 conditional=previous_probability/corrected.gather(-1,index).clamp_min(EPS)
 predicted=current.gather(-1,index)*(confidence.gather(-1,index)*conditional+(1-confidence.gather(-1,index))/s.grid_sizes[regions][:,None,None,:])
 edge=mean_probes(predicted,obs['weights']).masked_fill(~valid,0)
 padded=torch.cat((prior,torch.zeros((len(prior),1),device=device)),1)
 padded.scatter_(1,torch.where(valid,safe,len(s.grid)),edge)
 prior=padded[:,:len(s.grid)].contiguous()
 pi=torch.zeros((len(prior),s.fine.base_count),device=device).scatter_add_(1,s.fine.labels[None].expand(len(prior),-1),prior)
 # Expected full Patch mass is current coarse Patch mass, not forcibly one.
 expected=mean_probes(current,obs['weights']).sum(-1)
 s.m5c_prior_errors.append((prior.sum(-1)-expected).abs().max())
 return prior,pi

def save_evidence(s,q,k,start,ids,actual,obs,scale):
 valid=ids>=0;regions=s.grid[ids.long().clamp_min(0)]
 # Count once per unique patch ID (native packed supports are unique).
 count=torch.zeros((len(ids),s.regions),device=q.device).scatter_add_(1,regions,valid.float())
 remaining=s.grid_sizes[None]-count
 if bool((remaining<0).any()):raise ValueError('History count exceeds region population')
 raw=selected_probe_logits(q,k,actual,obs['positions'],scale)
 missing=(obs['grid_logits']+remaining.clamp_min(1).log()[:,None,None,:]).masked_fill(remaining[:,None,None,:]==0,-torch.inf)
 known_logsum=raw.logsumexp(-1)
 special_logsum=obs['special_logits'].logsumexp(-1)
 logz=torch.logaddexp(torch.logaddexp(missing.logsumexp(-1),known_logsum),special_logsum)[:,:,:,None]
 probability=(raw-logz).exp()
 unobserved=(obs['grid_logits']-logz).exp()
 corrected=(unobserved*remaining[:,None,None,:]).scatter_add_(-1,regions[:,None,None,:].expand_as(probability),probability)
 special=(obs['special_logits']-logz).exp()
 error=(corrected.sum(-1)+special.sum(-1)-1).abs().max();s.m5c_seed_errors.append(error)
 cache=dict(history_dense_log_normalizer=logz,previous_coarse_grid_probability=obs['coarse_grid_probability'],previous_corrected_grid_probability=corrected,
  previous_corrected_special_probability=special,unobserved_uniform_probability=unobserved)
 # Blocking copies are intentional: bounded GPU scratch and no stream races.
 # Old windows are released on consumption before new windows are cached.
 s.m5c_next[start]={key:value.detach().to('cpu') for key,value in cache.items()}
 s.m5c_probe_qk+=int(valid.sum())*4

def host_bytes(cache):return sum(t.numel()*t.element_size() for window in cache.values() for t in window.values())
