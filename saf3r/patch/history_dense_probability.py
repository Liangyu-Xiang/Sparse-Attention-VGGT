"""Full-Key probability estimators from the supplied history-transfer document.

Special tokens are singleton regions. All calculations precede probe/head mean.
No population bias is used in D2, and no Dense oracle is accepted by this API.
"""
import torch
ASYMMETRIC=((.5,1.),(1.,.5),(2.,1.),(1.,2.))
EPS=1e-30

def coarse(grid_logits,sizes,special_logits):
 logits=torch.cat((grid_logits+sizes.log(),special_logits),-1)
 return logits.softmax(-1)

def estimate_previous_dense_probs(grid_logits,sizes,edge_regions,edge_logits,special_logits):
 """Edges must be distinct. Full region vector includes singleton specials."""
 count=torch.bincount(edge_regions,minlength=len(sizes)).to(sizes)
 if bool((count>sizes).any()):raise ValueError('More observed edges than region members')
 remaining=sizes-count
 # Positive logsumexp, algebraically identical to baseline subtraction, avoids
 # cancellation when almost every member has been observed.
 missing=grid_logits+remaining.clamp_min(1).log()
 missing=missing.masked_fill(remaining==0,-torch.inf)
 logz=torch.cat((missing,edge_logits,special_logits),-1).logsumexp(-1,keepdim=True)
 edges=(edge_logits-logz).exp()
 unobserved=(grid_logits-logz).exp()
 patch=unobserved*remaining
 patch=patch.scatter_add(-1,edge_regions.expand_as(edges),edges)
 specials=(special_logits-logz).exp()
 return dict(edge_prob=edges,full_grid_prob=torch.cat((patch,specials),-1),
  coarse_grid_prob=coarse(grid_logits,sizes,special_logits),unobserved_uniform_prob=unobserved,
  logz=logz,normalization_mode='all_keys_including_special_singletons')

def beta(r,mode='old',down=1.,up=1.):
 if mode=='old':return r.clamp_max(1)
 if mode=='symmetric':return torch.minimum(r,1/r).clamp(0,1)
 return torch.where(r<1,r.pow(down),r.pow(-up)).clamp(0,1)

def transfer_history_probs(history,current_coarse_grid_prob,edge_regions,sizes,method='M7b'):
 prev=history['coarse_grid_prob'];full=history['full_grid_prob'];p=history['edge_prob']
 r=(current_coarse_grid_prob+EPS)/(prev+EPS)
 regions=edge_regions.expand_as(p)
 if method=='M0':grid=full;edge=p;normalized=False
 elif method in ('M1','M2','M3'):
  factor=r if method=='M1' else beta(r,'old' if method=='M2' else 'symmetric')
  grid=full*factor;edge=p*factor.gather(-1,regions);normalized=False
 else:
  transported=full*r;transported=transported/transported.sum(-1,keepdim=True).clamp_min(EPS)
  grid=transported if method.startswith(('M6','M7')) else current_coarse_grid_prob
  condition=p/full.gather(-1,regions).clamp_min(EPS)
  uniform=1/sizes[edge_regions]
  if method=='M8':edge=grid.gather(-1,regions)*uniform
  else:
   confidence=torch.ones_like(r)
   if method.startswith(('M5','M7')):
    suffix=method[2:]
    if suffix in ('a','b'):confidence=beta(r,'old' if suffix=='a' else 'symmetric')
    else:
     down,up=map(float,suffix.removeprefix('c_').split('_'))
     confidence=beta(r,'asymmetric',down,up)
   mix=confidence.gather(-1,regions)*condition+(1-confidence.gather(-1,regions))*uniform
   edge=grid.gather(-1,regions)*mix
  normalized=True
 return dict(edge_probability=edge,grid_probability=grid,is_normalized_probability=normalized,
  normalization_sum=grid.sum(-1),ratio=r)

METHODS=['M0','M1','M2','M3','M4','M5a','M5b','M6','M7a','M7b','M8']+[f'{m}c_{d}_{u}' for m in ('M5','M7') for d,u in ASYMMETRIC]
