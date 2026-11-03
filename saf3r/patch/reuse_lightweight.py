"""Lightweight global-history contribution model, isolated from reference Reuse.

Simple: change compensation + matched four-state loss, retain legacy Exact
qualification and Grid executor as a controlled intermediate.
Full: same state model; one observation, all clusters eligible for Exact,
indexed contiguous-tile Attention and direct cluster packing.
"""
from dataclasses import dataclass
import math,torch
from torch import nn
from .reuse_residual_exploration import ReuseExploreState
from .region_history_attention import prepare_qkv
from .fused_shared_av import stage
from . import reuse_global_history as history
from .reuse_probe_cache import observe
from saf3r.triton import reuse_light_states as rule

@dataclass(frozen=True)
class LightweightConfig:
 mode:str='full'
 retention:float=.2
 clustering_mode:str='adaptive_local'
 cluster_neighbors:int=32
 cluster_fraction:float|None=None
 source_width:int=32
 local_iterations:int=2
 grid_y:int=2
 grid_x:int=2
 window_groups:int=64
 dual_iterations:int=16
 attention_parts:int=4
 attention_warps:int=4
 selection_graph:bool=False
 probe_count:int=4
 graph_capacity:int=8
 attention_block_keys:int=64
 exact_filter:str='all'
 denominator:str='partition'
 anchor_mode:str='partition'
 anchor_count:int=256
 budget_graph:bool=True
 budget_repair:bool=False
 budget_repair_passes:int=2
 fused_coarse:bool=True
 cache_probe_indices:bool=True
 capture:bool=False
 def __post_init__(self):
  if self.mode not in ('simple','full') or self.denominator not in ('partition','global'):raise ValueError('Unknown independent model variant')
  if not 0<self.retention<=.2 or self.attention_parts not in (1,2,4,8):raise ValueError('Invalid retention or execution partition')
  if self.probe_count not in (1,4) or (self.mode=='simple' and self.probe_count!=4):raise ValueError('Single probe requires lightweight execution')
  if self.exact_filter not in ('legacy','all') or (self.mode=='full' and self.exact_filter!='all'):raise ValueError('Grid-free execution requires all-cluster Exact eligibility')
  if self.anchor_mode not in ('partition','sampled') or (self.anchor_mode=='sampled' and (self.denominator!='global' or self.mode!='full')):raise ValueError('Sampled anchors require global denominator/full execution')
  if not self.cache_probe_indices or self.capture:raise ValueError('Shape probe caching required; use diagnostic_hook for scoped captures')
  if self.budget_repair and self.selection_graph:raise ValueError('Budget repair currently requires eager selection')
  if self.budget_repair_passes not in (1,2):raise ValueError('Budget repair passes must be 1 or 2')

class LightweightState(ReuseExploreState):
 def reset(self,frames,height,width,device):
  super().reset(frames,height,width,device)
  if self.config.anchor_mode=='sampled':
   self.regions=min(self.config.anchor_count,len(self.patches))
   self.anchor_ids=self.patches[(torch.arange(self.regions,device=device)*len(self.patches)//self.regions)]
   self.grid_sizes=torch.full((self.regions,),len(self.patches)/self.regions,device=device)
   self.grid_log_sizes=self.grid_sizes.log()
   self.grid=None
 def prepare(self,k,features):
  f=self.fine.base_count;population=torch.bincount(self.fine.labels,minlength=f).float()
  if self.config.anchor_mode=='sampled':pooled=k[:,self.anchor_ids].float().contiguous()
  else:
   pooled=torch.zeros((len(k),self.regions,k.shape[-1]),device=k.device)
   pooled.index_add_(1,self.grid,k[:,self.patches].float());pooled/=self.grid_sizes[None,:,None]
  self.history_pooled_fp32=pooled.contiguous();overlap=None
  if self.config.exact_filter=='legacy' and self.previous_layer>=0:
   pair=self.fine.labels*self.regions+self.grid;unique,count=torch.unique(pair,sorted=True,return_counts=True)
   overlap=torch.sparse_coo_tensor(torch.stack((unique//self.regions,unique%self.regions)),count.float()/self.grid_sizes[unique%self.regions],size=(f,self.regions),device=k.device).coalesce()
  return population,pooled.to(k.dtype).contiguous(),overlap

class LightweightAttention(nn.Module):
 def __init__(self,original,state,layer):
  super().__init__();self.original=original;self.state=state;self.layer=layer
 def forward(self,x,pos=None):
  o,s=self.original,self.state;c=s.config
  if self.training or len(x)!=1 or self.layer!=s.previous_layer+1:raise ValueError('Ordered batch-one inference required')
  with stage(s,'light_qkv'):q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
  with stage(s,'light_clustering'),torch.autocast('cuda',enabled=False):
   features=v.mean(0,dtype=torch.float32);s.fine.update(features[s.patches],self.layer)
  with stage(s,'light_geometry'),torch.autocast('cuda',enabled=False):
   population,pooled,overlap=s.prepare(k,features[s.patches])
   rms=features[s.patches].square().sum(1).mean().clamp_min(1.e-12).sqrt()
   z=(features[s.patches]/rms).contiguous();zspecial=(features[s.specials]/rms).contiguous()
   geometry=rule.prepare(z,s.fine.labels,population) if self.layer else None
  if self.layer==0:history.ensure_cache(s,q)
  s.m5c_count_errors=[];s.m5c_seed_errors=[];s.m5c_prior_errors=[]
  output=torch.empty_like(v);next_ids=[];grids=[];real_grids=[];records=[]
  for start in range(0,s.groups,c.window_groups):
   g=min(c.window_groups,s.groups-start);first=start*32;end=min(s.n,(start+g)*32)
   with stage(s,'light_observation'),torch.autocast('cuda',enabled=False):obs=observe(s,q,k,start,g,o.scale)
   if self.layer==0:
    seed=torch.arange(s.budget,device=x.device)*len(population)//s.budget
    ids=s.fine.reps[seed][None].expand(g,-1).int().contiguous()
    weights=population[seed][None].expand(g,-1).contiguous();counts=torch.full((g,),s.budget,device=x.device,dtype=torch.int32)
    selected=None;bg=hp=errors=None
    rec={f'state_{i}_clusters':torch.full((g,),len(population) if i==1 else 0,device=x.device) for i in [0,1,5,10]}
   else:
    with stage(s,'light_prior'),torch.autocast('cuda',enabled=False):
     old=s.history_ids[start:start+g];bg,hp,aux=history.components(s,start,old,obs)
     special_prob=(obs['special_logits']-obs['coarse_logz']).exp()
     special_mean=history.mean_probes(special_prob,obs)
     o0=rule.moment(old,hp,bg,z)+special_mean@zspecial
    with stage(s,'light_exact_qualification'),torch.autocast('cuda',enabled=False):
     if c.exact_filter=='legacy':
      from saf3r.triton.reuse_coarse_observation import patch
      dot=obs['probe_q']@pooled[:,None].transpose(-1,-2)
      probability,demand=patch(dot,s.grid_log_sizes,s.previous_grid[start:start+g],s.previous_real_grid[start:start+g],o.scale)
      residual=torch.sparse.mm(overlap,demand.T).T
      candidate=residual.topk(max(1,math.ceil(len(population)*.25)),dim=1).indices
      allowed=torch.zeros_like(residual,dtype=torch.bool).scatter_(1,candidate,True)
     else:allowed=torch.ones((g,len(population)),device=x.device,dtype=torch.bool)
    if c.selection_graph:
     from .reuse_light_graph import select
     with stage(s,'light_selection_graph'),torch.autocast('cuda',enabled=False):
      ids,weights,counts,state_stats,errors,selected=select(s,self.layer,old,hp,bg,z,s.fine.reps,s.fine.labels,population,allowed,o0,geometry)
      rec={f'state_{state}_clusters':state_stats[i] for i,state in enumerate([0,1,5,10])}
    else:
     with stage(s,'light_state_loss'),torch.autocast('cuda',enabled=False):
      errors,costs,qualified=rule.states(old,hp,bg,z,s.fine.reps,s.fine.labels,population,allowed,o0,geometry)
     with stage(s,'light_budget'),torch.autocast('cuda',enabled=False):
      from saf3r.triton.reuse_budget_graph import budget as graph_budget
      from saf3r.triton.reuse_compact import budget
      selected,_=(graph_budget if c.budget_graph else budget)(errors,costs,errors[:,:,0].amax(1)+1.e-12,s.budget,c.dual_iterations)
      if c.budget_repair:
       from .reuse_budget_repair import repair_underfilled
       selected=repair_underfilled(errors,costs,selected,s.budget,passes=c.budget_repair_passes)
     with stage(s,'light_pack'),torch.autocast('cuda',enabled=False):
      ids,weights,counts=rule.pack(selected,costs,s.fine.reps,population,qualified,geometry,s.budget)
      rec={f'state_{state}_clusters':(selected==i).sum(1) for i,state in enumerate([0,1,5,10])}
   actual=s.patches[ids.long().clamp_min(0)].masked_fill(ids<0,-1).int().contiguous();bias=weights.log().contiguous()
   with stage(s,'light_attention'):
    if c.mode=='simple':
     from saf3r.triton.reuse_serial_tuned import weighted_attention
     out,mass,raw=weighted_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale,s.grid[ids.long().clamp_min(0)],s.regions,evidence_layout={},emit_evidence=self.layer<23)
     ev=dict(raw=raw,selected_lse=None,special_lse=None)
     patch_mass=mass[:,0].sum(1,keepdim=True).clamp_min(1.e-30)
     grids.append((mass[:,0]/patch_mass).half());real_grids.append((mass[:,1]/patch_mass).half())
    else:
     from saf3r.triton.reuse_light_attention import weighted_attention
     out,ev=weighted_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale,parts=c.attention_parts,block_keys=c.attention_block_keys,warps=c.attention_warps,probe_count=c.probe_count,emit_evidence=self.layer<23)
   output[:,first:end]=out
   if self.layer<23:
    with stage(s,'light_history_producer'),torch.autocast('cuda',enabled=False):
     probabilities,eta=history.produce(s,start,ids,obs,ev['raw'],ev['selected_lse'],ev['special_lse'],c.denominator)
   else:probabilities=eta=None
   hook=getattr(s,'diagnostic_hook',None)
   if hook is not None:hook(s,self.layer,start,q,k,v,ids,bias,obs,probabilities,eta,dict(bg=bg,historical=hp,errors=errors,selected=selected))
   next_ids.append(ids);rec['counts']=counts;records.append(rec)
  s.history_ids=torch.cat(next_ids);s.previous_layer=self.layer
  if grids:s.previous_grid=torch.cat(grids);s.previous_real_grid=torch.cat(real_grids)
  joined={name:torch.cat([r[name] for r in records]) for name in records[0]}
  bad=(joined['counts']>s.budget).any()
  if s.m5c_count_errors:bad=bad|torch.stack(s.m5c_count_errors).any()
  if s.m5c_seed_errors:bad=bad|(torch.stack(s.m5c_seed_errors).max()>1.e-5)
  if bool(bad):raise RuntimeError('Unique edge budget or global probability mass invalid')
  rows=(s.n-torch.arange(s.groups,device=x.device)*32).clamp_max(32)
  s.records.append(dict(layer=self.layer,patch_budget=s.budget,special_keys=len(s.specials),fine_clusters=len(population),canonical_regions=s.regions,
   **{name+'_mean':value.float().mean() for name,value in joined.items()},actual_unique_density=((joined['counts']+len(s.specials))*rows).sum()/s.n**2,
   modeling=dict(history='headmean_dense_probability_global_shrink',compensation='Rep population; History/Exact own contribution',loss='squared normalized-output contribution proxy',exact_eligibility=c.exact_filter),
   execution=dict(grid_sorted=c.mode=='simple',attention_parts=1 if c.mode=='simple' else c.attention_parts,extra_selected_qk=0,cluster_pairwise_distances=False,mass_transfer=False),
   seed_normalization_max_error=torch.stack(s.m5c_seed_errors).max() if s.m5c_seed_errors else torch.tensor(0.,device=x.device)))
  if self.layer==23:history.release(s)
  s.history_pooled_fp32=None
  return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))

def install_lightweight_reuse(model,config=None):
 config=LightweightConfig() if config is None else config
 if hasattr(model,'reuse_explore_state'):raise ValueError('Fresh unpatched model required')
 state=LightweightState(config)
 for layer,block in enumerate(model.aggregator.global_blocks):block.attn=LightweightAttention(block.attn,state,layer).eval()
 def reset(module,inputs):
  images=inputs[0]
  if images.ndim!=5 or len(images)!=1:raise ValueError('Batch-one sequence required')
  state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14,images.device)
 state.reset_handle=model.aggregator.register_forward_pre_hook(reset);model.reuse_explore_state=state
 return state
