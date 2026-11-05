from . import zdrop_history_cleanup as history_rule
"""Independent Reuse + residual drop/representative/exact prototype.

Reference allocator: a separable output-contribution proxy and Lagrange budget
search, NOT an exact knapsack optimizer or a Dense-output error oracle.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .fused_shared_av import FullCoverageValue,stage
from .region_history_attention import prepare_qkv
from .value_evolving_attention import token_layout


@dataclass(frozen=True)
class ReuseExploreConfig:
    attention_block_queries:int=32
    history_split_export:bool=False
    sampled_split_launch:bool=False
    deterministic_reductions:bool=False
    stable_binned_graph:bool=False
    static_shared_graph:bool=False
    graph_cache_max_mib:int=2048
    query_group_size:int=32
    source_layout:str="sequential"
    source_temporal_span:int=1
    dense_special_queries:int=0
    protect_first_frame:bool=False
    dense_first_frame_queries:bool=False
    history_query_interval:int=32
    flash_group_chunk: int = 16
    p0_loss_pack: bool = False
    shared_value_loss: bool = False
    shared_value_weighting: str = 'unit'
    shared_value_allow_observed_rep: bool = False
    shared_value_features: str = 'head_mean'
    state_loss_variant: str = 'legacy'
    state_loss_operator: str = 'group_history'
    state_loss_history_gate: str = 'observed'
    state_loss_sketch_dim: int = 8
    fused_coarse:bool=False
    graph_cache_capacity:int=24
    attention_block_keys:int=128
    reuse_window_graphs:bool=False
    attention_warps:int=4
    attention_stages:int=3
    fuse_state_kernel:bool=False
    cache_probe_indices:bool=False
    window_graph:bool=False
    execution_kernel:str="reference"
    fuse_mask_ops:bool=False
    budget_graph:bool=False
    history_evidence_backend:str="reference"
    history_cache_format:str="packed"
    retention:float=.2
    state_l2:bool=False
    direction_drop:bool=False
    clustering_mode:str='source_anchored'
    cluster_neighbors:int=32
    hierarchical:bool=False
    parent_arity:int=4
    parent_passes:int=2
    parent_moment_mode:str='exact'
    cluster_fraction:float|None=None
    disable_population_compensation:bool=False
    grid_y:int=2
    grid_x:int=2
    source_width:int=32
    local_iterations:int=2
    window_groups:int=8
    history_per_cluster:int=4
    candidate_fraction:float=.25
    dual_iterations:int=16
    capture:bool=False
    query_shift:int=0
    allocation_objective:str="min_error"
    evidence_mode:str="second_qk"
    state_loss:str="output_proxy"
    prior_floor_fraction:float=1e-6
    drop_multiplier:float=2.
    history_mode:str="genuine_probability"
    mass_transfer:bool=True
    rep_gain:bool=False
    compact:bool=False
    attention_backend:str="indexed"
    layer_retentions:list|None=None
    refresh_interval:int=1
    reference_dual_iterations:int=16
    guard_groups:int=4
    guard_rel_l2:float=.005

    def __post_init__(self):
        if self.sampled_split_launch and (self.query_group_size!=256 or self.history_query_interval!=256 or self.attention_block_queries not in (32,64,128) or self.history_evidence_backend!='direct_sampled'):raise ValueError('Split sampled execution requires Query-256, interval256 and supported tiles')
        if self.attention_block_queries not in (32,64,128) or self.query_group_size%self.attention_block_queries:raise ValueError('Attention tile must divide Query group size')
        if (self.static_shared_graph or self.stable_binned_graph) and not (self.shared_value_loss and self.reuse_window_graphs):raise ValueError('Static graph topology requires reusable Shared Value graphs')
        if self.history_split_export and self.history_evidence_backend!='direct_sampled':raise ValueError('Split export requires sampled history')
        if min(self.graph_cache_capacity,self.graph_cache_max_mib)<1:raise ValueError('Graph cache limits must be positive')
        if self.query_group_size not in (32,128,256):raise ValueError('Query group must be 32/128/256')
        if self.query_group_size!=32 and (not self.shared_value_loss or self.history_query_interval%self.query_group_size):raise ValueError('Larger query groups require shared Value sampled history and aligned interval')
        if self.source_layout not in ('sequential','spatiotemporal') or self.source_temporal_span<1:raise ValueError('Invalid source layout')
        if self.dense_special_queries not in (0,1,5):raise ValueError('Invalid dense special-query count')
        if (self.dense_special_queries or self.dense_first_frame_queries) and self.query_shift:raise ValueError('Dense query correction requires original query order')
        if self.attention_block_keys not in (64,128,256):raise ValueError('Invalid attention key tile')
        if self.state_loss_variant!='legacy':
            from .state_loss_controller import GROUP_VARIANTS,RESIDUAL_VARIANTS
            allowed=GROUP_VARIANTS if self.state_loss_operator=='group_history' else RESIDUAL_VARIANTS
            if self.state_loss_operator not in ('group_history','residual_rep') or self.state_loss_variant not in allowed:
                raise ValueError('Invalid state-loss/operator pair')
            if not self.shared_value_loss or not self.shared_value_allow_observed_rep or self.disable_population_compensation or self.shared_value_weighting=='unit':
                raise ValueError('Consistent state losses require shared path, enabled Rep and explicit mass calibration')
            if self.state_loss_history_gate not in ('native','observed') or self.state_loss_sketch_dim not in (4,8,16):
                raise ValueError('Invalid history gate or loss sketch dimension')
        if self.shared_value_features not in ('head_mean','head_concat'):
            raise ValueError('Invalid shared Value feature space')
        if self.shared_value_features!='head_mean' and not self.shared_value_loss:
            raise ValueError('Full-head Value features require shared_value_loss')
        if self.shared_value_weighting not in ('unit', 'rep_population', 'cluster_population', 'cluster_prior'):
            raise ValueError('Invalid shared Value execution weighting')
        if (self.shared_value_weighting != 'unit' or self.shared_value_allow_observed_rep) and not self.shared_value_loss:
            raise ValueError('Shared Value repair options require shared_value_loss')
        if self.shared_value_weighting != 'unit' and self.disable_population_compensation:
            raise ValueError('Shared Value weighting requires disable_population_compensation=false')
        if self.history_query_interval not in (32,256):raise ValueError('History Query interval must be32 or256')
        if self.shared_value_loss and (self.history_evidence_backend!='direct_sampled' or not self.window_graph or not self.fuse_state_kernel or not self.fuse_mask_ops or self.p0_loss_pack or self.hierarchical or not self.state_l2 or not self.direction_drop or not self.compact):raise ValueError('Shared Value loss requires the fused non-hierarchical sampled-history window path')
        if self.history_evidence_backend=='direct_sampled' and (self.execution_kernel!='direct_sparse' or self.history_mode!='dense_m5c' or not self.cache_probe_indices or self.window_groups%(self.history_query_interval//self.query_group_size)):raise ValueError('Sampled history requires aligned Direct Sparse windows and coarse probe cache')
        if self.history_evidence_backend=='direct_probability' and (self.execution_kernel!='direct_sparse' or self.history_mode!='dense_m5c' or not self.cache_probe_indices):raise ValueError('Probability history requires Direct Sparse, dense_m5c consumer and cached probe indices')
        if self.hierarchical and (not self.compact or self.history_mode!='dense_m5c' or self.history_per_cluster!=4 or not self.mass_transfer or self.disable_population_compensation or self.attention_backend!='indexed'):raise ValueError('Hierarchy requires full M5c compact transfer with History Top-4')
        if self.parent_arity not in (2,4) or self.parent_passes not in (1,2) or self.parent_moment_mode not in ('exact','calibrated'):raise ValueError('Invalid parent operating-point configuration')
        if self.clustering_mode not in ('source_anchored','adaptive_local','adaptive_certified','source_local_unanchored') or self.cluster_neighbors<1:raise ValueError('Invalid clustering policy')
        if self.history_mode=='dense_m5c' and (self.attention_backend!='indexed' or not self.compact):raise ValueError('M5c requires indexed compact inference')
        if self.cluster_fraction is not None and (not math.isfinite(self.cluster_fraction) or not 0<self.cluster_fraction<=1):raise ValueError('Invalid cluster fraction')
        if self.attention_backend not in ('indexed','shared_grid_guarded'):raise ValueError('Invalid attention backend')
        if self.attention_backend=='shared_grid_guarded' and (not self.compact or self.query_shift):raise ValueError('Guarded shared bank requires fixed-group compact mode')
        if min(self.refresh_interval,self.reference_dual_iterations,self.guard_groups)<1 or not math.isfinite(self.guard_rel_l2) or self.guard_rel_l2<=0:raise ValueError('Invalid guard configuration')
        from .layer_retention_schedule import validate_schedule
        validate_schedule(self.retention,self.layer_retentions)
        if self.compact and (self.state_loss!='cosine_distortion' or self.history_mode not in ('all_relative_logit','dense_m5c') or not self.mass_transfer or not self.rep_gain or self.allocation_objective!='min_error'):raise ValueError('Compact path requires the complete combined method')
        if self.history_mode not in ('genuine_probability','all_probability','all_relative_logit','dense_m5c'):raise ValueError('Invalid history mode')
        if self.drop_multiplier not in (1.,2.):raise ValueError('Invalid drop multiplier')
        if self.history_mode!='genuine_probability' and self.state_loss!='cosine_distortion':raise ValueError('Revised history requires cosine states')
        if self.history_mode in ('all_relative_logit','dense_m5c') and self.query_shift:raise ValueError('Relative history requires fixed query groups')
        if self.state_loss not in ('output_proxy','cosine_distortion'):raise ValueError('Invalid state loss')
        if not 0<self.prior_floor_fraction<=1e-3:raise ValueError('Invalid prior floor')
        if not 0<=self.query_shift<32:raise ValueError('Query shift must be 0..31')
        if self.allocation_objective not in ('min_error','max_error'):raise ValueError('Invalid allocation objective')
        if self.evidence_mode not in ('second_qk','saved_logits','fused_grid'):raise ValueError('Invalid evidence mode')
        if not math.isfinite(self.retention) or not .05<=self.retention<=.2:raise ValueError('Retention must be5–20%')
        if min(self.grid_y,self.grid_x,self.source_width,self.local_iterations,self.window_groups,self.dual_iterations)<1:
            raise ValueError('Positive grid/window/iteration sizes required')
        if not 1<=self.history_per_cluster<=8 or not 0<self.candidate_fraction<=1:
            raise ValueError('Invalid history cap/candidate fraction')


def canonical_layout(frames,height,width,gy,gx,device):
    gy=min(gy,height);gx=min(gx,width)
    spatial=(torch.arange(height,device=device)[:,None]*gy//height)*gx+torch.arange(width,device=device)[None]*gx//width
    labels=(spatial.flatten()[None]+torch.arange(frames,device=device)[:,None]*gy*gx).flatten()
    return labels,frames*gy*gx


def budget_states(errors,costs,budget,iterations=16,maximize=False,lambda_upper=None):
    """Independent region operating points; feasible <=B, no fixed split.

    Exact knapsack optimum is not claimed. Costs must be nonnegative; first
    state is drop with zero cost. Integer regions can leave unused budget.
    """
    if maximize:
        feasible=torch.isfinite(errors)
        ceiling=errors.masked_fill(~feasible,-torch.inf).amax(-1,keepdim=True)
        errors=(ceiling-errors).masked_fill(~feasible,torch.inf)
    lo=torch.zeros(len(errors),device=errors.device)
    hi=errors[:,:,0].amax(1)+1e-12 if lambda_upper is None else lambda_upper
    for _ in range(iterations):
        mid=(lo+hi)*.5
        selected=(errors+mid[:,None,None]*costs).argmin(-1)
        count=costs.gather(2,selected[:,:,None]).squeeze(-1).sum(1)
        over=count>budget
        lo=torch.where(over,mid,lo);hi=torch.where(over,hi,mid)
    selected=(errors+hi[:,None,None]*costs).argmin(-1)
    return selected,costs.gather(2,selected[:,:,None]).squeeze(-1).sum(1)


def top_history(ids,weights,beta,grid,labels,reps,limit):
    """Current-canonical stability times previous measured edge importance.

    Cap candidates per current fine cluster for bounded state enumeration.
    Current representative is excluded, avoiding duplicate reuse+rep supports.
    """
    g=beta.shape[0];f=len(reps);device=beta.device
    if ids is None:
        return (torch.full((g,f,limit),-1,device=device,dtype=torch.long),
                torch.zeros((g,f,limit),device=device),torch.zeros(g,device=device))
    valid=ids>=0;safe=ids.long().clamp_min(0);fine=labels[safe]
    valid &= safe!=reps[fine]
    u=weights.float()*beta.gather(1,grid[safe])
    valid &= u>0
    order=u.masked_fill(~valid,-1).argsort(dim=1,descending=True,stable=True)
    label=fine.gather(1,order).masked_fill(~valid.gather(1,order),f)
    group_order=label.argsort(dim=1,stable=True)
    order=order.gather(1,group_order);label=label.gather(1,group_order)
    pos=torch.arange(ids.shape[1],device=device)[None].expand(g,-1)
    first=torch.ones_like(label,dtype=torch.bool);first[:,1:]=label[:,1:]!=label[:,:-1]
    start=torch.where(first,pos,0).cummax(1).values
    rank=pos-start;keep=(label<f)&(rank<limit)
    slot=torch.where(keep,label*limit+rank,f*limit)
    out=torch.full((g,f*limit+1),-1,device=device,dtype=torch.long)
    mass=torch.zeros((g,f*limit+1),device=device)
    out.scatter_(1,slot,ids.long().gather(1,order).masked_fill(~keep,-1))
    mass.scatter_(1,slot,u.gather(1,order).masked_fill(~keep,0))
    discarded=valid.sum(1)-keep.sum(1)
    return out[:,:-1].reshape(g,f,limit),mass[:,:-1].reshape(g,f,limit),discarded


def operating_points(pi,pi_hat,means,rep_values,population,hids,hmass,features,exact_allowed):
    """Same vector-output units for drop, rep, reuse-prefix, reuse+rep, exact.

    Target pi*mean_V is a proxy for true attention-conditioned region output.
    pi_hat comes from current representative-probe QK, not assumed equal to pi.
    """
    g,f=pi.shape;limit=hids.shape[-1]
    target=pi[:,:,None]*means[None]
    total=hmass.sum(-1)
    hmass=hmass*torch.minimum(torch.ones_like(total),pi/total.clamp_min(1e-30))[:,:,None]
    hv=(features[hids.clamp_min(0)]*hmass[:,:,:,None]).cumsum(2)
    error=[target.norm(dim=-1),(target-pi_hat[:,:,None]*rep_values[None]).norm(dim=-1)]
    cost=[torch.zeros_like(pi),torch.ones_like(pi)]
    for t in range(1,limit+1):
        feasible=hids[:,:,t-1]>=0
        error.append((target-hv[:,:,t-1]).norm(dim=-1).masked_fill(~feasible,torch.inf))
        cost.append(torch.full_like(pi,t))
    for t in range(1,limit+1):
        feasible=(hids[:,:,t-1]>=0)&(population[None]>t)
        residual=pi_hat*(population[None]-t).clamp_min(0)/population.clamp_min(1)[None]
        estimate=hv[:,:,t-1]+residual[:,:,None]*rep_values[None]
        error.append((target-estimate).norm(dim=-1).masked_fill(~feasible,torch.inf))
        cost.append(torch.full_like(pi,t+1))
    error.append(torch.zeros_like(pi).masked_fill(~exact_allowed,torch.inf))
    cost.append(population[None].expand(g,-1).float())
    return torch.stack(error,-1),torch.stack(cost,-1)


def pack_support(selected,hids,reps,labels,population,budget):
    g,f=selected.shape;limit=hids.shape[-1];device=selected.device
    exact=selected==2*limit+2
    rep=(selected==1)|((selected>=limit+2)&(selected<=2*limit+1))
    t=torch.where((selected>=2)&(selected<=limit+1),selected-1,
                  torch.where((selected>=limit+2)&(selected<=2*limit+1),selected-limit-1,0))
    hvalid=(torch.arange(limit,device=device)[None,None]<t[:,:,None])&(hids>=0)
    hist=hids.masked_fill(~hvalid,-1).reshape(g,-1)
    rep_ids=reps[None].expand(g,-1).masked_fill(~rep,-1)
    member=torch.arange(len(labels),device=device)[None].expand(g,-1)
    member=member.masked_fill(~exact[:,labels],-1)
    raw=torch.cat((hist,rep_ids,member),1)
    weights=torch.cat((hvalid.reshape(g,-1).float(),
                       (population[None]-t).float()*rep,(member>=0).float()),1)
    genuine=torch.cat((hvalid.reshape(g,-1),torch.zeros_like(rep),(member>=0)),1)
    valid=raw>=0;pos=valid.long().cumsum(1)-1
    slots=pos.masked_fill(~valid,budget)
    out=torch.full((g,budget+1),-1,device=device,dtype=torch.long)
    pop=torch.zeros((g,budget+1),device=device)
    real=torch.zeros((g,budget+1),device=device,dtype=torch.bool)
    out.scatter_(1,slots,raw);pop.scatter_(1,slots,weights);real.scatter_(1,slots,genuine)
    return out[:,:-1].contiguous(),pop[:,:-1].contiguous(),real[:,:-1].contiguous(),dict(
        reuse_edges=hvalid.sum((1,2)),representatives=rep.sum(1),exact_edges=(member>=0).sum(1),
        dropped_clusters=(selected==0).sum(1),counts=valid.sum(1))


def reference_attention(q,k,v,ids,log_weights,specials,scale):
    output=torch.empty_like(q);mass=torch.zeros(ids.shape,device=q.device)
    for group,row in enumerate(ids):
        start=group*32;end=min(q.shape[1],start+32)
        all_ids=torch.cat((specials,row.long()));valid=all_ids>=0
        bias=torch.cat((torch.zeros(len(specials),device=q.device),log_weights[group]))
        z=q[:,start:end].float()@k[:,all_ids.clamp_min(0)].float().transpose(-1,-2)*scale+bias
        p=z.masked_fill(~valid[None,None],-torch.inf).softmax(-1)
        output[:,start:end]=(p@v[:,all_ids.clamp_min(0)].float()).to(q.dtype)
        mass[group]=p[:,:,len(specials):].mean((0,1))
    return output,mass


def regroup_history(state,order):
    """Overlap-weight previous group-mean proxies; union and coalesce Key IDs.

    Cyclic shifted groups overlap at most three old groups (including short tail).
    No probability per individual query is reconstructed or claimed here.
    """
    old_inverse=torch.empty_like(order)
    old_inverse[state.query_order]=torch.arange(state.n,device=order.device)
    old_group=old_inverse[order]//32
    padded=torch.nn.functional.pad(old_group,(0,state.groups*32-state.n),value=state.groups)
    matrix=padded.reshape(state.groups,32).sort(1).values
    starts=torch.ones_like(matrix,dtype=torch.bool);starts[:,1:]=matrix[:,1:]!=matrix[:,:-1]
    valid=matrix<state.groups;rank=(starts&valid).long().cumsum(1)-1
    # A new contiguous 32-query circular interval crosses <=3 old groups.
    source=torch.zeros((state.groups,4),device=order.device,dtype=torch.long)
    slots=torch.where(starts&valid,rank,3)
    source.scatter_(1,slots,matrix.clamp_max(state.groups-1))
    counts=torch.zeros((state.groups,4),device=order.device).scatter_add_(1,torch.where(valid,rank,3),valid.float())
    source=source[:,:3];fraction=counts[:,:3]/counts[:,:3].sum(1,keepdim=True)
    for name in ('previous_grid','previous_real_grid'):
        value=getattr(state,name)
        setattr(state,name,(value[source].float()*fraction[:,:,None]).sum(1).to(value.dtype))
    ids=state.history_ids[source].reshape(state.groups,-1)
    weights=(state.history_weights[source].float()*fraction[:,:,None]).reshape(state.groups,-1)
    # Sorting allows exact coalescing of the same historical Key from source groups.
    sentinel=len(state.patches)
    key,perm=ids.long().masked_fill(ids<0,sentinel).sort(dim=1,stable=True)
    weights=weights.gather(1,perm).masked_fill(key==sentinel,0)
    first=torch.ones_like(key,dtype=torch.bool);first[:,1:]=key[:,1:]!=key[:,:-1]
    segment=first.long().cumsum(1)-1
    total=torch.zeros_like(weights).scatter_add_(1,segment,weights)
    position=first.long().cumsum(1)-1
    # Keep the sorted IDs in place and attach each segment sum only to its first slot.
    weights=torch.where(first,total.gather(1,position),0)
    state.history_ids=key.masked_fill(~first|(key==sentinel),-1).int()
    state.history_weights=weights.half()


class ReuseExploreState:
    def __init__(self,config):self.config=config;self.profiling=False

    def reset(self,frames,height,width,device):
        self._forward_serial=getattr(self,'_forward_serial',0)+1
        self.graph_cache_creations=0;self.graph_cache_bytes=0;self.graph_cache_hits=0;self.graph_cache_misses=0;self.graph_cache_evictions=0;self.graph_constant_updates=0
        if hasattr(self,'_window_graphs'):
            self._window_graphs={};self._window_graph_layer=None
        c=self.config;self.frames,self.height,self.width=frames,height,width
        from .protected_frame_keys import protected_layout,check_protected_budget
        self.patches,self.specials,self.semantic_specials,cluster_frames=protected_layout(frames,height,width,device,c.protect_first_frame)
        self.n=frames*(height*width+5);self.groups=math.ceil(self.n/c.query_group_size)
        if c.protect_first_frame:check_protected_budget(c,self.n,len(self.specials),cluster_frames)
        self.budget=math.floor(c.retention*self.n)-len(self.specials)
        if self.budget<cluster_frames:raise ValueError('Budget must accommodate one fine cluster per active frame')
        cluster_count=self.budget if c.cluster_fraction is None else math.floor(c.cluster_fraction*len(self.patches))
        if not cluster_frames<=cluster_count<=len(self.patches):raise ValueError('Cluster count must be between active frames and patch count')
        if c.clustering_mode=='source_anchored':self.fine=FullCoverageValue(cluster_frames,height*width,cluster_count,device,c.source_width,c.local_iterations)
        elif c.clustering_mode=='source_local_unanchored':
            from .source_local_unanchored import SourceLocalUnanchoredValue
            self.fine=SourceLocalUnanchoredValue(cluster_frames,height,width,cluster_count,device,c.source_width,c.local_iterations,c.source_layout,c.source_temporal_span)
        else:
            from .adaptive_value_clustering import AdaptiveFullCoverageValue
            self.fine=AdaptiveFullCoverageValue(cluster_frames,height*width,cluster_count,device,c.source_width,c.local_iterations,mode='local' if c.clustering_mode=='adaptive_local' else 'certified',neighbors=c.cluster_neighbors)
        self.grid,self.regions=canonical_layout(cluster_frames,height,width,c.grid_y,c.grid_x,device)
        self.grid_sizes=torch.bincount(self.grid,minlength=self.regions).float()
        self.history_ids=None;self.history_weights=None;self.history_relative=None;self.previous_grid=None;self.previous_real_grid=None
        if c.cache_probe_indices:
            from .reuse_probe_cache import prepare
            prepare(self)
        self.query_order=torch.arange(self.n,device=device)
        self.ready_allocator_trims=0;self.ready_written={};self.previous_layer=-1;self.records=[];self.captured={};self.profile_events=[]
        self.m5c_previous_q=None;self.m5c_previous_k=None;self.m5c_previous_scale=None;self.m5c_reconstruction_qk=0
        self.m5c_previous={};self.m5c_next={};self.m5c_prior_errors=[];self.m5c_seed_errors=[];self.m5c_probe_qk=0

    def prepare(self,k,features):
        f=self.fine.base_count;r=self.regions
        population=torch.bincount(self.fine.labels,minlength=f).float()
        means=None  # Unread in this compact squared-L2 method.
        pooled=torch.zeros((len(k),r,k.shape[-1]),device=k.device)
        pooled.index_add_(1,self.grid,k[:,self.patches].float())
        pooled=pooled/self.grid_sizes[None,:,None]
        if self.config.history_mode=='dense_m5c':self.history_pooled_fp32=pooled.contiguous()
        pooled=pooled.to(k.dtype).contiguous()
        overlap=None
        if self.previous_layer>=0:
            pair=self.fine.labels*r+self.grid
            unique,count=torch.unique(pair,sorted=True,return_counts=True)
            overlap=torch.sparse_coo_tensor(torch.stack((unique//r,unique%r)),
                count.float()/self.grid_sizes[unique%r],size=(f,r),device=k.device).coalesce()
        return population,means,pooled,overlap


class ReuseResidualExploration(nn.Module):
    def __init__(self,original,state,layer):
        super().__init__();self.original,self.state,self.layer=original,state,layer

    def forward(self,x,pos=None):
        o,s=self.original,self.state;c=s.config
        if self.training or len(x)!=1 or self.layer!=s.previous_layer+1:raise ValueError('Batch-one ordered inference required')
        if c.capture:raise RuntimeError('Fused-grid capture exposes grid summaries, not per-edge masses; capture mode unsupported')
        if c.history_mode!='dense_m5c' or not c.state_l2 or not c.direction_drop or not c.compact or c.query_shift or c.hierarchical:raise RuntimeError('Cleanup supports only the frozen active method')
        rule=history_rule
        if c.fuse_mask_ops:
            from . import reuse_inference_ops as rule
        from .layer_retention_schedule import layer_budget
        s.budget=layer_budget(c,self.layer,s.n,len(s.specials))
        q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
        with stage(s,'value_update'),torch.autocast(x.device.type,enabled=False):
            if c.shared_value_features=='head_concat':
                features=v[:,s.patches].permute(1,0,2).reshape(len(s.patches),-1).float()/math.sqrt(len(v))
            else:
                features=v.mean(0,dtype=torch.float32)[s.patches]
            s.fine.update(features,self.layer)
        with stage(s,'pooling'),torch.autocast(x.device.type,enabled=False):
            population,means,pooled,overlap=s.prepare(k,features)
            if c.state_loss=="cosine_distortion":
                from .reuse_cosine_distortion import normalized_values,history_prior,cosine_states,transfer_weights,relative_logit_prior
                zvalues=(features.float()/features.float().square().sum(-1).mean().clamp_min(1e-12).sqrt()).contiguous() if self.layer else None
            geometry=None
            if self.layer and c.compact and q.is_cuda:
                from saf3r.triton import reuse_compact as compact
                if c.shared_value_loss:
                    if c.state_loss_variant!='legacy':
                        from .state_loss_controller import prepare_geometry as prepare_consistent_geometry
                        geometry=prepare_consistent_geometry(v,o,s.patches,s.specials,s.fine.labels,s.fine.reps,population,c)
                    else:
                        from saf3r.triton.reuse_shared_value_states import prepare_geometry as prepare_shared_geometry
                        geometry=prepare_shared_geometry(zvalues,s.fine.labels,s.fine.reps,population,min_width=64 if (c.static_shared_graph or c.stable_binned_graph) else 1)
                else:
                    geometry=compact.prepare_geometry(zvalues,s.fine.labels,s.fine.reps,population,l2=c.state_l2)
                from saf3r.triton.reuse_drop_moment import prepare as prepare_drop,drop_norm
                if geometry is not None:
                    if c.shared_value_loss and c.static_shared_graph:
                        # Every cluster executes once; masked unused Grid cells contribute zero.
                        # Fixed topology and buffers do not depend on population values.
                        geometry['drop_bins']=[(geometry['width'],torch.arange(len(population),device=population.device,dtype=torch.int64))]
                    else:geometry['drop_bins']=prepare_drop(population)
            if self.layer and (c.state_l2 or c.direction_drop) and geometry is None:raise RuntimeError('Combined method requires compact geometry')
            if geometry is not None:geometry['fuse_state_kernel']=c.fuse_state_kernel
            if c.shared_value_loss and self.layer and c.state_loss_variant=='legacy':
                from saf3r.triton.reuse_shared_value_states import prepare as prepare_shared_value
                prepare_shared_value(zvalues,s.fine.labels,s.fine.reps,s.grid,s.regions,geometry)
                if c.stable_binned_graph:
                    # Preserve each original population bin's M/C reduction layout.
                    # Only buffer capacities and empty launch slots become fixed.
                    bins=dict(geometry['drop_bins'])
                    geometry['drop_bins']=[(size,bins.get(size,torch.empty(0,device=x.device,dtype=torch.int32)))
                        for size in (1,2,4,8,16,32,64,128,256,512) if size<=geometry['width']]
                    geometry['bounded_bins']=True
                geometry['shared_value_weighting']=c.shared_value_weighting
                geometry['shared_value_allow_observed_rep']=c.shared_value_allow_observed_rep
            token_grid=None
            if c.history_mode=='all_relative_logit':
                token_grid=torch.zeros(s.n,device=x.device,dtype=torch.long)
                token_grid[s.patches]=s.grid
        s.parent_prepared=None
        if c.hierarchical and self.layer:
            from . import hierarchical_reuse as hierarchy
            with stage(s,'parent_structure_prepare'),torch.autocast(x.device.type,enabled=False):
                if bool(((zvalues.norm(dim=-1)-1).abs()<1e-3).all()):
                    s.parent_prepared=hierarchy.prepare(zvalues,s.fine.reps,s.fine.labels,geometry,c.parent_arity)
        order=None
        if c.query_shift:
            order=(torch.arange(s.n,device=x.device)+self.layer*c.query_shift%32)%s.n
            if self.layer:regroup_history(s,order)
            s.query_order=order
            q=q[:,order].contiguous()
        if c.history_mode=='dense_m5c':
            if c.history_evidence_backend=='direct_sampled':
                from . import reuse_sampled_history as m5c
            elif c.history_evidence_backend=='direct_probability':
                from . import reuse_probability_history as m5c
            elif c.history_evidence_backend=='direct_legacy':
                from . import reuse_direct_history as m5c
            elif c.history_evidence_backend=='grid_fused':
                from . import reuse_grid_fused_history as m5c
            elif c.history_evidence_backend=='grid_local':
                from . import reuse_grid_local_history as m5c
            elif c.history_evidence_backend=='reference':
                from . import reuse_headmean_history as m5c
            else:raise ValueError('Unknown history evidence backend')
            if self.layer==0:m5c.ensure_cache(s,q)
            if c.history_evidence_backend=='direct_sampled':
                from .layer_retention_schedule import begin_history_layer
                begin_history_layer(s,q)
            s.m5c_next={};s.m5c_count_errors=[];s.m5c_reconstruction_qk=0;s.m5c_prior_errors=[];s.m5c_seed_errors=[];s.m5c_probe_qk=0
        s.sampled_scratch_peak=0
        output=torch.empty_like(v);history=[];grids=[];real_grids=[];stats=[];capture=[];relative_evidence=[]
        if self.layer==0:
            from .reuse_cluster_seed import prepare_seed
            with stage(s,'cluster_seed_prepare'):
                seed_ids,seed_pops,seed_real=prepare_seed(s.fine.reps,s.fine.labels,population,s.budget)
        candidates=max(1,math.ceil(s.fine.base_count*c.candidate_fraction))
        for start in range(0,s.groups,c.window_groups):
            g=min(c.window_groups,s.groups-start);first=start*c.query_group_size;end=min(s.n,(start+g)*c.query_group_size)
            if c.history_mode=='dense_m5c':
                with stage(s,'m5c_coarse_observation'),torch.autocast(x.device.type,enabled=False):
                    observer=m5c.observe
                    if c.cache_probe_indices and c.history_evidence_backend not in ('direct_legacy','direct_probability','direct_sampled'):
                        from .reuse_probe_cache import observe as observer
                    m5c_observation=observer(s,q,k,start,g,o.scale)
            if self.layer==0:
                ids=seed_ids[None].expand(g,-1).contiguous()
                pops=seed_pops[None].expand(g,-1).contiguous();real=seed_real[None].expand(g,-1).contiguous()
                record=dict(reuse_edges=torch.zeros(g,device=x.device),representatives=torch.full((g,),min(s.budget,s.fine.base_count),device=x.device),
                    exact_edges=torch.full((g,),max(0,s.budget-s.fine.base_count),device=x.device),dropped_clusters=torch.full((g,),max(0,s.fine.base_count-s.budget),device=x.device),
                    counts=torch.full((g,),s.budget,device=x.device),history_cap_discarded=torch.zeros(g,device=x.device))
            else:
                if c.cache_probe_indices:
                    probe=m5c_observation['positions'];pq=m5c_observation['probe_q']
                else:
                    probe=(torch.arange(start,start+g,device=x.device)[:,None]*32+
                           torch.tensor([4,12,20,28],device=x.device)).clamp_max(s.n-1)
                    pq=q[:,probe]
                with stage(s,'coarse_observation'):
                    if c.history_evidence_backend in ('direct_legacy','direct_probability','direct_sampled'):
                        probability=m5c_observation['patch_probability']
                        previous=s.previous_grid[start:start+g].float()
                        beta=torch.where(previous>1e-12,(probability/previous.clamp_min(1e-12)).clamp_max(1),0)
                        demand=(probability-beta*s.previous_real_grid[start:start+g].float()).clamp_min(0)
                        normalizer=None
                    elif c.fused_coarse:
                        from saf3r.triton.reuse_coarse_observation import patch
                        dot=pq@pooled[:,None].transpose(-1,-2)
                        probability,demand=patch(dot,s.grid_log_sizes,s.previous_grid[start:start+g],s.previous_real_grid[start:start+g],o.scale)
                        normalizer=None
                    else:
                        z=(pq@pooled[:,None].transpose(-1,-2)).float()*o.scale+(s.grid_log_sizes if c.cache_probe_indices else s.grid_sizes.log())
                        normalizer=z.logsumexp(-1,keepdim=True) if not c.compact or not c.cache_probe_indices else None
                        probability=z.softmax(-1).mean((0,2))
                        previous=s.previous_grid[start:start+g].float()
                        beta=torch.where(previous>1e-12,(probability/previous.clamp_min(1e-12)).clamp_max(1),0)
                        demand=(probability-beta*s.previous_real_grid[start:start+g].float()).clamp_min(0)
                with stage(s,'residual_fine_probe'),torch.autocast(x.device.type,enabled=False):
                    residual=torch.sparse.mm(overlap,demand.T).T
                    candidate=residual.topk(candidates,dim=1).indices
                    allowed=torch.zeros_like(residual,dtype=torch.bool).scatter_(1,candidate,True)
                    if not c.compact:
                        ck=k[:,s.patches[s.fine.reps[candidate]]]
                        logits=(pq@ck.transpose(-1,-2)).float()*o.scale+population[candidate].log()[None,:,None]
                        fine_mass=(logits-normalizer).exp().mean((0,2))
                        pi_hat=pi.clone().scatter_(1,candidate,fine_mass)
                        pi_hat/=pi_hat.sum(1,keepdim=True).clamp_min(1e-30)
                with stage(s,'reuse_and_budget'),torch.autocast(x.device.type,enabled=False):
                    if c.window_graph:
                        from .reuse_window_graph import select
                        old_ids=s.history_ids[start:start+g]
                        if c.shared_value_loss:
                            prior,pi,uniform=m5c.build_prior(s,start,old_ids,m5c_observation,return_uniform=True)
                        else:
                            prior,pi=m5c.build_prior(s,start,old_ids,m5c_observation)
                            uniform=None
                        extra={}
                        if c.state_loss_variant!='legacy':
                            obs=m5c_observation
                            extra['special_p']=(obs['special_logits'][:,:,:1,:]-obs['coarse_logz'][:,:,:1,:]).exp().mean((1,2)).contiguous()
                        ids,real,pops,record=select(s,self.layer,prior,pi,old_ids,allowed,zvalues,s.fine.reps,s.fine.labels,population,geometry,uniform=uniform,**extra)
                    else:
                        old_ids=None if s.history_ids is None else s.history_ids[start:start+g]
                        if c.history_mode=='dense_m5c':
                            prior,pi=m5c.build_prior(s,start,old_ids,m5c_observation)
                            history_context=rule.gate(prior,pi,population,old_ids,s.fine.labels,s.fine.reps)
                        elif c.history_mode=='all_relative_logit':
                            prior,pi=relative_logit_prior(probability,s.grid,s.grid_sizes,old_ids,
                                s.history_relative[start:start+g],s.fine.labels,s.fine.base_count)
                        else:history_beta=beta
                        hids=None;discarded=torch.zeros(g,device=x.device)
                        if c.state_loss=="cosine_distortion":
                            if c.history_mode not in ('all_relative_logit','dense_m5c'):
                                prior=history_prior(pi,s.fine.labels,population,old_ids,old_weights,beta,s.grid,c.prior_floor_fraction)
                            if geometry is not None:
                                if c.direction_drop:
                                    vec=zvalues  # shared layer-RMS-scaled Value for Drop and L2 states
                                    geometry['drop_values']=drop_norm(prior,vec,geometry,s.fine.base_count,geometry['drop_bins'])
                                errors,costs=rule.states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier,geometry,history_context)
                            else:errors,costs=cosine_states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier)
                        else:
                            errors,costs=operating_points(pi,pi_hat,means,features[s.fine.reps],population,
                                                        hids,hmass,features,allowed)
                        # Subtracting the Rep baseline is an exact gain reformulation.
                        # Keep the same feasible discrete dual solve (not a greedy exchange).
                        objective=geometry['objective'] if geometry is not None else (errors-errors[:,:,1:2] if c.rep_gain else errors)
                        if geometry is not None:
                            solver=compact.budget
                            if c.budget_graph:
                                from saf3r.triton.reuse_budget_graph import budget as solver
                            selected,count=solver(objective,costs,errors[:,:,0].amax(1)+1e-12,s.budget,c.dual_iterations)
                            selected=rule.decode(selected)
                        else:
                            selected,count=budget_states(objective,costs,s.budget,c.dual_iterations,maximize=c.allocation_objective=="max_error",
                                lambda_upper=errors[:,:,0].amax(1)+1e-12 if c.rep_gain else None)
                        if s.parent_prepared is not None:
                            with stage(s,'parent_joint_allocation'):
                                selected,parent_owners,parent_steps=hierarchy.allocate(errors,costs,pi,zvalues,s.fine.reps,hids,prior,selected,s.parent_prepared,s.budget,c.dual_iterations,c.parent_passes,geometry,c.parent_moment_mode)
                        if geometry is not None:
                            ids,real,record=rule.pack(selected,hids,s.fine.reps,s.fine.labels,population,s.budget,history_context)
                        else:
                            ids,pops,real,record=pack_support(selected,hids,s.fine.reps,s.fine.labels,population,s.budget)
                        if c.state_loss=="cosine_distortion":
                            if c.mass_transfer:
                                if geometry is not None:pops=rule.transfer(selected,s.fine.labels,s.fine.reps,hids,prior,pi,ids,geometry,history_context)
                                else:pops,_=transfer_weights(selected,zvalues,s.fine.reps,s.fine.labels,hids,prior,ids)
                            if s.parent_prepared is not None:
                                with stage(s,'parent_cross_cluster_transfer'):
                                    pops,parent_record=hierarchy.transfer(pops,ids,prior,pi,parent_owners,hids)
                                    record.update(parent_record)
                                    record['parent_proxy_loss']=parent_steps[-1]['proxy_loss']
                                    record['parent_baseline_proxy_loss']=parent_steps[0]['baseline_proxy_loss']
                                    record['parent_allocation_fallbacks']=torch.stack([step['fallback'] for step in parent_steps]).sum(0)
                                    record['truly_dropped_clusters']=(selected==0).sum(1)-parent_record['delegated_clusters']
                            limit=c.history_per_cluster
                            if geometry is not None:
                                record.update(rule.diagnostics(errors,selected,hids,old_ids,s.fine.labels,s.fine.reps,history_context))
                            else:
                                reused=torch.where((selected>=2)&(selected<=limit+1),selected-1,
                                    torch.where((selected>=limit+2)&(selected<=2*limit+1),selected-limit-1,0))
                                for amount in range(limit+1):record[f'selected_history_count_{amount}_clusters']=(reused==amount).sum(1)
                        if c.rep_gain and geometry is None:
                            observed=(old_ids>=0)
                            overlap_rep=observed&(old_ids.long().clamp_min(0)==s.fine.reps[s.fine.labels[old_ids.long().clamp_min(0)]])
                            gains=errors[:,:,1:2]-errors
                            finite=torch.isfinite(gains)
                            record.update(observed_history=observed.sum(1),history_rep_overlap=overlap_rep.sum(1),
                                history_candidates=(hids>=0).sum((1,2)),
                                positive_gain_states=((gains>0)&finite).sum((1,2)),
                                positive_gain_sum=gains.masked_fill(~finite,0).clamp_min(0).sum((1,2)),
                                negative_gain_sum=gains.masked_fill(~finite,0).clamp_max(0).sum((1,2)),
                                selected_gain=gains.gather(2,selected[:,:,None]).sum((1,2)),
                                gain_min=gains.masked_fill(~finite,torch.inf).amin((1,2)),
                                gain_max=gains.masked_fill(~finite,-torch.inf).amax((1,2)))
                            for state_id in range(gains.shape[-1]):
                                present=finite[:,:,state_id]
                                record[f'state_{state_id}_clusters']=(selected==state_id).sum(1)
                                record[f'gain_{state_id}']=gains[:,:,state_id].masked_fill(~present,0).sum(1)/present.sum(1).clamp_min(1)
                        record['history_cap_discarded']=torch.zeros_like(discarded)
                        if geometry is None:
                            record['proxy_drop_error']=(errors[:,:,0]).sum(1)
                            record['proxy_selected_error']=errors.gather(2,selected[:,:,None]).sum((1,2))
            actual=s.patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int().contiguous()
            if c.disable_population_compensation or (c.shared_value_loss and c.shared_value_weighting=='unit'):
                pops=(ids>=0).to(dtype=pops.dtype)
            bias=pops.log().contiguous()
            with stage(s,'execution_and_evidence'):
                if q.is_cuda and c.execution_kernel=='direct_sparse':
                    if c.history_evidence_backend=='direct_sampled':
                        from saf3r.triton.reuse_direct_sampled_probability import direct_sparse_attention_sampled
                        result,direct_evidence=direct_sparse_attention_sampled(q[:,first:end],k,v,actual,s.specials,o.scale,bias,
                            probe_positions=(m5c_observation['positions'][:,:1]-first).int().contiguous(),probe_count=1,
                            probe_group_stride=c.history_query_interval//c.query_group_size,query_group_size=c.query_group_size,debiased_log=True,scope=getattr(s,'direct_scope',None),
                            block_queries=c.attention_block_queries,block_keys=c.attention_block_keys,
                            num_warps=c.attention_warps if (c.attention_block_queries!=32 or c.history_split_export) else 4,
                            num_stages=c.attention_stages if (c.attention_block_queries!=32 or c.history_split_export) else 2,
                            split_export=c.history_split_export,execution_split=c.sampled_split_launch)
                        s.sampled_scratch_peak=max(s.sampled_scratch_peak,direct_evidence['scratch_bytes']);s.sampled_compiler_resources=direct_evidence['compiler_resources']
                    else:
                        from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
                        probability_history=c.history_evidence_backend=='direct_probability'
                        result,direct_evidence=direct_sparse_attention(q[:,first:end],k,v,actual,s.specials,o.scale,logit_bias=bias,
                            probe_positions=(m5c_observation['positions']-first).int().contiguous() if probability_history else None,
                            return_history_probability=probability_history,history_export='debiased_log_lse_probe' if probability_history else 'lse_probe',scope=getattr(s,'direct_scope',None))
                    ready_logits=None
                elif q.is_cuda:
                    if c.execution_kernel=='serial_tuned':
                        from saf3r.triton.reuse_serial_tuned import weighted_attention
                        from functools import partial
                        weighted_attention=partial(weighted_attention,stages=c.attention_stages)
                    elif c.execution_kernel=='torch_flash':
                        from saf3r.triton.reuse_torch_flash import weighted_attention
                        from functools import partial
                        weighted_attention=partial(weighted_attention,group_chunk=c.flash_group_chunk,scope=getattr(s,'flash_scope',None))
                    elif c.execution_kernel=='split_fused':
                        from saf3r.triton.reuse_split_fused_history import weighted_attention
                        from functools import partial
                        weighted_attention=partial(weighted_attention,warps=c.attention_warps,stages=c.attention_stages,block_keys=c.attention_block_keys)
                    elif c.execution_kernel=='split':
                        from saf3r.triton.reuse_split_grid_attention import weighted_attention
                    elif c.execution_kernel=='packed':
                        from saf3r.triton.reuse_packed_grid_attention import weighted_attention
                    else:
                        from saf3r.triton.reuse_single_qk_grid import weighted_attention
                    reference=None
                    if c.history_mode=='all_relative_logit':
                        reference=(pooled,token_grid)
                    execution=weighted_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale,
                        key_grid=s.grid[ids.clamp_min(0)],regions=s.regions,emit_evidence=self.layer<23,evidence_layout=m5c_observation if c.history_evidence_backend in ('grid_local','grid_fused') else None)
                    if reference is None:result,mass,ready_logits=execution;del execution
                    else:
                        result,mass,delta=execution
                        relative_evidence.append(delta)
                else:
                    result,mass=reference_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale)
                    if c.history_mode=='all_relative_logit':
                        safe=actual.long().clamp_min(0)
                        raw=(q[:,first:end].float().mean(1)[:,None,:]*k[:,safe].float()).sum(-1)*o.scale
                        ref=(q[:,first:end].float().mean(1)[:,None,None,:]*pooled[:,s.grid[ids.clamp_min(0)]].float()).sum(-1)*o.scale
                        relative_evidence.append((raw-ref).mean(0).masked_fill(ids<0,0))
            if c.execution_kernel=='direct_sparse':
                with stage(s,'sampled_probe_grid_summary' if c.history_evidence_backend=='direct_sampled' else 'sparse_probe_grid_summary' if c.history_evidence_backend=='direct_probability' else 'direct_independent_legacy_observation'):
                    mass=m5c.observe_execution(s,q,k,start,ids,actual,bias,m5c_observation,o.scale,direct_evidence)
            output[:,first:end]=result
            if c.history_mode=='dense_m5c' and self.layer<23:
                m5c_observation['ready_logits']=ready_logits
                with stage(s,'m5c_evidence_and_host_copy'),torch.autocast(x.device.type,enabled=False):
                    m5c.save_evidence(s,q,k,start,ids,actual,m5c_observation,o.scale)
                del ready_logits,m5c_observation['ready_logits']
            with stage(s,'history_summary'):
                patch_mass=mass[:,0].sum(1,keepdim=True).clamp_min(1e-30)
                grid=mass[:,0]/patch_mass
                real_grid=mass[:,1]/patch_mass
                real=ids>=0
                history.append(ids.masked_fill(~real,-1).int());pass  # This legacy half-precision history bank is unread by D2+M5c.
                grids.append(grid.half());real_grids.append(real_grid.half())
            stats.append(record)
            if c.capture:capture.append(dict(ids=actual.clone(),population=pops.clone(),genuine=real.clone(),
                                             mass=mass.clone(),reuse=record['reuse_edges'].clone()))
        if c.fuse_mask_ops and self.layer:
            if bool(torch.cat([r['implementation_validation_error'] for r in stats]).any()):raise RuntimeError('Deferred support or mass conservation check failed')
        if c.history_mode=='dense_m5c':
            if s.m5c_count_errors and bool(torch.stack(s.m5c_count_errors).any()):raise ValueError('History count exceeds region population')
            s.m5c_reconstruction_qk=int(s.m5c_reconstruction_qk)
            s.m5c_probe_qk=int(s.m5c_probe_qk)
        s.history_ids=torch.cat(history);s.history_weights=None
        if relative_evidence:s.history_relative=torch.cat(relative_evidence)
        if c.history_mode=='dense_m5c':
            if s.m5c_previous:raise RuntimeError('Unconsumed M5c history windows')
            s.m5c_previous=s.m5c_next;s.m5c_next={}
            s.m5c_previous_q=None;s.m5c_previous_k=None;s.m5c_previous_scale=None
            if c.history_evidence_backend=='direct_sampled':
                from .layer_retention_schedule import finish_history_layer
                finish_history_layer(s)
            if self.layer==23:m5c.release(s)
            s.history_pooled_fp32=None
            m5c.trim(s)
        if self.layer==0:
            for record in stats:
                for state_id in range(11):
                    record[f'state_{state_id}_clusters']=torch.full_like(record['counts'],min(s.budget,s.fine.base_count)) if state_id==1 else torch.zeros_like(record['counts'])
        s.previous_grid=torch.cat(grids);s.previous_real_grid=torch.cat(real_grids)
        s.previous_layer=self.layer
        joined={key:torch.cat([r[key] for r in stats]) for key in stats[0]}
        rows=(s.n-torch.arange(s.groups,device=x.device)*c.query_group_size).clamp_max(c.query_group_size)
        s.records.append(dict(layer=self.layer,query_group_size=c.query_group_size,query_groups=s.groups,query_offset=(self.layer*c.query_shift)%32,patch_budget=s.budget,special_keys=len(s.specials),
            semantic_special_keys=len(s.semantic_specials),protected_patch_keys=len(s.specials)-len(s.semantic_specials),protect_first_frame=c.protect_first_frame,
            fine_clusters=s.fine.base_count,canonical_regions=s.regions,
            initial_seed_extra_members=max(0,s.budget-s.fine.base_count) if self.layer==0 else 0,
            **{key+'_mean':val.float().mean() for key,val in joined.items()},
            actual_unique_density=((joined['counts']+len(s.specials))*rows).sum()/s.n**2,
            unused_budget_mean=(s.budget-joined['counts']).float().mean(),
            canonical_qk_per_head=0 if self.layer==0 else 4*s.groups*s.regions,
            fine_probe_qk_per_head=0 if self.layer==0 or c.compact else 4*s.groups*candidates,
            compact=c.compact,geometry_width=0 if geometry is None else geometry['width'],
            geometry_cache_bytes=0 if geometry is None else sum(geometry[name].numel()*geometry[name].element_size() for name in ('members','rank','distances','rep_distance','norm2','shared_cells','shared_drop','shared_rep','shared_grid','loss_patch_values','loss_special_values') if name in geometry),
            final_unique_qk_per_head=((joined['counts']+len(s.specials))*rows).sum(),
            evidence_extra_qk_per_head=0,
            evidence_scratch_peak_bytes=min(c.window_groups*32,s.n)*q.shape[0]*s.budget*4 if c.evidence_mode=='saved_logits' else 0,
            stored_history_bytes=s.history_ids.numel()*4,
            stored_grid_bytes=s.previous_grid.numel()*4,
            population_compensation=not c.disable_population_compensation and (c.state_loss=="output_proxy" or not c.mass_transfer),head_aware=False,
            history_mode=c.history_mode,drop_multiplier=c.drop_multiplier,rep_gain=c.rep_gain,
            relative_logit_evidence_bytes=0 if s.history_relative is None else s.history_relative.numel()*4,
            state_loss=c.state_loss,mass_transfer="estimated assigned mass / own prior mass" if c.state_loss=="cosine_distortion" and c.mass_transfer else "original population weights",
            selector=('Rep-relative weighted cosine distortion, discrete dual budget search; not exact knapsack' if c.rep_gain else 'separable vector-output proxy, discrete dual budget search; not exact knapsack'),
            history=c.history_mode+'; compact relative logits FP32, probabilities FP16; no fixed retention ratio',
            evidence='single_qk_fused_grid',allocation_objective=c.allocation_objective,query_shift=c.query_shift))
        if c.clustering_mode!='source_anchored':s.records[-1]['clustering']=s.fine.records[-1]
        if c.hierarchical:
            s.records[-1].update(hierarchical=True,parent_arity=c.parent_arity,parent_passes=c.parent_passes,parent_moment_mode=c.parent_moment_mode,parent_active=s.parent_prepared is not None,
                parent_groups=0 if s.parent_prepared is None else len(s.parent_prepared['parents']),
                parent_fallback_reason=('L0 retains the native Rep seed' if not self.layer else 'non-unit or zero normalized Value; native flat allocation retained') if s.parent_prepared is None else None,
                state0_meaning='no locally emitted keys; delegated vs truly ignored clusters recorded separately',
                exact_parent_proxy='child Rep and most distant member, both real emitted Exact keys')

        if c.history_mode=='dense_m5c':
            s.records[-1].update(history_estimator='Head-mean conditionals + head-mean M5c',beta_down_exponent=1.,beta_up_exponent=2.,
                probability_normalization='all Keys including specials; no patch renormalization',
                history_cache_device='gpu_prepared_conditionals',history_cache_dtype='headmean_positive_fp16_u14_edges_float32_grid_statistics',
                history_estimator_head_aware=False,history='Previous conditional probabilities averaged over heads before caching; current grid probabilities averaged before confidence; M5c beta(1,2); no head matching',
                history_gpu_reconstruction_bytes=0,
                extra_previous_layer_probe_qk_per_head=s.m5c_reconstruction_qk,history_cpu_bytes=m5c.host_bytes(s),history_calibration_gpu_bytes=m5c.device_bytes(s),ready_probe_scratch_bytes=0 if c.execution_kernel=='direct_sparse' else (min(c.window_groups,s.groups)*q.shape[0]*4*s.budget*4 if self.layer<23 else 0),ready_edge_cache_bytes=0 if s.ready_edges is None else s.ready_edges.numel()*s.ready_edges.element_size(),ready_allocator_trims=s.ready_allocator_trims,
                seed_normalization_max_error=torch.stack(s.m5c_seed_errors).max() if s.m5c_seed_errors else torch.tensor(0.,device=x.device),
                prior_patch_mass_max_error=torch.stack(s.m5c_prior_errors).max() if s.m5c_prior_errors else torch.tensor(0.,device=x.device),
                extra_known_probe_qk_per_head=s.m5c_probe_qk,extra_coarse_probe_qk_per_head=4*s.groups*(s.regions+len(s.specials)))
        if c.execution_kernel=='direct_sparse':
            s.records[-1].update(direct_logical_main_qk_per_head=((joined['counts']+len(s.specials))*rows).sum(),direct_logical_legacy_observation_qk_per_head=(joined['counts']*rows).sum(),direct_logical_conditional_cache_qk_per_head=(joined['counts']*4).sum() if self.layer<23 else 0,extra_known_probe_qk_per_head=(joined['counts']*4).sum() if self.layer<23 else 0,grid_reordering_in_attention=False,legacy_observation_repeats_qk=True)
        if c.history_evidence_backend=='direct_probability':
            s.records[-1].update(history_estimator='Debiased log head-mean sparse P + coarse missing completion + M5c',
                history='Compensated sparse P divided by shared assignment bias in log space; ratio of head means, coarse missing-token completion; M5c beta(1,2)',
                history_cache_dtype='signed_fp16_log_conditionals_float32_grid_statistics',history_cache_device='bounded_gpu_with_optional_pinned_spill',
                evidence='final_sparse_LSE_plus_four_probe_QK',legacy_observation_repeats_qk=False,
                direct_logical_legacy_observation_qk_per_head=0,direct_logical_conditional_cache_qk_per_head=0,
                extra_known_probe_qk_per_head=((joined['counts']+len(s.specials))*4).sum(),
                evidence_extra_qk_per_head=((joined['counts']+len(s.specials))*4).sum(),
                ready_probe_scratch_bytes=min(c.window_groups,s.groups)*4*(s.budget+len(s.specials))*4,
                history_estimator_head_aware=False,history_probe_summary=True)
        if c.history_evidence_backend=='direct_sampled':
            s.records[-1].update(history_estimator='One-Probe debiased sparse history with own-Grid uniform fallback',
                history_query_interval=c.history_query_interval,history_sampled_groups=math.ceil(s.groups/(c.history_query_interval//c.query_group_size)),
                history_unsampled_groups=s.groups-math.ceil(s.groups/(c.history_query_interval//c.query_group_size)),
                history='Sampled groups:Probe0,M5c beta and missing-key completion;unsampled groups:own coarse-uniform prior,no neighbor support reuse',
                history_cache_dtype='signed_fp16_log_conditionals_one_probe_sampled_groups_only',history_cache_device='gpu_compact_sampled',
                evidence='inline_one_probe_no_extra_QK',legacy_observation_repeats_qk=False,
                direct_logical_legacy_observation_qk_per_head=0,direct_logical_conditional_cache_qk_per_head=0,
                extra_known_probe_qk_per_head=0,evidence_extra_qk_per_head=0,extra_previous_layer_probe_qk_per_head=0,
                ready_probe_scratch_bytes=s.sampled_scratch_peak,history_estimator_head_aware=False,
                sampled_compiler_resources=s.sampled_compiler_resources,coarse_observation_probes=4)
        if c.shared_value_loss:
            s.records[-1].update(shared_value_loss=True,shared_value_weighting=c.shared_value_weighting,
                shared_value_allow_observed_rep=c.shared_value_allow_observed_rep,
                shared_value_features=c.shared_value_features,
                population_compensation=c.shared_value_weighting in ('rep_population','cluster_population'),
                mass_transfer='none' if c.shared_value_weighting=='unit' else c.shared_value_weighting,
                selector='Shared Cluster-Grid squared-L2 distortion; four-state discrete dual budget; not exact knapsack')
        if c.state_loss_variant!='legacy':
            s.records[-1].update(state_loss_variant=c.state_loss_variant,state_loss_operator=c.state_loss_operator,
                state_loss_history_gate=c.state_loss_history_gate,state_loss_sketch_dim=c.state_loss_sketch_dim,
                probability_fallback='Uniform within current cluster for unobserved members; preserve sampled historical evidence',
                execution_weighting='Estimated assigned cluster probability / retained own probability; matches support roles',
                ordinary_history_represents_others=c.state_loss_operator=='group_history',
                selector='Support/weight-consistent state loss; discrete16-step budget,not exact knapsack')
        if c.capture:s.captured[self.layer]=dict(windows=capture,specials=s.specials.clone(),
            query_order=None if order is None else order.clone(),canonical=s.grid.clone(),fine_labels=s.fine.labels.clone(),fine_reps=s.fine.reps.clone())
        if order is not None:
            restored=torch.empty_like(output);restored[:,order]=output;output=restored
        if c.dense_special_queries or c.dense_first_frame_queries:
            from .dense_special_queries import correct_output
            with stage(s,'dense_special_query_correction'):
                output,dense_ids=correct_output(q,k,v,output,s.semantic_specials,c.dense_special_queries,o.scale,
                    anchor_tokens=s.height*s.width+5 if c.dense_first_frame_queries else 0)
            extra=(s.n-len(s.specials)-joined['counts'][dense_ids//c.query_group_size]).sum()
            record=s.records[-1]
            record['sparse_observer_unique_density']=record['actual_unique_density']
            record['actual_unique_density']=record['actual_unique_density']+extra/s.n**2
            record['dense_special_query_count']=len(dense_ids)
            record['dense_first_frame_queries']=c.dense_first_frame_queries
            record['dense_special_query_logical_qk_per_head']=len(dense_ids)*s.n
            record['final_unique_qk_per_head']=record['final_unique_qk_per_head']+extra
            record['direct_logical_main_qk_per_head']=record['direct_logical_main_qk_per_head']+len(dense_ids)*s.n
            record['dense_query_history_semantics']='Sparse observer probabilities remain the history stream; dense output correction is separate and both are timed'
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))


def install_reuse_residual_exploration(model,config=None):
    if config is not None and config.deterministic_reductions:
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    if any(hasattr(model,name) for name in ('value_region_av_state','fused_shared_av_state','attention_first_residual_value_state','reuse_explore_state')):
        raise ValueError('Install only on a fresh unpatched model')
    config=ReuseExploreConfig() if config is None else config
    if len(model.aggregator.global_blocks)!=24:raise ValueError('24 global layers required')
    if config.attention_backend=='shared_grid_guarded':
        from .reuse_guarded import GuardedState,GuardedReuse
        state=GuardedState(config);wrapper=GuardedReuse
    else:state=ReuseExploreState(config);wrapper=ReuseResidualExploration
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn=wrapper(block.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0]
        if images.ndim!=5 or len(images)!=1:raise ValueError('Batch-one sequences required')
        state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14,images.device)
    state.reset_handle=model.aggregator.register_forward_pre_hook(reset)
    model.reuse_explore_state=state
    return state
