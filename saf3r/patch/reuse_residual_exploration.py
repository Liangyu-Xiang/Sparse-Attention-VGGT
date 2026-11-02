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
        if self.hierarchical and (not self.compact or self.history_mode!='dense_m5c' or self.history_per_cluster!=4 or not self.mass_transfer or self.disable_population_compensation or self.attention_backend!='indexed'):raise ValueError('Hierarchy requires full M5c compact transfer with History Top-4')
        if self.parent_arity not in (2,4) or self.parent_passes not in (1,2) or self.parent_moment_mode not in ('exact','calibrated'):raise ValueError('Invalid parent operating-point configuration')
        if self.clustering_mode not in ('source_anchored','adaptive_local','adaptive_certified') or self.cluster_neighbors<1:raise ValueError('Invalid clustering policy')
        if self.history_mode=='dense_m5c' and (self.attention_backend!='indexed' or not self.compact):raise ValueError('M5c requires indexed compact inference')
        if self.cluster_fraction is not None and (not math.isfinite(self.cluster_fraction) or not 0<self.cluster_fraction<=1):raise ValueError('Invalid cluster fraction')
        if self.attention_backend not in ('indexed','shared_grid_guarded'):raise ValueError('Invalid attention backend')
        if self.attention_backend=='shared_grid_guarded' and (not self.compact or self.query_shift):raise ValueError('Guarded shared bank requires fixed-group compact mode')
        if min(self.refresh_interval,self.reference_dual_iterations,self.guard_groups)<1 or not math.isfinite(self.guard_rel_l2) or self.guard_rel_l2<=0:raise ValueError('Invalid guard configuration')
        if self.layer_retentions is not None and (len(self.layer_retentions)!=24 or any(not math.isfinite(r) or not .05<=r<=self.retention for r in self.layer_retentions)):raise ValueError('24 layer retentions required within5% and maximum retention')
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
        c=self.config;self.frames,self.height,self.width=frames,height,width
        self.patches,self.specials=token_layout(frames,height*width,5,device)
        self.n=frames*(height*width+5);self.groups=math.ceil(self.n/32)
        self.budget=math.floor(c.retention*self.n)-len(self.specials)
        if self.budget<frames:raise ValueError('Budget must accommodate one fine cluster per frame')
        cluster_count=self.budget if c.cluster_fraction is None else math.floor(c.cluster_fraction*len(self.patches))
        if cluster_count<self.budget:raise ValueError('Cluster count must cover the initial patch budget')
        if c.clustering_mode=='source_anchored':self.fine=FullCoverageValue(frames,height*width,cluster_count,device,c.source_width,c.local_iterations)
        else:
            from .adaptive_value_clustering import AdaptiveFullCoverageValue
            self.fine=AdaptiveFullCoverageValue(frames,height*width,cluster_count,device,c.source_width,c.local_iterations,mode='local' if c.clustering_mode=='adaptive_local' else 'certified',neighbors=c.cluster_neighbors)
        self.grid,self.regions=canonical_layout(frames,height,width,c.grid_y,c.grid_x,device)
        self.grid_sizes=torch.bincount(self.grid,minlength=self.regions).float()
        self.history_ids=None;self.history_weights=None;self.history_relative=None;self.previous_grid=None;self.previous_real_grid=None
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
        q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
        with stage(s,'value_update'),torch.autocast(x.device.type,enabled=False):
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
                geometry=compact.prepare_geometry(zvalues,s.fine.labels,s.fine.reps,population,l2=c.state_l2)
                from saf3r.triton.reuse_drop_moment import prepare as prepare_drop,drop_norm
                if geometry is not None:geometry['drop_bins']=prepare_drop(population)
            if self.layer and (c.state_l2 or c.direction_drop) and geometry is None:raise RuntimeError('Combined method requires compact geometry')
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
            from . import reuse_headmean_history as m5c
            if self.layer==0:m5c.ensure_cache(s,q)
            s.m5c_next={};s.m5c_count_errors=[];s.m5c_reconstruction_qk=0;s.m5c_prior_errors=[];s.m5c_seed_errors=[];s.m5c_probe_qk=0
        output=torch.empty_like(v);history=[];grids=[];real_grids=[];stats=[];capture=[];relative_evidence=[]
        candidates=max(1,math.ceil(s.fine.base_count*c.candidate_fraction))
        for start in range(0,s.groups,c.window_groups):
            g=min(c.window_groups,s.groups-start);first=start*32;end=min(s.n,(start+g)*32)
            if c.history_mode=='dense_m5c':
                with stage(s,'m5c_coarse_observation'),torch.autocast(x.device.type,enabled=False):
                    m5c_observation=m5c.observe(s,q,k,start,g,o.scale)
            if self.layer==0:
                seed=torch.arange(s.budget,device=x.device)*s.fine.base_count//s.budget
                ids=s.fine.reps[seed][None].expand(g,-1).contiguous()
                pops=population[seed][None].expand(g,-1).contiguous();real=torch.zeros_like(ids,dtype=torch.bool)
                record=dict(reuse_edges=torch.zeros(g,device=x.device),representatives=torch.full((g,),s.budget,device=x.device),
                    exact_edges=torch.zeros(g,device=x.device),dropped_clusters=torch.full((g,),s.fine.base_count-s.budget,device=x.device),
                    counts=torch.full((g,),s.budget,device=x.device),history_cap_discarded=torch.zeros(g,device=x.device))
            else:
                probe=(torch.arange(start,start+g,device=x.device)[:,None]*32+
                       torch.tensor([4,12,20,28],device=x.device)).clamp_max(s.n-1)
                pq=q[:,probe]
                with stage(s,'coarse_observation'):
                    z=(pq@pooled[:,None].transpose(-1,-2)).float()*o.scale+s.grid_sizes.log()
                    normalizer=z.logsumexp(-1,keepdim=True)
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
                    old_ids=None if s.history_ids is None else s.history_ids[start:start+g]
                    if c.history_mode=='dense_m5c':
                        prior,pi=m5c.build_prior(s,start,old_ids,m5c_observation)
                        history_context=history_rule.gate(prior,pi,population,old_ids,s.fine.labels,s.fine.reps)
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
                            errors,costs=history_rule.states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier,geometry,history_context)
                        else:errors,costs=cosine_states(pi,zvalues,s.fine.reps,s.fine.labels,population,hids,allowed,prior,c.drop_multiplier)
                    else:
                        errors,costs=operating_points(pi,pi_hat,means,features[s.fine.reps],population,
                                                    hids,hmass,features,allowed)
                    # Subtracting the Rep baseline is an exact gain reformulation.
                    # Keep the same feasible discrete dual solve (not a greedy exchange).
                    objective=geometry['objective'] if geometry is not None else (errors-errors[:,:,1:2] if c.rep_gain else errors)
                    if geometry is not None:
                        selected,count=compact.budget(objective,costs,errors[:,:,0].amax(1)+1e-12,s.budget,c.dual_iterations)
                        selected=history_rule.decode(selected)
                    else:
                        selected,count=budget_states(objective,costs,s.budget,c.dual_iterations,maximize=c.allocation_objective=="max_error",
                            lambda_upper=errors[:,:,0].amax(1)+1e-12 if c.rep_gain else None)
                    if s.parent_prepared is not None:
                        with stage(s,'parent_joint_allocation'):
                            selected,parent_owners,parent_steps=hierarchy.allocate(errors,costs,pi,zvalues,s.fine.reps,hids,prior,selected,s.parent_prepared,s.budget,c.dual_iterations,c.parent_passes,geometry,c.parent_moment_mode)
                    if geometry is not None:
                        ids,real,record=history_rule.pack(selected,hids,s.fine.reps,s.fine.labels,population,s.budget,history_context)
                    else:
                        ids,pops,real,record=pack_support(selected,hids,s.fine.reps,s.fine.labels,population,s.budget)
                    if c.state_loss=="cosine_distortion":
                        if c.mass_transfer:
                            if geometry is not None:pops=history_rule.transfer(selected,s.fine.labels,s.fine.reps,hids,prior,pi,ids,geometry,history_context)
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
                            record.update(history_rule.diagnostics(errors,selected,hids,old_ids,s.fine.labels,s.fine.reps,history_context))
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
            if c.disable_population_compensation:
                pops=(ids>=0).to(dtype=pops.dtype)
            bias=pops.log().contiguous()
            with stage(s,'execution_and_evidence'):
                if q.is_cuda:
                    from saf3r.triton.reuse_single_qk_grid import weighted_attention
                    reference=None
                    if c.history_mode=='all_relative_logit':
                        reference=(pooled,token_grid)
                    execution=weighted_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale,
                        key_grid=s.grid[ids.clamp_min(0)],regions=s.regions,emit_evidence=self.layer<23)
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
            if self.layer==23:m5c.release(s)
            s.history_pooled_fp32=None
            m5c.trim(s)
        if self.layer==0:
            for record in stats:
                for state_id in range(11):
                    record[f'state_{state_id}_clusters']=record['counts'] if state_id==1 else torch.zeros_like(record['counts'])
        s.previous_grid=torch.cat(grids);s.previous_real_grid=torch.cat(real_grids)
        s.previous_layer=self.layer
        joined={key:torch.cat([r[key] for r in stats]) for key in stats[0]}
        rows=(s.n-torch.arange(s.groups,device=x.device)*32).clamp_max(32)
        s.records.append(dict(layer=self.layer,query_offset=(self.layer*c.query_shift)%32,patch_budget=s.budget,special_keys=len(s.specials),
            fine_clusters=s.fine.base_count,canonical_regions=s.regions,
            **{key+'_mean':val.float().mean() for key,val in joined.items()},
            actual_unique_density=((joined['counts']+len(s.specials))*rows).sum()/s.n**2,
            unused_budget_mean=(s.budget-joined['counts']).float().mean(),
            canonical_qk_per_head=0 if self.layer==0 else 4*s.groups*s.regions,
            fine_probe_qk_per_head=0 if self.layer==0 or c.compact else 4*s.groups*candidates,
            compact=c.compact,geometry_width=0 if geometry is None else geometry['width'],
            geometry_cache_bytes=0 if geometry is None else sum(geometry[name].numel()*geometry[name].element_size() for name in ('members','rank','distances','rep_distance')),
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
                extra_previous_layer_probe_qk_per_head=s.m5c_reconstruction_qk,history_cpu_bytes=m5c.host_bytes(s),history_calibration_gpu_bytes=m5c.device_bytes(s),ready_probe_scratch_bytes=min(c.window_groups,s.groups)*q.shape[0]*4*s.budget*4 if self.layer<23 else 0,ready_edge_cache_bytes=0 if s.ready_edges is None else s.ready_edges.numel()*s.ready_edges.element_size(),ready_allocator_trims=s.ready_allocator_trims,
                seed_normalization_max_error=torch.stack(s.m5c_seed_errors).max() if s.m5c_seed_errors else torch.tensor(0.,device=x.device),
                prior_patch_mass_max_error=torch.stack(s.m5c_prior_errors).max() if s.m5c_prior_errors else torch.tensor(0.,device=x.device),
                extra_known_probe_qk_per_head=s.m5c_probe_qk,extra_coarse_probe_qk_per_head=4*s.groups*(s.regions+len(s.specials)))
        if c.capture:s.captured[self.layer]=dict(windows=capture,specials=s.specials.clone(),
            query_order=None if order is None else order.clone(),canonical=s.grid.clone(),fine_labels=s.fine.labels.clone(),fine_reps=s.fine.reps.clone())
        if order is not None:
            restored=torch.empty_like(output);restored[:,order]=output;output=restored
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))


def install_reuse_residual_exploration(model,config=None):
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
