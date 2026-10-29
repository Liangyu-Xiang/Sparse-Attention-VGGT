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
    representative_mode:str="nearest"
    evidence_mode:str="second_qk"

    def __post_init__(self):
        if not 0<=self.query_shift<32:raise ValueError('Query shift must be 0..31')
        if self.representative_mode not in ('nearest','farthest'):raise ValueError('Invalid representative mode')
        if self.evidence_mode not in ('second_qk','saved_logits'):raise ValueError('Invalid evidence mode')
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


def budget_states(errors,costs,budget,iterations=16):
    """Independent region operating points; feasible <=B, no fixed split.

    Exact knapsack optimum is not claimed. Costs must be nonnegative; first
    state is drop with zero cost. Integer regions can leave unused budget.
    """
    lo=torch.zeros(len(errors),device=errors.device)
    hi=errors[:,:,0].amax(1)+1e-12
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
        self.fine=FullCoverageValue(frames,height*width,self.budget,device,c.source_width,c.local_iterations)
        self.grid,self.regions=canonical_layout(frames,height,width,c.grid_y,c.grid_x,device)
        self.grid_sizes=torch.bincount(self.grid,minlength=self.regions).float()
        self.history_ids=None;self.history_weights=None;self.previous_grid=None;self.previous_real_grid=None
        self.query_order=torch.arange(self.n,device=device)
        self.previous_layer=-1;self.records=[];self.captured={};self.profile_events=[]

    def prepare(self,k,features):
        f=self.fine.base_count;r=self.regions
        population=torch.bincount(self.fine.labels,minlength=f).float()
        means=torch.zeros((f,features.shape[-1]),device=k.device).index_add_(0,self.fine.labels,features)
        means/=population[:,None]
        pooled=torch.zeros((len(k),r,k.shape[-1]),device=k.device)
        pooled.index_add_(1,self.grid,k[:,self.patches].float())
        pooled=(pooled/self.grid_sizes[None,:,None]).to(k.dtype).contiguous()
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
        q,k,v=prepare_qkv(o,x,pos);q=q.to(v.dtype).contiguous();k=k.to(v.dtype).contiguous()
        with stage(s,'value_update'),torch.autocast(x.device.type,enabled=False):
            features=v.mean(0,dtype=torch.float32)[s.patches]
            s.fine.update(features,self.layer)
            nearest_reps=s.fine.reps
            if c.representative_mode=="farthest":
                labels=s.fine.labels;count=s.fine.base_count
                means=torch.zeros((count,features.shape[-1]),device=x.device).index_add_(0,labels,features)
                means/=torch.bincount(labels,minlength=count)[:,None]
                distance=(features-means[labels]).square().sum(-1)
                maxima=torch.full((count,),-torch.inf,device=x.device).scatter_reduce_(0,labels,distance,reduce="amax",include_self=True)
                candidates=torch.where(distance==maxima[labels],torch.arange(len(labels),device=x.device),len(labels))
                s.fine.reps=torch.full((count,),len(labels),device=x.device,dtype=torch.long).scatter_reduce_(0,labels,candidates,reduce="amin",include_self=True)
        with stage(s,'pooling'),torch.autocast(x.device.type,enabled=False):
            population,means,pooled,overlap=s.prepare(k,features)
        order=None
        if c.query_shift:
            order=(torch.arange(s.n,device=x.device)+self.layer*c.query_shift%32)%s.n
            if self.layer:regroup_history(s,order)
            s.query_order=order
            q=q[:,order].contiguous()
        output=torch.empty_like(v);history=[];weights=[];grids=[];real_grids=[];stats=[];capture=[]
        candidates=max(1,math.ceil(s.fine.base_count*c.candidate_fraction))
        for start in range(0,s.groups,c.window_groups):
            g=min(c.window_groups,s.groups-start);first=start*32;end=min(s.n,(start+g)*32)
            if self.layer==0:
                ids=s.fine.reps[None].expand(g,-1).contiguous()
                pops=population[None].expand(g,-1).contiguous();real=torch.zeros_like(ids,dtype=torch.bool)
                record=dict(reuse_edges=torch.zeros(g,device=x.device),representatives=torch.full((g,),s.budget,device=x.device),
                    exact_edges=torch.zeros(g,device=x.device),dropped_clusters=torch.zeros(g,device=x.device),
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
                    pi=torch.sparse.mm(overlap,probability.T).T.contiguous()
                    residual=torch.sparse.mm(overlap,demand.T).T
                    candidate=residual.topk(candidates,dim=1).indices
                    allowed=torch.zeros_like(pi,dtype=torch.bool).scatter_(1,candidate,True)
                    ck=k[:,s.patches[s.fine.reps[candidate]]]
                    logits=(pq@ck.transpose(-1,-2)).float()*o.scale+population[candidate].log()[None,:,None]
                    fine_mass=(logits-normalizer).exp().mean((0,2))
                    pi_hat=pi.clone().scatter_(1,candidate,fine_mass)
                    pi_hat/=pi_hat.sum(1,keepdim=True).clamp_min(1e-30)
                with stage(s,'reuse_and_budget'),torch.autocast(x.device.type,enabled=False):
                    old_ids=None if s.history_ids is None else s.history_ids[start:start+g]
                    old_weights=None if s.history_weights is None else s.history_weights[start:start+g]
                    hids,hmass,discarded=top_history(old_ids,old_weights,beta,s.grid,s.fine.labels,s.fine.reps,c.history_per_cluster)
                    errors,costs=operating_points(pi,pi_hat,means,features[s.fine.reps],population,
                                                hids,hmass,features,allowed)
                    selected,count=budget_states(errors,costs,s.budget,c.dual_iterations)
                    ids,pops,real,record=pack_support(selected,hids,s.fine.reps,s.fine.labels,population,s.budget)
                    record['history_cap_discarded']=discarded
                    record['proxy_drop_error']=(errors[:,:,0]).sum(1)
                    record['proxy_selected_error']=errors.gather(2,selected[:,:,None]).sum((1,2))
            actual=s.patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int().contiguous()
            bias=pops.log().contiguous()
            with stage(s,'execution_and_evidence'):
                if q.is_cuda:
                    from saf3r.triton.reuse_explore_av import weighted_attention
                    result,mass=weighted_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale,single_qk=c.evidence_mode=="saved_logits")
                else:result,mass=reference_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale)
            output[:,first:end]=result
            with stage(s,'history_summary'):
                mass=mass/mass.sum(1,keepdim=True).clamp_min(1e-30)
                key_grid=s.grid[ids.clamp_min(0)]
                grid=torch.zeros((g,s.regions),device=x.device).scatter_add_(1,key_grid,mass)
                real_grid=torch.zeros_like(grid).scatter_add_(1,key_grid,mass*real)
                history.append(ids.masked_fill(~real,-1).int());weights.append((mass*real).half())
                grids.append(grid.half());real_grids.append(real_grid.half())
            stats.append(record)
            if c.capture:capture.append(dict(ids=actual.clone(),population=pops.clone(),genuine=real.clone(),
                                             mass=mass.clone(),reuse=record['reuse_edges'].clone()))
        s.history_ids=torch.cat(history);s.history_weights=torch.cat(weights)
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
            fine_probe_qk_per_head=0 if self.layer==0 else 4*s.groups*candidates,
            final_unique_qk_per_head=((joined['counts']+len(s.specials))*rows).sum(),
            evidence_extra_qk_per_head=(joined['counts']*rows).sum() if c.evidence_mode=='second_qk' else 0,
            evidence_scratch_peak_bytes=min(c.window_groups*32,s.n)*q.shape[0]*s.budget*4 if c.evidence_mode=='saved_logits' else 0,
            stored_history_bytes=s.history_ids.numel()*4+s.history_weights.numel()*2,
            stored_grid_bytes=s.previous_grid.numel()*4,
            population_compensation=True,head_aware=False,
            selector='separable vector-output proxy, discrete dual budget search; not exact knapsack',
            history='FP16 group-mean conditional patch mass; genuine edges only; no fixed retention ratio',
            evidence=c.evidence_mode,representative_mode=c.representative_mode,query_shift=c.query_shift))
        if c.capture:s.captured[self.layer]=dict(windows=capture,specials=s.specials.clone(),
            query_order=None if order is None else order.clone(),canonical=s.grid.clone(),fine_labels=s.fine.labels.clone(),fine_reps=s.fine.reps.clone())
        if c.representative_mode=="farthest":s.fine.reps=nearest_reps
        if order is not None:
            restored=torch.empty_like(output);restored[:,order]=output;output=restored
        return o.proj_drop(o.proj(output.transpose(0,1).reshape(1,s.n,-1)))


def install_reuse_residual_exploration(model,config=None):
    if any(hasattr(model,name) for name in ('value_region_av_state','fused_shared_av_state','attention_first_residual_value_state','reuse_explore_state')):
        raise ValueError('Install only on a fresh unpatched model')
    config=ReuseExploreConfig() if config is None else config
    if len(model.aggregator.global_blocks)!=24:raise ValueError('24 global layers required')
    state=ReuseExploreState(config)
    for layer,block in enumerate(model.aggregator.global_blocks):
        block.attn=ReuseResidualExploration(block.attn,state,layer).eval()
    def reset(module,inputs):
        images=inputs[0]
        if images.ndim!=5 or len(images)!=1:raise ValueError('Batch-one sequences required')
        state.reset(images.shape[1],images.shape[-2]//14,images.shape[-1]//14,images.device)
    state.reset_handle=model.aggregator.register_forward_pre_hook(reset)
    model.reuse_explore_state=state
    return state
