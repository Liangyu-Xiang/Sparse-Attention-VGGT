"""Unified History-reuse + current-layer Value exploration.

History is the complete previous sparse-mask patch support: every patch QK edge
that was actually executed, regardless of whether it entered as Rep, Exact, or
earlier History. History is a relation prior, never a predicted current-layer
attention probability. Current coarse attention supplies current-layer importance; current
V supplies representation redundancy. History and Value states compete under one
global budget through a common bounded loss.

The old artificial target pi*mean(V), residual subtraction, and fine
representative QK probes are deliberately removed.
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
    candidate_fraction:float=.25  # current-importance fraction eligible for exact expansion; no QK probe
    dual_iterations:int=16
    capture:bool=False

    def __post_init__(self):
        if not math.isfinite(self.retention) or not .05<=self.retention<=.2:raise ValueError('Retention must be5–20%')
        if min(self.grid_y,self.grid_x,self.source_width,self.local_iterations,self.window_groups,self.dual_iterations)<1:
            raise ValueError('Positive grid/window/iteration sizes required')
        if not 0<self.candidate_fraction<=1:
            raise ValueError('Invalid candidate fraction')


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


def compact_history_bank(ids,weights):
    """Remove per-group padding once per layer; preserve all executed patch relations."""
    valid=ids>=0
    if not valid.any():
        return ids[:,:0].contiguous(),weights[:,:0].contiguous()
    width=int(valid.sum(1).max().item())
    pos=valid.long().cumsum(1)-1
    slot=pos.masked_fill(~valid,width)
    out=torch.full((len(ids),width+1),-1,device=ids.device,dtype=ids.dtype)
    score=torch.zeros((len(ids),width+1),device=weights.device,dtype=weights.dtype)
    out.scatter_(1,slot,ids)
    score.scatter_(1,slot,weights)
    return out[:,:width].contiguous(),score[:,:width].contiguous()


def history_relation_items(ids,weights,beta,current_region,grid):
    """Cheap query-group History candidates, without Fine-cluster remapping.

    A previous genuine QK edge remains a relation candidate by identity. Old
    attention is used only to distribute salience *within its previous canonical
    region*. The absolute current scale comes from the CURRENT coarse region
    probability. Thus no previous probability is transferred across layers.

        conditional_salience_e = old_weight_e / sum_{j in H_r} old_weight_j
        gain_e = p_r^current * beta_r * conditional_salience_e

    Every patch edge actually executed by the previous sparse mask is eligible,
    regardless of whether that key entered as a representative, an Exact member,
    or an already reused History edge. Filtering happens in the NEXT layer via
    current coarse importance/persistence and the previous edge's measured
    within-region salience.

    The gains of all old edges in a region are bounded by current coarse
    importance times persistence, giving natural saturation without a fixed
    History ratio, threshold, or per-cluster History cap.
    """
    g=beta.shape[0];device=beta.device
    if ids is None:
        return (torch.empty((g,0),device=device,dtype=torch.long),
                torch.empty((g,0),device=device),
                torch.empty((g,0),device=device,dtype=torch.bool))
    valid=ids>=0
    safe=ids.long().clamp_min(0)
    region=grid[safe]
    old=weights.float().clamp_min(0)*valid
    totals=torch.zeros_like(current_region).scatter_add_(1,region,old)
    conditional=old/totals.gather(1,region).clamp_min(1e-30)
    gain=current_region.gather(1,region)*beta.gather(1,region)*conditional
    gain=gain.masked_fill(~valid,0)
    return ids.long(),gain,valid


def exploration_operating_points(pi,rep_dist,population,exact_allowed):
    """Current-layer Value exploration only.

    Coarse attention supplies current importance; current V supplies redundancy.
    History is deliberately absent from these states. History and exploration
    meet only in the shared global budget allocator.
    """
    drop=pi
    rep=pi*rep_dist[None]
    exact=torch.zeros_like(pi).masked_fill(~exact_allowed,torch.inf)
    errors=torch.stack((drop,rep,exact),-1)
    costs=torch.stack((torch.zeros_like(pi),torch.ones_like(pi),
                       population[None].expand_as(pi)), -1)
    return errors,costs


def joint_operating_points(history_gain,history_valid,explore_errors,explore_costs):
    """Put relation reuse and Value exploration on one shared budget.

    Each History edge is a two-state item: drop incurs its missed relation gain,
    keep costs one QK edge and removes that loss. Each Fine cluster keeps its
    Drop/Rep/Exact Value states. No History candidate is evaluated by current
    Value representativeness before it can compete for budget.
    """
    if history_gain.shape[1]:
        inf=torch.full_like(history_gain,torch.inf)
        h_errors=torch.stack((history_gain,torch.zeros_like(history_gain),inf),-1)
        h_errors[:,:,1]=h_errors[:,:,1].masked_fill(~history_valid,torch.inf)
        h_costs=torch.stack((torch.zeros_like(history_gain),torch.ones_like(history_gain),
                             torch.zeros_like(history_gain)),-1)
        return (torch.cat((h_errors,explore_errors),1),
                torch.cat((h_costs,explore_costs),1))
    return explore_errors,explore_costs


def pack_support(selected,history_ids,reps,labels,population,budget):
    """Pack independently selected History relations and Value states.

    History is selected globally at query-group level. Only AFTER selection do
    we map retained History IDs to current Fine labels for population accounting.
    This mapping does not participate in deciding whether the History edge is
    valuable.

    If the current Value representative is itself a retained History key, the
    two roles are merged into one physical key and their populations are summed.
    Exact expansion suppresses duplicate retained History members.
    """
    g=selected.shape[0];h=history_ids.shape[1];f=len(reps);device=selected.device
    hkeep=(selected[:,:h]==1)&(history_ids>=0)
    estate=selected[:,h:]
    rep=estate==1;exact=estate==2

    safe=history_ids.clamp_min(0)
    hcluster=labels[safe]
    hcount=torch.zeros((g,f),device=device)
    hcount.scatter_add_(1,hcluster,hkeep.float())

    # Selected History keys are explicit unit-population relations by default.
    hist=history_ids.masked_fill(~hkeep,-1)
    hist_pop=hkeep.float()

    # A representative covers only members not already explicit History edges.
    residual=(population[None]-hcount).clamp_min(0)
    rep_selected=rep & (residual>0)
    rep_ids=reps[None].expand(g,-1).masked_fill(~rep_selected,-1)
    rep_pop=residual*rep_selected

    # If the representative is itself one of the retained History IDs, merge
    # the residual population into that History slot and suppress duplicate rep.
    if h:
        rep_for_hist=reps[hcluster]
        hist_is_rep=hkeep & rep.gather(1,hcluster) & (history_ids==rep_for_hist)
        hist_pop=hist_pop + hist_is_rep.float()*residual.gather(1,hcluster)
        duplicate_rep=torch.zeros((g,f),device=device,dtype=torch.int32)
        duplicate_rep.scatter_add_(1,hcluster,hist_is_rep.int())
        duplicate_rep=duplicate_rep>0
        rep_ids=rep_ids.masked_fill(duplicate_rep,-1)
        rep_pop=rep_pop.masked_fill(duplicate_rep,0)

    # Exact expansion emits every not-already-retained member of exact clusters.
    member=torch.arange(len(labels),device=device)[None].expand(g,-1)
    exact_member=exact[:,labels]
    if h:
        selected_map=torch.zeros((g,len(labels)),device=device,dtype=torch.bool)
        selected_map.scatter_(1,safe,hkeep)
        exact_member &= ~selected_map
    member=member.masked_fill(~exact_member,-1)

    raw=torch.cat((hist,rep_ids,member),1)
    pop=torch.cat((hist_pop,rep_pop,(member>=0).float()),1)
    relation=torch.cat((hkeep,torch.zeros_like(rep_selected),(member>=0)),1)

    valid=raw>=0;pos=valid.long().cumsum(1)-1
    slots=pos.masked_fill(~valid,budget)
    out=torch.full((g,budget+1),-1,device=device,dtype=torch.long)
    out_pop=torch.zeros((g,budget+1),device=device)
    real=torch.zeros((g,budget+1),device=device,dtype=torch.bool)
    out.scatter_(1,slots,raw);out_pop.scatter_(1,slots,pop);real.scatter_(1,slots,relation)
    out=out[:,:-1].contiguous();out_pop=out_pop[:,:-1].contiguous();real=real[:,:-1].contiguous()
    return out,out_pop,real,dict(
        reuse_edges=hkeep.sum(1),representatives=(rep_ids>=0).sum(1),
        exact_edges=(member>=0).sum(1),dropped_clusters=(estate==0).sum(1),
        counts=valid.sum(1),history_rep_merges=(hist_pop>1).sum(1))

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
        self.history_ids=None;self.history_weights=None;self.previous_grid=None
        self.previous_layer=-1;self.records=[];self.captured={};self.profile_events=[]

    def prepare(self,k,features):
        f=self.fine.base_count;r=self.regions
        population=torch.bincount(self.fine.labels,minlength=f).float()
        unit=features.float()/features.float().norm(dim=-1,keepdim=True).clamp_min(1e-12)
        mean_unit=torch.zeros((f,unit.shape[-1]),device=k.device).index_add_(0,self.fine.labels,unit)
        mean_unit/=population[:,None]
        rep_unit=unit[self.fine.reps]
        rep_dist=((1-(mean_unit*rep_unit).sum(-1)).clamp(0,2)*.5)
        pooled=torch.zeros((len(k),r,k.shape[-1]),device=k.device)
        pooled.index_add_(1,self.grid,k[:,self.patches].float())
        pooled=(pooled/self.grid_sizes[None,:,None]).to(k.dtype).contiguous()
        pair=self.fine.labels*r+self.grid
        unique,count=torch.unique(pair,sorted=True,return_counts=True)
        overlap=torch.sparse_coo_tensor(torch.stack((unique//r,unique%r)),
            count.float()/self.grid_sizes[unique%r],size=(f,r),device=k.device).coalesce()
        return population,unit,mean_unit,rep_dist,pooled,overlap



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
        with stage(s,'pooling'),torch.autocast(x.device.type,enabled=False):
            population,unit_features,mean_unit,rep_dist,pooled,overlap=s.prepare(k,features)
        output=torch.empty_like(v);history=[];weights=[];grids=[];stats=[];capture=[]
        exact_count=max(1,math.ceil(s.fine.base_count*c.candidate_fraction))
        for start in range(0,s.groups,c.window_groups):
            g=min(c.window_groups,s.groups-start);first=start*32;end=min(s.n,(start+g)*32)
            probe=(torch.arange(start,start+g,device=x.device)[:,None]*32+
                   torch.tensor([4,12,20,28],device=x.device)).clamp_max(s.n-1)
            pq=q[:,probe]
            with stage(s,'coarse_observation'):
                z=(pq@pooled[:,None].transpose(-1,-2)).float()*o.scale+s.grid_sizes.log()
                probability=z.softmax(-1).mean((0,2))
                pi=torch.sparse.mm(overlap,probability.T).T.contiguous()
            if self.layer==0:
                ids=s.fine.reps[None].expand(g,-1).contiguous()
                pops=population[None].expand(g,-1).contiguous();real=torch.zeros_like(ids,dtype=torch.bool)
                record=dict(reuse_edges=torch.zeros(g,device=x.device),representatives=torch.full((g,),s.budget,device=x.device),
                    exact_edges=torch.zeros(g,device=x.device),dropped_clusters=torch.zeros(g,device=x.device),
                    counts=torch.full((g,),s.budget,device=x.device),history_rep_merges=torch.zeros(g,device=x.device),
                    history_candidates=torch.zeros(g,device=x.device))
            else:
                with stage(s,'reuse_prior'):
                    previous=s.previous_grid[start:start+g].float()
                    beta=torch.where(previous>1e-12,(probability/previous.clamp_min(1e-12)).clamp_max(1),0)
                    candidate=pi.topk(exact_count,dim=1).indices
                    allowed=torch.zeros_like(pi,dtype=torch.bool).scatter_(1,candidate,True)
                with stage(s,'reuse_and_budget'),torch.autocast(x.device.type,enabled=False):
                    old_ids=None if s.history_ids is None else s.history_ids[start:start+g]
                    old_weights=None if s.history_weights is None else s.history_weights[start:start+g]
                    hids,hgain,hvalid=history_relation_items(old_ids,old_weights,beta,probability,s.grid)
                    e_errors,e_costs=exploration_operating_points(pi,rep_dist,population,allowed)
                    errors,costs=joint_operating_points(hgain,hvalid,e_errors,e_costs)
                    selected,count=budget_states(errors,costs,s.budget,c.dual_iterations)
                    ids,pops,real,record=pack_support(selected,hids,s.fine.reps,s.fine.labels,population,s.budget)
                    record['history_candidates']=hvalid.sum(1)
                    record['proxy_drop_error']=errors[:,:,0].masked_fill(torch.isinf(errors[:,:,0]),0).sum(1)
                    record['proxy_selected_error']=errors.gather(2,selected[:,:,None]).squeeze(-1).sum(1)
            actual=s.patches[ids.clamp_min(0)].masked_fill(ids<0,-1).int().contiguous()
            bias=pops.log().contiguous()
            with stage(s,'execution_and_evidence'):
                if q.is_cuda:
                    from saf3r.triton.reuse_explore_av import weighted_attention_probe_mass
                    result,mass=weighted_attention_probe_mass(q[:,first:end],k,v,actual,bias,s.specials,o.scale)
                else:result,mass=reference_attention(q[:,first:end],k,v,actual,bias,s.specials,o.scale)
            output[:,first:end]=result
            with stage(s,'history_summary'):
                # History is the complete patch support that actually executed QK
                # in this layer.  For population representatives, divide aggregate
                # probability mass by population to recover a single-edge salience
                # proxy; do NOT discard the edge just because it was a representative.
                mass=mass/mass.sum(1,keepdim=True).clamp_min(1e-30)
                valid=ids>=0
                relation_mass=(mass/pops.clamp_min(1))*valid
                history.append(ids.masked_fill(~valid,-1).int());weights.append(relation_mass.half())
                grids.append(probability.half())
            stats.append(record)
            if c.capture:
                if self.layer==0:
                    history_candidates=torch.empty((g,0),device=x.device,dtype=torch.int32)
                    history_selected=torch.empty((g,0),device=x.device,dtype=torch.int32)
                else:
                    history_candidates=s.patches[hids.clamp_min(0)].masked_fill(~hvalid,-1).int()
                    hkeep=(selected[:,:hids.shape[1]]==1)&hvalid
                    history_selected=s.patches[hids.clamp_min(0)].masked_fill(~hkeep,-1).int()
                capture.append(dict(ids=actual.clone(),population=pops.clone(),genuine=real.clone(),
                    all_executed_history=(ids>=0).clone(),mass=mass.clone(),reuse=record['reuse_edges'].clone(),
                    history_candidates=history_candidates.clone(),
                    history_selected=history_selected.clone()))
        full_history=torch.cat(history);full_weights=torch.cat(weights)
        s.history_ids,s.history_weights=compact_history_bank(full_history,full_weights)
        s.previous_grid=torch.cat(grids)
        s.previous_layer=self.layer
        joined={key:torch.cat([r[key] for r in stats]) for key in stats[0]}
        rows=(s.n-torch.arange(s.groups,device=x.device)*32).clamp_max(32)
        s.records.append(dict(layer=self.layer,patch_budget=s.budget,special_keys=len(s.specials),
            fine_clusters=s.fine.base_count,canonical_regions=s.regions,
            **{key+'_mean':val.float().mean() for key,val in joined.items()},
            actual_unique_density=((joined['counts']+len(s.specials))*rows).sum()/s.n**2,
            unused_budget_mean=(s.budget-joined['counts']).float().mean(),
            canonical_qk_per_head=4*s.groups*s.regions,
            fine_probe_qk_per_head=0,
            final_unique_qk_per_head=((joined['counts']+len(s.specials))*rows).sum(),
            evidence_extra_qk_per_head=(joined['counts']*torch.minimum(rows,torch.full_like(rows,4))).sum(),
            stored_history_bytes=s.history_ids.numel()*4+s.history_weights.numel()*2,
            history_reuse_ratio_mean=(joined['reuse_edges'].float()/joined['counts'].clamp_min(1).float()).mean(),
            history_filter_keep_ratio_mean=torch.where(joined['history_candidates']>0,
                joined['reuse_edges'].float()/joined['history_candidates'].float(),torch.zeros_like(joined['reuse_edges'].float())).mean(),
            stored_grid_bytes=s.previous_grid.numel()*2,
            population_compensation=True,head_aware=False,
            selector='global History relation items + current Value Drop/Rep/Exact items under one dual budget',
            history='all previously executed sparse-mask patch edges; gain=current coarse importance x persistence x old within-region salience',
            evidence='4/32-query probe evidence for next-layer salience; full repeated selected-QK evidence removed'))
        if c.capture:s.captured[self.layer]=dict(windows=capture,specials=s.specials.clone(),
            canonical=s.grid.clone(),fine_labels=s.fine.labels.clone(),fine_reps=s.fine.reps.clone())
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
