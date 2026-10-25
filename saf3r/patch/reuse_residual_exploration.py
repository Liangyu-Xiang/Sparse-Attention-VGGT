"""History-conditioned reuse + residual current-layer Value exploration.

History is the complete previous sparse-mask patch support: every patch QK edge
that was actually executed, regardless of whether it entered as Rep, Exact, or
earlier History.  History reliability is determined only from historical
relation evidence and current coarse persistence; current Value similarity never
invalidates a reliable History edge.

Exploration is conditioned on the History edges selected at the current
Lagrange price.  Retained History keys become already-paid Value anchors inside
their current Fine clusters, so Drop/Rep/Exact decisions optimize only residual
Value distortion and incremental physical cost.  Thus reuse can replace
redundant exploration instead of merely competing with an independent
Exploration loss.

The old artificial target pi*mean(V), explicit current-minus-previous residual
subtraction, and fine representative QK probes remain deliberately removed.
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
    history_conditioned_exploration:bool=True
    physical_overlap_accounting:bool=True  # legacy control path only
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

    Every previous executed patch QK edge remains a relation candidate by identity. Old
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


def conditioned_exploration_at_price(history_gain,history_valid,history_ids,price,
                                     pi,rep_dist,population,exact_allowed,
                                     labels,reps,unit_features,mean_unit):
    """One Lagrange-price step for History-conditioned Exploration.

    History is selected *only* from relation reliability:
        keep h iff history_gain(h) > price.

    The selected History keys are then treated as already-paid anchors in their
    current Value clusters.  Exploration optimizes only residual representation
    distortion:

      Drop : pi_C * D(C | H_C),                         incremental cost 0
      Rep  : pi_C * D(C | H_C union {r_C}),             cost 0 or 1
      Exact: 0,                                          cost |C|-|H_C|

    D is a cheap cluster-centre cosine distortion.  In particular, Value
    similarity can suppress a redundant *new representative* but can never
    remove a History edge that passed the reliability decision.

    This is an efficient flattened approximation to the intended outer-History /
    inner-Exploration dependence.  It deliberately does not resurrect the old
    pi*mean(V) output target or current-minus-previous probability subtraction.
    """
    g,f=pi.shape;device=pi.device
    h=0 if history_ids is None else history_ids.shape[1]
    if h:
        hkeep=history_valid & (history_gain>price[:,None])
        safe=history_ids.long().clamp_min(0)
        hcluster=labels[safe]

        hcount=torch.zeros((g,f),device=device)
        hcount.scatter_add_(1,hcluster,hkeep.float())

        # Use the same normalized-Value/cluster-centre geometry as rep_dist.
        # Unselected/padded History cannot become a residual anchor.
        hu=unit_features[safe]
        hc=mean_unit[hcluster]
        hdist=((1-(hu*hc).sum(-1)).clamp(0,2)*.5).masked_fill(~hkeep,torch.inf)
        history_dist=torch.full((g,f),torch.inf,device=device)
        history_dist.scatter_reduce_(1,hcluster,hdist,reduce='amin',include_self=True)
        history_dist=torch.where(torch.isfinite(history_dist),history_dist,torch.ones_like(history_dist))

        hist_is_rep=hkeep & (safe==reps[hcluster])
        rep_in_history=torch.zeros((g,f),device=device,dtype=torch.int32)
        rep_in_history.scatter_add_(1,hcluster,hist_is_rep.int())
        rep_in_history=rep_in_history>0
    else:
        hkeep=torch.empty((g,0),device=device,dtype=torch.bool)
        hcount=torch.zeros((g,f),device=device)
        history_dist=torch.ones((g,f),device=device)
        rep_in_history=torch.zeros((g,f),device=device,dtype=torch.bool)

    residual=(population[None]-hcount).clamp_min(0)
    drop_error=pi*history_dist
    rep_error=pi*torch.minimum(history_dist,rep_dist[None])
    exact_error=torch.zeros_like(pi).masked_fill(~exact_allowed,torch.inf)

    rep_cost=((residual>0)&~rep_in_history).float()
    exact_cost=residual
    errors=torch.stack((drop_error,rep_error,exact_error),-1)
    costs=torch.stack((torch.zeros_like(pi),rep_cost,exact_cost),-1)
    score=errors+price[:,None,None]*costs
    estate=score.argmin(-1)

    # If the already-selected History key is itself the representative, assigning
    # it the representative role is free.  Prefer that role on an exact tie so
    # population compensation covers the remaining cluster members at zero
    # additional QK cost.
    free_rep=rep_in_history & (rep_error<=drop_error)
    estate=torch.where(free_rep & (estate==0),torch.ones_like(estate),estate)

    exploration_cost=costs.gather(2,estate[:,:,None]).squeeze(-1).sum(1)
    total_cost=hkeep.sum(1).float()+exploration_cost
    return hkeep,estate,total_cost,errors,costs,history_dist,hcount,rep_in_history


def conditioned_budget_states(history_gain,history_valid,history_ids,
                              pi,rep_dist,population,exact_allowed,
                              labels,reps,unit_features,mean_unit,budget,iterations=16):
    """Approximate the nested History -> residual Exploration problem.

    A single shadow price is searched only to satisfy the global physical QK
    budget.  At each price, History is filtered from relation evidence first;
    Exploration is then recomputed *conditioned on that retained History*.
    This preserves the causal ordering without materializing an exponential
    History/Exploration state table.
    """
    g=pi.shape[0];device=pi.device
    hmax=history_gain.masked_fill(~history_valid,0).amax(1) if history_gain.shape[1] else torch.zeros(g,device=device)
    # Drop/Rep errors are bounded by pi<=1, so this upper bound always permits
    # the zero-cost all-drop solution.
    lo=torch.zeros(g,device=device)
    hi=torch.maximum(hmax,pi.amax(1))+1e-6
    latest=None
    for _ in range(iterations):
        mid=(lo+hi)*.5
        state=conditioned_exploration_at_price(
            history_gain,history_valid,history_ids,mid,pi,rep_dist,population,exact_allowed,
            labels,reps,unit_features,mean_unit)
        over=state[2]>budget
        lo=torch.where(over,mid,lo)
        hi=torch.where(over,hi,mid)
        latest=state
    state=conditioned_exploration_at_price(
        history_gain,history_valid,history_ids,hi,pi,rep_dist,population,exact_allowed,
        labels,reps,unit_features,mean_unit)
    return state+(hi,)


def pack_conditioned_support(hkeep,estate,history_ids,reps,labels,population,budget):
    """Pack fixed History first, then only the residual Exploration support."""
    g,f=estate.shape;device=estate.device
    h=0 if history_ids is None else history_ids.shape[1]
    safe=history_ids.long().clamp_min(0) if h else torch.empty((g,0),device=device,dtype=torch.long)
    hist=history_ids.long().masked_fill(~hkeep,-1) if h else safe
    hcluster=labels[safe] if h else safe
    hcount=torch.zeros((g,f),device=device)
    if h:
        hcount.scatter_add_(1,hcluster,hkeep.float())

    residual=(population[None]-hcount).clamp_min(0)
    rep=estate==1
    exact=estate==2

    if h:
        hist_is_rep=hkeep & (safe==reps[hcluster])
        duplicate_rep=torch.zeros((g,f),device=device,dtype=torch.int32)
        duplicate_rep.scatter_add_(1,hcluster,hist_is_rep.int())
        duplicate_rep=duplicate_rep>0
    else:
        hist_is_rep=torch.empty((g,0),device=device,dtype=torch.bool)
        duplicate_rep=torch.zeros((g,f),device=device,dtype=torch.bool)

    hist_pop=hkeep.float()
    if h:
        # A retained History representative can cover the residual cluster
        # population without another physical QK edge.
        hist_pop=hist_pop + hist_is_rep.float()*(
            residual.gather(1,hcluster)*rep.gather(1,hcluster))

    rep_physical=rep & (residual>0) & ~duplicate_rep
    rep_ids=reps[None].expand(g,-1).masked_fill(~rep_physical,-1)
    rep_pop=residual*rep_physical

    member=torch.arange(len(labels),device=device)[None].expand(g,-1)
    exact_member=exact[:,labels]
    if h:
        selected_map=torch.zeros((g,len(labels)),device=device,dtype=torch.bool)
        selected_map.scatter_(1,safe,hkeep)
        exact_member &= ~selected_map
    member=member.masked_fill(~exact_member,-1)

    raw=torch.cat((hist,rep_ids,member),1)
    pop=torch.cat((hist_pop,rep_pop,(member>=0).float()),1)
    relation=torch.cat((hkeep,torch.zeros_like(rep_physical),(member>=0)),1)
    valid=raw>=0
    pos=valid.long().cumsum(1)-1
    if bool((valid.sum(1)>budget).any()):
        raise RuntimeError('Conditioned selector exceeded physical sparse budget')
    slots=pos.masked_fill(~valid,budget)
    out=torch.full((g,budget+1),-1,device=device,dtype=torch.long)
    out_pop=torch.zeros((g,budget+1),device=device)
    real=torch.zeros((g,budget+1),device=device,dtype=torch.bool)
    out.scatter_(1,slots,raw);out_pop.scatter_(1,slots,pop);real.scatter_(1,slots,relation)
    out=out[:,:-1].contiguous();out_pop=out_pop[:,:-1].contiguous();real=real[:,:-1].contiguous()

    selected_history_in_exact=(hkeep & exact.gather(1,hcluster)).sum(1) if h else torch.zeros(g,device=device)
    return out,out_pop,real,dict(
        reuse_edges=hkeep.sum(1),
        representatives=(rep_ids>=0).sum(1),
        exact_edges=(member>=0).sum(1),
        dropped_clusters=(estate==0).sum(1),
        counts=valid.sum(1),
        history_rep_merges=(hist_is_rep & rep.gather(1,hcluster)).sum(1) if h else torch.zeros(g,device=device),
        history_exact_overlap_selected=selected_history_in_exact,
        population_representative_clusters=rep.sum(1))


def physical_joint_operating_points(history_gain,history_valid,history_ids,reps,
                                    explore_errors,explore_costs,num_patches):
    """Merge History/current-Rep overlap before budget optimization.

    If a previous History key is also the current representative of cluster C,
    it is one physical key, not two independent cost-1 items.  The cluster gets
    four states:

      Drop         : lose History gain + cluster drop loss, cost 0
      History-only : keep the QK relation only, cluster still dropped, cost 1
      Rep           : same physical key serves History and representative, cost 1
      Exact         : exact cluster, which also executes the representative, cost |C|

    The overlapping standalone History item is disabled and its utility is
    transferred into the cluster states.  Non-overlapping History and clusters
    retain their original semantics.  Exact overlap with *other* History members
    is deliberately left unchanged in this causal ablation.
    """
    g=explore_errors.shape[0];f=explore_errors.shape[1];device=explore_errors.device
    h=0 if history_ids is None else history_ids.shape[1]
    inf=torch.full((g,1),torch.inf,device=device)

    if h:
        valid=history_valid
        safe=torch.where(valid,history_ids.long(),torch.full_like(history_ids.long(),num_patches))
        table=torch.full((g,num_patches+1),-1,device=device,dtype=torch.long)
        pos=torch.arange(h,device=device)[None].expand(g,-1)
        table.scatter_(1,safe,torch.where(valid,pos,torch.full_like(pos,-1)))
        overlap_index=table[:,reps]
        overlap_valid=overlap_index>=0
        safe_overlap=overlap_index.clamp_min(0)
        overlap_gain=history_gain.gather(1,safe_overlap)*overlap_valid

        matched_count=torch.zeros((g,h),device=device,dtype=torch.int32)
        matched_count.scatter_add_(1,safe_overlap,overlap_valid.int())
        matched_history=matched_count>0

        hdrop=torch.where(valid & ~matched_history,history_gain,torch.zeros_like(history_gain))
        hkeep=torch.zeros_like(history_gain).masked_fill(~valid | matched_history,torch.inf)
        hinf=torch.full_like(history_gain,torch.inf)
        h_errors=torch.stack((hdrop,hkeep,hinf,hinf),-1)
        h_costs=torch.stack((torch.zeros_like(history_gain),torch.ones_like(history_gain),
                             torch.zeros_like(history_gain),torch.zeros_like(history_gain)),-1)
    else:
        overlap_index=torch.full((g,f),-1,device=device,dtype=torch.long)
        overlap_valid=torch.zeros((g,f),device=device,dtype=torch.bool)
        overlap_gain=torch.zeros((g,f),device=device)
        matched_history=torch.empty((g,0),device=device,dtype=torch.bool)
        h_errors=torch.empty((g,0,4),device=device)
        h_costs=torch.empty((g,0,4),device=device)

    drop,rep,exact=explore_errors.unbind(-1)
    c0=drop+overlap_gain
    c1=torch.where(overlap_valid,drop,rep)
    c2=torch.where(overlap_valid,rep,exact)
    c3=torch.where(overlap_valid,exact,torch.full_like(exact,torch.inf))
    c_errors=torch.stack((c0,c1,c2,c3),-1)

    zero=torch.zeros_like(drop);one=torch.ones_like(drop)
    rep_cost=explore_costs[:,:,1];exact_cost=explore_costs[:,:,2]
    c_costs=torch.stack((
        zero,
        torch.where(overlap_valid,one,rep_cost),
        torch.where(overlap_valid,rep_cost,exact_cost),
        torch.where(overlap_valid,exact_cost,zero)), -1)

    return (torch.cat((h_errors,c_errors),1),torch.cat((h_costs,c_costs),1),
            overlap_valid,overlap_index,matched_history)


def pack_support_physical(selected,history_ids,reps,labels,population,budget,overlap_valid):
    """Pack support for physical History/Rep overlap accounting.

    For an overlapping cluster, state 1 is History-only, state 2 is the merged
    History+Rep physical key, and state 3 is Exact.  For a non-overlapping
    cluster, states 1/2 remain Rep/Exact.
    """
    g=selected.shape[0];h=history_ids.shape[1];f=len(reps);device=selected.device
    hstate=selected[:,:h];estate=selected[:,h:]
    hkeep=(hstate==1)&(history_ids>=0)

    overlap_hist_only=overlap_valid&(estate==1)
    overlap_rep=overlap_valid&(estate==2)
    nonoverlap_rep=(~overlap_valid)&(estate==1)
    exact=((~overlap_valid)&(estate==2))|(overlap_valid&(estate==3))
    dropped=torch.where(overlap_valid,estate<=1,estate==0)

    hist_main=history_ids.masked_fill(~hkeep,-1)
    hist_overlap=reps[None].expand(g,-1).masked_fill(~(overlap_hist_only|overlap_rep),-1)
    hist=torch.cat((hist_main,hist_overlap),1)
    hvalid=hist>=0
    safe_hist=hist.clamp_min(0)
    hcluster=labels[safe_hist]
    hcount=torch.zeros((g,f),device=device)
    hcount.scatter_add_(1,hcluster,hvalid.float())

    residual=(population[None]-hcount).clamp_min(0)
    hist_pop=hvalid.float()
    # The overlap Rep state uses the History key as the representative too.
    second_pop=torch.where(overlap_rep,1+residual,torch.ones_like(residual))
    hist_pop[:,h:]=hist_pop[:,h:]*second_pop

    rep_ids=reps[None].expand(g,-1).masked_fill(~nonoverlap_rep,-1)
    rep_pop=residual*nonoverlap_rep

    member=torch.arange(len(labels),device=device)[None].expand(g,-1)
    exact_member=exact[:,labels]
    if hist.shape[1]:
        selected_map=torch.zeros((g,len(labels)),device=device,dtype=torch.bool)
        selected_map.scatter_(1,safe_hist,hvalid)
        exact_member &= ~selected_map
    member=member.masked_fill(~exact_member,-1)

    raw=torch.cat((hist,rep_ids,member),1)
    pop=torch.cat((hist_pop,rep_pop,(member>=0).float()),1)
    relation=torch.cat((hvalid,torch.zeros_like(nonoverlap_rep),(member>=0)),1)
    valid=raw>=0;pos=valid.long().cumsum(1)-1
    slots=pos.masked_fill(~valid,budget)
    out=torch.full((g,budget+1),-1,device=device,dtype=torch.long)
    out_pop=torch.zeros((g,budget+1),device=device)
    real=torch.zeros((g,budget+1),device=device,dtype=torch.bool)
    out.scatter_(1,slots,raw);out_pop.scatter_(1,slots,pop);real.scatter_(1,slots,relation)
    out=out[:,:-1].contiguous();out_pop=out_pop[:,:-1].contiguous();real=real[:,:-1].contiguous()

    main_cluster=labels[history_ids.clamp_min(0)]
    history_exact_overlap_selected=(hkeep & exact.gather(1,main_cluster)).sum(1)
    return out,out_pop,real,dict(
        reuse_edges=hvalid.sum(1),
        representatives=(rep_ids>=0).sum(1),
        exact_edges=(member>=0).sum(1),
        dropped_clusters=dropped.sum(1),
        counts=valid.sum(1),
        history_rep_merges=overlap_rep.sum(1),
        history_exact_overlap_selected=history_exact_overlap_selected,
        population_representative_clusters=(nonoverlap_rep|overlap_rep).sum(1))


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
                    history_candidates=torch.zeros(g,device=x.device),
                    history_rep_candidate_overlap=torch.zeros(g,device=x.device),
                    history_exact_overlap_selected=torch.zeros(g,device=x.device),
                    population_representative_clusters=torch.full((g,),s.budget,device=x.device),
                    optimizer_cost=torch.full((g,),s.budget,device=x.device),
                    conditioned_history_distortion=torch.ones(g,device=x.device),
                    history_selected_cluster_coverage=torch.zeros(g,device=x.device),
                    shadow_price=torch.zeros(g,device=x.device),
                    missed_history_gain=torch.zeros(g,device=x.device),
                    history_conditioned_clusters=torch.zeros(g,device=x.device),
                    history_only_clusters=torch.zeros(g,device=x.device),
                    history_plus_rep_clusters=torch.zeros(g,device=x.device),
                    history_plus_exact_clusters=torch.zeros(g,device=x.device))
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
                    history_hits=torch.zeros((g,len(s.patches)),device=x.device,dtype=torch.int32)
                    history_hits.scatter_add_(1,hids.clamp_min(0),hvalid.int())
                    history_rep_candidate_overlap=(history_hits[:,s.fine.reps]>0).sum(1)
                    if c.history_conditioned_exploration:
                        (hkeep,estate,count,errors,costs,history_dist,hcount,rep_in_history,price)=conditioned_budget_states(
                            hgain,hvalid,hids,pi,rep_dist,population,allowed,s.fine.labels,s.fine.reps,
                            unit_features,mean_unit,s.budget,c.dual_iterations)
                        ids,pops,real,record=pack_conditioned_support(
                            hkeep,estate,hids,s.fine.reps,s.fine.labels,population,s.budget)
                        selected=None
                        record['conditioned_history_distortion']=history_dist.mean(1)
                        record['history_selected_cluster_coverage']=(hcount>0).float().mean(1)
                        record['shadow_price']=price
                    else:
                        e_errors,e_costs=exploration_operating_points(pi,rep_dist,population,allowed)
                        if c.physical_overlap_accounting:
                            errors,costs,overlap_valid,overlap_index,matched_history=physical_joint_operating_points(
                                hgain,hvalid,hids,s.fine.reps,e_errors,e_costs,len(s.patches))
                            selected,count=budget_states(errors,costs,s.budget,c.dual_iterations)
                            ids,pops,real,record=pack_support_physical(
                                selected,hids,s.fine.reps,s.fine.labels,population,s.budget,overlap_valid)
                        else:
                            errors,costs=joint_operating_points(hgain,hvalid,e_errors,e_costs)
                            selected,count=budget_states(errors,costs,s.budget,c.dual_iterations)
                            ids,pops,real,record=pack_support(selected,hids,s.fine.reps,s.fine.labels,population,s.budget)
                            record['population_representative_clusters']=record['representatives']
                            record['history_exact_overlap_selected']=torch.zeros_like(record['reuse_edges'])
                        record['conditioned_history_distortion']=torch.ones(g,device=x.device)
                        record['history_selected_cluster_coverage']=torch.zeros(g,device=x.device)
                        record['shadow_price']=torch.zeros(g,device=x.device)
                    record['history_candidates']=hvalid.sum(1)
                    record['history_rep_candidate_overlap']=history_rep_candidate_overlap
                    record['optimizer_cost']=count
                    record['proxy_drop_error']=errors[:,:,0].masked_fill(torch.isinf(errors[:,:,0]),0).sum(1)
                    record['proxy_selected_error']=errors.gather(
                        2,estate[:,:,None] if c.history_conditioned_exploration else selected[:,:,None]).squeeze(-1).sum(1)
                    if c.history_conditioned_exploration:
                        record['missed_history_gain']=(hgain*(hvalid&~hkeep)).sum(1)
                        record['history_conditioned_clusters']=(hcount>0).sum(1)
                        record['history_only_clusters']=((hcount>0)&(estate==0)).sum(1)
                        record['history_plus_rep_clusters']=((hcount>0)&(estate==1)).sum(1)
                        record['history_plus_exact_clusters']=((hcount>0)&(estate==2)).sum(1)
                    else:
                        record['missed_history_gain']=torch.zeros(g,device=x.device)
                        record['history_conditioned_clusters']=torch.zeros(g,device=x.device)
                        record['history_only_clusters']=torch.zeros(g,device=x.device)
                        record['history_plus_rep_clusters']=torch.zeros(g,device=x.device)
                        record['history_plus_exact_clusters']=torch.zeros(g,device=x.device)
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
                    if c.history_conditioned_exploration:
                        history_selected=s.patches[hids.clamp_min(0)].masked_fill(~hkeep,-1).int()
                    else:
                        hkeep_capture=(selected[:,:hids.shape[1]]==1)&hvalid
                        history_selected=s.patches[hids.clamp_min(0)].masked_fill(~hkeep_capture,-1).int()
                        if c.physical_overlap_accounting:
                            # Merged History-only/History+Rep states disable the
                            # standalone History item, but still retain its relation.
                            estate_capture=selected[:,hids.shape[1]:]
                            merged=overlap_valid & ((estate_capture==1)|(estate_capture==2))
                            merged_ids=s.patches[s.fine.reps][None].expand(g,-1).masked_fill(~merged,-1).int()
                            history_selected=torch.cat((history_selected,merged_ids),1)
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
            physical_budget_utilization_mean=(joined['counts'].float()/s.budget).mean(),
            optimizer_physical_gap_mean=(joined['optimizer_cost'].float()-joined['counts'].float()).mean(),
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
            selector=('History-conditioned residual Value exploration under one shadow-price budget'
                      if c.history_conditioned_exploration else
                      'shared History/Value dual budget with physical History-current-Rep overlap accounting'),
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
