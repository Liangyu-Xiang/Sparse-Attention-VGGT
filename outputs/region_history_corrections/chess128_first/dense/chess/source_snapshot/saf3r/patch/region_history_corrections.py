"""History-first region Attention fixes, plus explicitly diagnostic oracle modes.

The deployable policy never reads Dense member logits: hard history, real-member
population bias, and optional query-dependent refinement INSIDE selected cells.
Exact mass / Dense-best-member require diagnostic inputs, never implicit oracles.
"""
import math
import torch
from .region_history_attention import (
    pooled_group_probabilities,table_from_labels,select_by_parent,
    install_region_history_attention)
from .residual_history_av import pack_mask


def group_probabilities(q,kfine,children,eligible,labels,group_of_fine,mode,diagnostic_logits=None):
    if mode=='mean':return pooled_group_probabilities(q,kfine,children,eligible)
    safe=children.clamp_min(0)
    active=eligible[:,safe]&(children[None]>=0)
    count=active.sum(2)
    if mode=='size_bias':
        pooled=torch.einsum('rgc,hgcd->hrgd',active.float(),kfine[:,safe].float())
        pooled/=count.clamp_min(1)[None,:,:,None]
        population=torch.bincount(labels,minlength=kfine.shape[1]).float()
        size=(active*population[safe][None]).sum(2)
        z=(q.float()[:,:,None]*pooled).sum(-1)*q.shape[-1]**-.5
        z+=size.clamp_min(1).log()[None]
        return z.masked_fill(count[None]==0,-torch.inf).softmax(-1).nan_to_num().mean(0),count
    if mode!='exact_mass':raise ValueError('Invalid group scoring')
    if diagnostic_logits is None:raise ValueError('Exact mass is diagnostic-only and needs real-member Dense logits')
    token_active=eligible[:,labels]
    z=diagnostic_logits.float().masked_fill(~token_active[None],-torch.inf)
    maximum=z.amax(-1,keepdim=True)
    maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum))
    evidence=(z-maximum).exp()
    mass=torch.zeros((*z.shape[:2],len(children)),device=z.device,dtype=torch.float32)
    group_ids=group_of_fine[labels]
    mass.scatter_add_(2,group_ids[None,None].expand_as(z),evidence)
    # Exactly equivalent to per-head softmax(logsumexp(member logits)),
    # with normalization over the current eligible support, then head mean.
    return (mass/mass.sum(-1,keepdim=True).clamp_min(1e-30)).mean(0),count


def materialize_cells(q,kpatch,labels,reps,selected,retained,terminal,
                      diagnostic_mean_weights=None,pair_batch=1024):
    final=retained.clone();rr,ff=selected.nonzero(as_tuple=True)
    probes=torch.zeros(len(selected),device=q.device,dtype=torch.long)
    if terminal=='representative':final[rr,reps[ff]]=True;return final,probes
    if terminal not in ['local_top1','best_member']:raise ValueError('Invalid terminal policy')
    if terminal=='best_member' and diagnostic_mean_weights is None:
        raise ValueError('Best-member oracle needs Dense per-head-softmax-then-mean probabilities')
    members=table_from_labels(labels)
    for start in range(0,len(rr),pair_batch):
        r,f=rr[start:start+pair_batch],ff[start:start+pair_batch]
        ids=members[f];valid=ids>=0;safe=ids.clamp_min(0)
        if terminal=='best_member':scores=diagnostic_mean_weights[r[:,None],safe].masked_fill(~valid,-torch.inf)
        else:
            z=(q[:,r,None].float()*kpatch[:,safe].float()).sum(-1)*q.shape[-1]**-.5
            scores=z.masked_fill(~valid[None],-torch.inf).softmax(-1).mean(0)
            probes.index_add_(0,r,valid.sum(1))
        chosen=ids.gather(1,scores.argmax(1)[:,None]).squeeze(1)
        final[r,chosen]=True
    return final,probes


def select_regions_corrected(q,kpatch,layout,coarse_prob,token_coarse,history,budget,variant='A',
        history_policy='hard',group_scoring='size_bias',terminal='local_top1',
        diagnostic_logits=None,diagnostic_mean_weights=None,details=None):
    labels,reps,parent,child=layout;rows=q.shape[1];nf=len(reps)
    if history_policy not in ['hard','gate']:raise ValueError('Invalid history policy')
    retained=torch.zeros((rows,len(labels)),device=q.device,dtype=torch.bool)
    if history is not None and history.numel():
        valid=history>=0;safe=history.long().clamp_min(0)
        if history_policy=='gate':
            gate=torch.zeros_like(coarse_prob,dtype=torch.bool)
            order=coarse_prob.argsort(dim=1,descending=True,stable=True)
            gate.scatter_(1,order[:,:math.ceil(coarse_prob.shape[1]*.5)],True)
            valid&=gate.gather(1,token_coarse[safe])
        r,c=valid.nonzero(as_tuple=True);retained[r,safe[r,c]]=True
    used=torch.zeros((rows,nf),device=q.device,dtype=torch.long)
    used.scatter_add_(1,labels.expand(rows,-1),retained.long())
    inherited=retained.sum(1);target=torch.maximum(inherited,torch.full_like(inherited,budget))
    missing=target-inherited;eligible=(used==0)&(missing[:,None]>0)
    if bool((eligible.sum(1)<missing).any()):raise ValueError('Insufficient uncovered fine cells')
    p1,c1=group_probabilities(q,kpatch[:,reps],table_from_labels(parent),eligible,labels,parent,
        group_scoring,diagnostic_logits)
    probes=c1.gt(0).sum(1)
    if variant=='A':selected=select_by_parent(p1,parent,eligible,missing)
    elif variant=='B':
        order=p1.argsort(dim=1,descending=True,stable=True);counts=c1.gather(1,order)
        desired=torch.maximum((eligible.sum(1).float()*.5).ceil().long(),missing)
        before=counts.cumsum(1)-counts
        chosen=torch.zeros_like(c1,dtype=torch.bool).scatter_(1,order,(before<desired[:,None])&(counts>0))
        second=eligible&chosen[:,parent]
        p2,c2=group_probabilities(q,kpatch[:,reps],table_from_labels(child),second,labels,child,
            group_scoring,diagnostic_logits)
        probes+=c2.gt(0).sum(1)
        selected=select_by_parent(p2,child,second,missing)
    else:raise ValueError('Invalid variant')
    final,terminal_probes=materialize_cells(q,kpatch,labels,reps,selected,retained,terminal,diagnostic_mean_weights)
    assert bool((final.sum(1)==target).all())
    assert not bool((selected&(used>0)).any())
    if details is not None:details.update(retained=retained,selected_fine=selected)
    return pack_mask(final),dict(history_retained=inherited,explored=missing,
        intermediate_probes=probes,history_covered_fine=(used>0).sum(1),terminal_member_qk=terminal_probes)


def influence_scores(probabilities,values,dense_output):
    """Mean-head exact SINGLE-DELETION influence; not optimal subset selection.

    Inputs: HxRxK probabilities, HxKxD values, HxRxD Dense outputs.
    Avoid the HxRxKxD distance tensor using squared-norm expansion.
    """
    distance=(dense_output.float().square().sum(-1)[:,:,None]
        +values.float().square().sum(-1)[:,None]
        -2*(dense_output.float()@values.float().transpose(-1,-2))).clamp_min(0).sqrt()
    return (probabilities/(1-probabilities).clamp_min(1e-12)*distance).mean(0)


def install_corrected_region_history_attention(model,variant='A',terminal='local_top1',**kwargs):
    """Deployable (no Dense oracle). Keep legacy installer for historical replay.

    Terminal local_top1 does extra actual QK within selected2–6-token cells.
    These probes, including repeated winning QK at final execution, are NOT free.
    """
    if terminal not in ['representative','local_top1']:raise ValueError('Oracle terminal is not deployable')
    state=install_region_history_attention(model,variant=variant,**kwargs)
    state.coarse_size_bias=True
    def selector(*args,**kwargs):
        return select_regions_corrected(*args,**kwargs,history_policy='hard',group_scoring='size_bias',terminal=terminal)
    state.selector=selector
    state.policy_revision='hard_history_size_bias_'+terminal
    return state
