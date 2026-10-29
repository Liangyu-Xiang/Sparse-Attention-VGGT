"""History-conditioned tokenwise cosine distortion and estimated mass transfer.

This is a representation objective, not a Dense-output-error oracle. All current
positive genuine historical observations inform the prior; the existing top-H
candidate cap only limits selectable historical representatives.
"""
import torch
import torch.nn.functional as F


def normalized_values(features):
    return F.normalize(features.float(),dim=-1,eps=1e-12).contiguous()


def history_prior(pi,labels,population,old_ids,old_weights,beta,grid,floor_fraction=1e-6):
    """Cluster mass pi, observed historical masses, uniform unknown residual.

    Overfull evidence is scaled down; if all cluster members are observed,
    their evidence is normalized to pi. A declared numerical full-support mix
    (1-eps)*prior + eps*pi/n permits finite transfer to a zero-prior candidate.
    No test truth enters this calculation.
    """
    if not 0<floor_fraction<=1e-3:raise ValueError('Invalid numerical prior floor')
    g,f=pi.shape;p=len(labels)
    count=torch.zeros_like(pi);total=torch.zeros_like(pi)
    if old_ids is not None:
        safe=old_ids.long().clamp_min(0)
        raw=old_weights.float()*beta.gather(1,grid[safe])
        valid=(old_ids>=0)&(raw>0)
        cluster=labels[safe]
        count.scatter_add_(1,cluster,valid.float())
        total.scatter_add_(1,cluster,raw.masked_fill(~valid,0))
        unknown=(population[None]-count).clamp_min(0)
        factor=torch.minimum(torch.ones_like(pi),pi/total.clamp_min(1e-30))
        factor=torch.where((unknown==0)&(total>0),pi/total.clamp_min(1e-30),factor)
        used=total*factor
        uniform=(pi-used).clamp_min(0)/unknown.clamp_min(1)
        prior=uniform[:,labels].clone()
        padded=torch.cat((prior,torch.zeros((g,1),device=pi.device)),1)
        index=torch.where(valid,safe,p)
        padded.scatter_(1,index,(raw*factor.gather(1,cluster)).masked_fill(~valid,0))
        prior=padded[:,:p]
    else:prior=pi[:,labels]/population[labels][None]
    prior=(1-floor_fraction)*prior+floor_fraction*pi[:,labels]/population[labels][None]
    return prior.contiguous()


def _costs_and_mask(pi,population,hids,exact_allowed,errors):
    limit=hids.shape[-1];g,f=pi.shape
    prefix=(hids>=0).int().cumprod(-1).bool()
    errors[:,:,0]=2*pi
    cost=[torch.zeros_like(pi),torch.ones_like(pi)]
    for t in range(1,limit+1):
        errors[:,:,1+t].masked_fill_(~prefix[:,:,t-1],torch.inf)
        cost.append(torch.full_like(pi,t))
    for t in range(1,limit+1):
        valid=prefix[:,:,t-1]&(population[None]>t)
        errors[:,:,1+limit+t].masked_fill_(~valid,torch.inf)
        cost.append(torch.full_like(pi,t+1))
    errors[:,:,-1]=torch.where(exact_allowed,torch.zeros_like(pi),torch.full_like(pi,torch.inf))
    cost.append(population[None].expand(g,-1).float())
    return errors,torch.stack(cost,-1)


def cosine_states(pi,z,reps,labels,population,hids,exact_allowed,prior,drop_multiplier=2.):
    g,f=pi.shape;limit=hids.shape[-1]
    if z.is_cuda:
        from saf3r.triton.reuse_cosine_distortion import distortion_losses
        errors=distortion_losses(z,labels,reps,hids,prior,f)
    else:
        errors=torch.zeros((g,f,2*limit+3),device=z.device)
        for group in range(g):
            for cluster in range(f):
                members=torch.where(labels==cluster)[0];a=prior[group,members]
                rep=reps[cluster]
                dr=(1-(z[members]*z[rep]).sum(-1)).clamp(0,2).masked_fill(members==rep,0)
                errors[group,cluster,1]=(a*dr).sum()
                minimum=torch.full_like(a,2.);chosen=torch.zeros_like(members,dtype=torch.bool)
                for slot in range(limit):
                    h=hids[group,cluster,slot]
                    if h>=0:
                        dist=(1-(z[members]*z[h]).sum(-1)).clamp(0,2).masked_fill(members==h,0)
                        minimum=torch.minimum(minimum,dist);chosen|=members==h
                    errors[group,cluster,2+slot]=(a*minimum).sum()
                    errors[group,cluster,2+limit+slot]=(a*dr.masked_fill(chosen,0)).sum()
    errors,costs=_costs_and_mask(pi,population,hids,exact_allowed,errors)
    errors[:,:,0]=drop_multiplier*pi
    return errors,costs


def relative_logit_prior(probability,grid,grid_sizes,old_ids,delta,labels,clusters):
    """Calibrate compact previous raw-logit residuals within CURRENT grids.

    Unobserved keys have likelihood ratio one. Each grid retains exactly its
    current coarse mass, even when a fine cluster spans multiple grids.
    delta is mean_{head,query}(raw QK - raw coarse QK), not compensated logits.
    """
    g=len(probability);p=len(grid)
    denominator=grid_sizes[None].expand(g,-1).clone()
    if old_ids is not None:
        safe=old_ids.long().clamp_min(0);valid=old_ids>=0
        ratio=delta.float().clamp(-30,30).exp()
        denominator.scatter_add_(1,grid[safe],(ratio-1).masked_fill(~valid,0))
    base=probability/denominator.clamp_min(1e-30)
    prior=base[:,grid].clone()
    if old_ids is not None:
        padded=torch.cat((prior,torch.zeros((g,1),device=prior.device)),1)
        padded.scatter_(1,torch.where(valid,safe,p),(base.gather(1,grid[safe])*ratio).masked_fill(~valid,0))
        prior=padded[:,:p].contiguous()
    pi=torch.zeros((g,clusters),device=prior.device).scatter_add_(1,labels[None].expand(g,-1),prior)
    return prior,pi


def transfer_weights(selected,z,reps,labels,hids,prior,packed_ids):
    """Assign each represented token to an emitted Key, then transfer its mass.

    Hist-only: nearest selected historical direction, with self ownership for
    selected Keys. Hist+Rep: each historical Key owns itself, Rep owns the rest.
    Execution multiplier = assigned estimated mass / own estimated mass >=1.
    Thus prior[key]*multiplier equals the transferred mass. Final current QK
    still modulates it; this does not force true Dense attention probabilities.
    """
    g,f=selected.shape;limit=hids.shape[-1];p=len(labels)
    if z.is_cuda:
        from saf3r.triton.reuse_cosine_distortion import transferred_masses
        masses=transferred_masses(z,labels,reps,hids,prior,selected)
    else:
        masses=torch.zeros((g,p),device=z.device)
        for group in range(g):
            for cluster in range(f):
                state=int(selected[group,cluster]);members=torch.where(labels==cluster)[0]
                if state==0:continue
                if state==2*limit+2:
                    masses[group,members]=prior[group,members];continue
                if state==1:destination=reps[cluster].expand_as(members)
                elif state<=limit+1:
                    keys=hids[group,cluster,:state-1]
                    d=(1-z[members]@z[keys].T).clamp(0,2)
                    d=d.masked_fill(members[:,None]==keys[None],-1)
                    destination=keys[d.argmin(-1)]
                else:
                    keys=hids[group,cluster,:state-limit-1]
                    destination=reps[cluster].expand_as(members).clone()
                    for key in keys:destination=torch.where(members==key,key,destination)
                masses[group].scatter_add_(0,destination,prior[group,members])
    safe=packed_ids.long().clamp_min(0)
    own=prior.gather(1,safe);assigned=masses.gather(1,safe)
    weights=(assigned/own.clamp_min(1e-30)).clamp_min(1)
    weights.masked_fill_(packed_ids<0,0)
    return weights.contiguous(),masses
