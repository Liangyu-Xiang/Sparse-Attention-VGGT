"""History-state-only changes on frozen shared-RMS Drop/squared-L2 baseline.

No changes to RMS scaling, Drop, clustering, query groups, budget, Exact
eligibility, attention backend or D2+M5c. State5 now emits ALL qualified history.
"""
import torch
REP_REMOVAL="historical_rep"


def gate(prior,pi,population,old_ids,labels,reps,rep_removal=None):
    g,n=prior.shape;f=len(reps);valid=old_ids>=0;safe=old_ids.long().clamp_min(0);cluster=labels[safe]
    count=torch.zeros((g,f),device=prior.device,dtype=torch.int32).scatter_add_(1,cluster,valid.int())
    seen=torch.zeros_like(count).scatter_add_(1,cluster,(valid&(old_ids==reps[cluster])).int())>0
    mode=REP_REMOVAL if rep_removal is None else rep_removal
    if mode not in ('any','qualified','historical_rep'):raise ValueError('Unknown Rep overlap rule')
    above=prior.gather(1,safe)>(pi/population[None]).gather(1,cluster)
    # User clarification: an observed current Rep is a historical Key even
    # when its estimated probability does not exceed the uniform threshold.
    eligible=valid&(above|((old_ids==reps[cluster]) if mode=='historical_rep' else torch.zeros_like(valid)))
    amount=torch.zeros_like(count).scatter_add_(1,cluster,eligible.int())
    selected=torch.zeros((g,n+1),device=prior.device,dtype=torch.bool)
    selected.scatter_(1,safe.masked_fill(~eligible,n),eligible)
    remove=selected[:,:n][:,reps] if mode=='qualified' else seen
    return dict(qualified=selected[:,:n].contiguous(),eligible_count=amount,observed_rep=seen,remove_rep=remove,history_count=count)


def states(pi,z,reps,labels,population,hids,allowed,prior,drop_multiplier,geometry,context):
    from .reuse_cosine_distortion import cosine_states
    from saf3r.triton.zdrop_history_states import history_loss
    if geometry is None or not geometry.get('l2',False) or 'drop_values' not in geometry:
        raise RuntimeError('Original RMS Drop and squared-L2 geometry required')
    # Keep the ORIGINAL Drop/Rep/Exact losses and cost conventions.
    errors,costs=cosine_states(pi,z,reps,labels,population,hids,allowed,prior,drop_multiplier,geometry=geometry)
    er=errors[:,:,1].clone();drop=errors[:,:,0].clone();exact=errors[:,:,10].clone()
    hist,owners=history_loss(prior,labels,geometry,context['qualified'],len(reps))
    errors.fill_(float('inf'));errors[:,:,0]=drop
    errors[:,:,1]=er.masked_fill(context['remove_rep'],float('inf'))
    errors[:,:,5]=hist.masked_fill(context['eligible_count']==0,float('inf'))
    errors[:,:,10]=exact  # Coexist with History; original allowed mask unchanged.
    costs[:,:,5]=context['eligible_count'].float()
    geometry['objective']=errors-er[:,:,None]
    context['owners']=owners
    return errors,costs


def pack(selected,hids,reps,labels,population,budget,context):
    from saf3r.triton.zdrop_history_states import _emit
    import triton
    g,f=selected.shape;n=len(labels);j=torch.arange(n,device=selected.device)
    state=selected[:,labels]
    flag=((state==10)|((state==1)&(j[None]==reps[labels][None]))|((state==5)&context['qualified'])).int().contiguous()
    prefix=flag.cumsum(1,dtype=torch.int32);counts=prefix[:,-1]
    expected=torch.where(selected==1,1,torch.where(selected==5,context['eligible_count'],torch.where(selected==10,population[None],0))).sum(1)
    if bool(((counts>budget)|(counts!=expected)).any()):raise RuntimeError('History state support/count/budget mismatch')
    if bool(((selected==1)&context['remove_rep']).any()):raise RuntimeError('Overlapping independent Rep selected')
    legal=(selected==0)|(selected==1)|(selected==5)|(selected==10)
    if not bool(legal.all()):raise RuntimeError('Obsolete prefix or mixed state selected')
    ids=torch.full((g,budget),-1,device=selected.device,dtype=torch.int32)
    _emit[(g,triton.cdiv(n,256))](flag,prefix,ids,n,budget,256,num_warps=4)
    return ids,ids>=0,dict(reuse_edges=(context['eligible_count']*(selected==5)).sum(1),representatives=(selected==1).sum(1),exact_edges=(population[None]*(selected==10)).sum(1),dropped_clusters=(selected==0).sum(1),counts=counts,native_cost_count_error=(counts-expected).abs())


def transfer(selected,labels,reps,hids,prior,pi,packed,geometry,context):
    g,n=prior.shape;j=torch.arange(n,device=prior.device)[None].expand(g,-1);state=selected[:,labels]
    destination=torch.where(state==10,j,torch.where(state==1,reps[labels][None],torch.where(state==5,context['owners'],-1))).long()
    covered=destination>=0
    mass=torch.zeros((g,n+1),device=prior.device)
    mass.scatter_add_(1,destination.masked_fill(~covered,n),prior.masked_fill(~covered,0.))
    safe=packed.long().clamp_min(0);valid=packed>=0
    own=prior.gather(1,safe);assigned=mass[:,:n].gather(1,safe).masked_fill(~valid,0.)
    error=(assigned.sum(1)-prior.masked_fill(~covered,0.).sum(1)).abs()
    if bool((error>1e-5).any()):raise RuntimeError('History representation mass transfer not conserved')
    context['mass_error']=error
    return (assigned/own.clamp_min(1e-30)).clamp_min(1.).masked_fill(~valid,0.).contiguous()


def diagnostics(errors,selected,hids,old_ids,labels,reps,context):
    g=len(selected);record={f'state_{i}_clusters':(selected==i).sum(1) for i in range(11)}
    record.update(observed_history=context['history_count'].sum(1),history_rep_overlap=context['observed_rep'].sum(1),history_candidates=context['eligible_count'].sum(1),qualified_history_clusters=(context['eligible_count']>0).sum(1),removed_rep_clusters=context['remove_rep'].sum(1),selected_history_keys=(context['eligible_count']*(selected==5)).sum(1),history_exact_selected_clusters=((selected==10)&(context['eligible_count']>0)).sum(1),selected_rep_observed_history_clusters=((selected==1)&context['observed_rep']).sum(1),history_mass_transfer_error=context['mass_error'])
    return record
