"""Dead-work cleanup only: four ordered states, unchanged mathematical method."""
import torch
from .zdrop_history_states import gate,pack,transfer
_STATE_IDS={}


def states(pi,z,reps,labels,population,hids,allowed,prior,drop_multiplier,geometry,context):
    from saf3r.triton.zdrop_history_cleanup import rep_loss
    from saf3r.triton.zdrop_history_states import history_loss
    if geometry is None or not geometry.get('l2',False) or 'drop_values' not in geometry:raise RuntimeError('Original squared-L2 and RMS Drop required')
    er=rep_loss(prior,labels,geometry,len(reps))
    hist,owner=history_loss(prior,labels,geometry,context['qualified'],len(reps))
    # Original tie order 0,1,5,10 is retained in compact positions 0,1,2,3.
    errors=torch.stack((geometry['drop_values'],er.masked_fill(context['remove_rep'],float('inf')),hist.masked_fill(context['eligible_count']==0,float('inf')),torch.where(allowed,0.,float('inf'))),-1)
    costs=torch.stack((torch.zeros_like(pi),torch.ones_like(pi),context['eligible_count'].float(),population[None].expand_as(pi)),-1)
    geometry['objective']=errors-er[:,:,None];context['owners']=owner
    return errors,costs


def decode(selected):
    key=(str(selected.device),selected.dtype)
    if key not in _STATE_IDS:_STATE_IDS[key]=torch.tensor([0,1,5,10],device=selected.device,dtype=selected.dtype)
    return _STATE_IDS[key][selected]


def diagnostics(errors,selected,hids,old_ids,labels,reps,context):
    g=len(selected);record={f'state_{i}_clusters':(selected==i).sum(1) for i in [0,1,5,10]}
    zero=torch.zeros(g,device=selected.device)
    for i in [2,3,4,6,7,8,9]:record[f'state_{i}_clusters']=zero
    record.update(observed_history=context['history_count'].sum(1),history_rep_overlap=context['observed_rep'].sum(1),history_candidates=context['eligible_count'].sum(1),qualified_history_clusters=(context['eligible_count']>0).sum(1),removed_rep_clusters=context['remove_rep'].sum(1),selected_history_keys=(context['eligible_count']*(selected==5)).sum(1),history_exact_selected_clusters=((selected==10)&(context['eligible_count']>0)).sum(1),selected_rep_observed_history_clusters=((selected==1)&context['observed_rep']).sum(1),history_mass_transfer_error=context['mass_error'])
    return record
