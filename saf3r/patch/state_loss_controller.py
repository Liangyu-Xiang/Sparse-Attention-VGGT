"""Consistent support/weight losses; no current member QK is an input."""
import torch
import triton
from .state_loss_reference import output_sketch


GROUP_VARIANTS=('group_centroid','group_centered','group_mixture','group_moment','group_moment_variance')
RESIDUAL_VARIANTS=('residual_energy','centered_residual_energy','transport_moment')


def prepare_geometry(v,module,patches,specials,labels,reps,population,config):
    y=output_sketch(v,module,patches,config.state_loss_sketch_dim)
    width=triton.next_power_of_2(int(population.max()))
    if width>512:raise ValueError('Loss controller supports at most512 members/cluster')
    order=labels.argsort(stable=True);offset=population.long().cumsum(0)-population.long()
    members=torch.full((len(reps),width),-1,device=v.device,dtype=torch.int32)
    members[labels[order],torch.arange(len(labels),device=v.device)-offset[labels[order]]]=order.int()
    return dict(width=width,members=members,shared_value_loss=True,
        state_loss_variant=config.state_loss_variant,loss_operator=config.state_loss_operator,
        loss_gate=config.state_loss_history_gate,loss_history_stride=config.history_query_interval//32,
        loss_patch_values=y[patches].contiguous(),loss_special_values=y[specials].contiguous())


def select_window(prior,pi,old_ids,allowed,reps,labels,population,geometry,cap,iterations,special_p):
    from saf3r.triton import state_loss_controller as kernel
    from saf3r.triton.reuse_compact import budget
    from . import reuse_inference_ops as rule
    context=kernel.gate(prior,pi,old_ids,labels,reps,population,geometry['loss_gate']=='observed')
    stats=kernel.statistics(prior,pi,labels,reps,population,geometry,context)
    errors,costs,objective=kernel.state_losses(prior,pi,allowed,population,reps,geometry,stats,special_p)
    selected,_=budget(objective,costs,errors[:,:,0].amax(1)+1e-12,cap,iterations)
    selected=rule.decode(selected)
    original_amount=context['eligible_count']
    if geometry['loss_operator']=='residual_rep':
        union=context['qualified'].clone()
        union.scatter_(1,reps.long()[None].expand(len(prior),-1),True)
        context=dict(context,qualified=union,
            eligible_count=original_amount+(~context['observed_rep']).int())
    ids,real,record=rule.pack(selected,None,reps,labels,population,cap,context)
    weights,reused=kernel.execution_weights(selected,ids,prior,pi,labels,reps,population,geometry,context,stats)
    context['mass_error']=torch.zeros(len(prior),device=prior.device)
    record.update(rule.diagnostics(errors,selected,None,old_ids,labels,reps,context))
    record['history_cap_discarded']=torch.zeros(len(prior),device=prior.device)
    history=selected==5
    if geometry['loss_operator']=='residual_rep':
        record['representatives']=(selected==1).sum(1)+history.sum(1)
        record['reuse_edges']=((context['eligible_count']-1)*history).sum(1)
        record['selected_history_keys']=record['reuse_edges']
    record['actual_history_reused_keys']=reused
    record['actual_history_key_reuse_fraction']=reused.float()/record['counts'].clamp_min(1)
    record['history_candidates']=original_amount.sum(1)
    return ids,real,weights,record
