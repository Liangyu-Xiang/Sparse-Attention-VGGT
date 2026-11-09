"""Native four states and costs; blend two residual compensation estimates."""
import os
import torch,triton
from .current_probe_states import _base,_historical

def select(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations,*,audit=None):
 from . import reuse_inference_ops as rule
 from .beta_unified_importance import CompactPrior
 from saf3r.triton import beta_compact_prior as compact
 from .rep_history_loss import _repair,_weights_A
 from .rep_history_prior_weights import weights_A
 from saf3r.triton.reuse_compact import budget
 is_compact=isinstance(prior,CompactPrior)
 g=pi.shape[0];n=len(labels);f=len(reps);w=geometry['probe_member_width']
 device=pi.device
 def mark(name):
  if audit is not None and 'events' in audit:
   event=torch.cuda.Event(enable_timing=True)
   event.record()
   audit['events'][name]=event
 mark('start')
 errors=torch.empty((g,f,4),device=device);costs=torch.empty_like(errors);objective=torch.empty_like(errors)
 if is_compact:
  stats=compact.cluster_statistics(prior,geometry['members'],labels,geometry['norm2'],geometry['rep_distance'])
  mass,drop,rep=stats.unbind(-1)
  pi.copy_(mass)
  errors[:,:,0]=drop;errors[:,:,1]=rep;errors[:,:,2]=float('inf');errors[:,:,3]=0.
  costs[:,:,0]=0.;costs[:,:,1]=1.;costs[:,:,2]=0.;costs[:,:,3]=population
  torch.sub(errors,rep.unsqueeze(-1),out=objective)
  context,kr,hm,rep_prior=compact.gate(prior,pi,population,labels,reps,geometry['rep_distance'],
                                        tiled_count=os.environ.get('BETA_TILED_GATE_COUNT')=='1')
  amount=context['eligible_count']
 else:
  _base[(g,triton.cdiv(f,4))](prior,geometry['members'],geometry['norm2'],geometry['rep_distance'],population,pi,errors,costs,objective,n,f,w,4,num_warps=4,enable_fp_fusion=False)
  context=rule.gate(prior,pi,population,old_ids,labels,reps);context['remove_rep']=torch.zeros_like(context['remove_rep']);amount=context['eligible_count']
  kr=torch.zeros_like(pi);hm=torch.zeros_like(pi)
  _historical[(g,triton.cdiv(old_ids.shape[1],256))](old_ids,prior,labels,geometry['rep_distance'],context['qualified'],kr,hm,n,f,old_ids.shape[1],256,num_warps=4,enable_fp_fusion=False)
 new_amount=torch.empty_like(amount)
 _repair[(g,triton.cdiv(f,256))](errors,costs,objective,pi,kr,kr,amount,context['observed_rep'],reps,population,context['qualified'],new_amount,n,f,256,1,num_warps=4,enable_fp_fusion=False)
 context['eligible_count']=new_amount
 mark('state_losses_done')
 if audit is not None:audit.update(cluster_mass=pi,errors=errors,costs=costs,qualified=context['qualified'])
 selected,_=budget(objective,costs,errors[:,:,0].amax(1)+1e-12,cap,iterations);selected=rule.decode(selected)
 mark('budget_done')
 ids,real,record=rule.pack(selected,None,reps,labels,population,cap,context,tiled=os.environ.get("BETA_TILED_PACK")=="1")
 mark('packing_done')
 weights=torch.empty(ids.shape,device=ids.device)
 population_weights=torch.empty_like(weights)
 _weights_A[(g,triton.cdiv(cap,256))](ids,labels,reps,selected,population,new_amount,population_weights,n,f,cap,256,num_warps=4)
 if is_compact:compact.weights(ids,labels,reps,selected,rep_prior,pi,hm,context['observed_rep'],weights)
 else:weights_A(ids,labels,reps,selected,prior,pi,hm,context['observed_rep'],weights)
 weights=(weights*population_weights).sqrt()
 context['mass_error']=torch.zeros(g,device=device);record.update(rule.diagnostics(errors,selected,None,old_ids,labels,reps,context));record['history_cap_discarded']=torch.zeros(g,device=device)
 mark('compensation_done')
 if audit is not None:audit['state']=selected
 return ids,real,weights,record
