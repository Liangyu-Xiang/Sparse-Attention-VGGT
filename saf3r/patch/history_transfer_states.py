"""Native four states and costs; blend two residual compensation estimates."""
import torch,triton
from .current_probe_states import _base,_historical

def select(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations):
 from . import reuse_inference_ops as rule
 from .rep_history_loss import _repair,_weights_A
 from .rep_history_prior_weights import weights_A
 from saf3r.triton.reuse_compact import budget
 g,n=prior.shape;f=len(reps);w=geometry['probe_member_width']
 errors=torch.empty((g,f,4),device=prior.device);costs=torch.empty_like(errors);objective=torch.empty_like(errors)
 _base[(g,triton.cdiv(f,4))](prior,geometry['members'],geometry['norm2'],geometry['rep_distance'],population,pi,errors,costs,objective,n,f,w,4,num_warps=4,enable_fp_fusion=False)
 context=rule.gate(prior,pi,population,old_ids,labels,reps);context['remove_rep']=torch.zeros_like(context['remove_rep']);amount=context['eligible_count']
 kr=torch.zeros_like(pi);hm=torch.zeros_like(pi)
 _historical[(g,triton.cdiv(old_ids.shape[1],256))](old_ids,prior,labels,geometry['rep_distance'],context['qualified'],kr,hm,n,f,old_ids.shape[1],256,num_warps=4,enable_fp_fusion=False)
 new_amount=torch.empty_like(amount)
 _repair[(g,triton.cdiv(f,256))](errors,costs,objective,pi,kr,kr,amount,context['observed_rep'],reps,population,context['qualified'],new_amount,n,f,256,1,num_warps=4,enable_fp_fusion=False)
 context['eligible_count']=new_amount
 selected,_=budget(objective,costs,errors[:,:,0].amax(1)+1e-12,cap,iterations);selected=rule.decode(selected)
 ids,real,record=rule.pack(selected,None,reps,labels,population,cap,context);weights=torch.empty(ids.shape,device=ids.device)
 population_weights=torch.empty_like(weights)
 _weights_A[(g,triton.cdiv(cap,256))](ids,labels,reps,selected,population,new_amount,population_weights,n,f,cap,256,num_warps=4)
 weights_A(ids,labels,reps,selected,prior,pi,hm,context['observed_rep'],weights)
 weights=(weights*population_weights).sqrt()
 context['mass_error']=torch.zeros(g,device=prior.device);record.update(rule.diagnostics(errors,selected,None,old_ids,labels,reps,context));record['history_cap_discarded']=torch.zeros(g,device=prior.device)
 return ids,real,weights,record
