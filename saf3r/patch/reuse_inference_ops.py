"""Execution-only fusion; validations deferred to one check per global layer."""
import torch,triton
from . import zdrop_history_states as reference
from .zdrop_history_cleanup import states,decode,diagnostics as original_diagnostics
from saf3r.triton import reuse_inference_ops as kernels
from saf3r.triton.zdrop_history_states import _emit

def gate(prior,pi,population,old_ids,labels,reps,rep_removal=None):
    if rep_removal not in (None,'historical_rep') or reference.REP_REMOVAL!='historical_rep':
        return reference.gate(prior,pi,population,old_ids,labels,reps,rep_removal)
    g,n=prior.shape;f=len(reps)
    count=torch.zeros((g,f),device=prior.device,dtype=torch.int32);seen=torch.zeros_like(count);amount=torch.zeros_like(count)
    qualified=torch.zeros((g,n),device=prior.device,dtype=torch.bool)
    kernels._gate[(g,triton.cdiv(old_ids.shape[1],256))](old_ids,prior,pi,population,labels,reps,count,seen,amount,qualified,n,f,old_ids.shape[1],256,num_warps=4,enable_fp_fusion=False)
    seen=seen>0
    return dict(qualified=qualified,eligible_count=amount,observed_rep=seen,remove_rep=seen,history_count=count)

def pack(selected,hids,reps,labels,population,budget,context):
    g,f=selected.shape;n=len(labels);flags=torch.empty((g,n),device=selected.device,dtype=torch.int32)
    kernels._flags[(g,triton.cdiv(n,256))](selected,labels,reps,context['qualified'],flags,n,f,256,num_warps=4)
    prefix=flags.cumsum(1,dtype=torch.int32);counts=prefix[:,-1];tiles=triton.cdiv(f,256)
    partial=torch.empty((g,tiles),device=selected.device,dtype=torch.int32);invalid=torch.empty_like(partial)
    kernels._validate[(g,tiles)](selected,context['eligible_count'],population,context['remove_rep'],partial,invalid,f,tiles,256,num_warps=4)
    expected=partial.sum(1).float()
    context['pack_error']=(counts>budget)|(counts!=expected)|(invalid.sum(1)>0)
    ids=torch.full((g,budget),-1,device=selected.device,dtype=torch.int32)
    _emit[(g,triton.cdiv(n,256))](flags,prefix,ids,n,budget,256,num_warps=4)
    return ids,ids>=0,dict(reuse_edges=(context['eligible_count']*(selected==5)).sum(1),representatives=(selected==1).sum(1),exact_edges=(population[None]*(selected==10)).sum(1),dropped_clusters=(selected==0).sum(1),counts=counts,native_cost_count_error=(counts-expected).abs())

def transfer(selected,labels,reps,hids,prior,pi,packed,geometry,context):
    g,n=prior.shape;f=len(reps);a=packed.shape[1];nt=triton.cdiv(n,256);at=triton.cdiv(a,256)
    mass=torch.zeros((g,n+1),device=prior.device);covered=torch.empty((g,nt),device=prior.device);assigned=torch.empty((g,at),device=prior.device)
    kernels._mass[(g,nt)](selected,labels,reps,context['owners'],prior,mass,covered,n,f,nt,256,num_warps=4)
    output=torch.empty((g,a),device=prior.device)
    kernels._weights[(g,at)](packed,prior,mass,output,assigned,n,a,at,256,num_warps=4,enable_fp_fusion=False)
    context['mass_error']=(assigned.sum(1)-covered.sum(1)).abs()
    return output

def diagnostics(errors,selected,hids,old_ids,labels,reps,context):
    result=original_diagnostics(errors,selected,hids,old_ids,labels,reps,context)
    result['implementation_validation_error']=context.get('pack_error',torch.zeros(len(selected),device=selected.device,dtype=torch.bool))|(context['mass_error']>1e-5)
    return result
