"""Finite algebraic logit/state-loss changes; four states and budgets unchanged."""
import torch,triton
from .previous_logit_delta import estimate as old_estimate,observe
from .one_layer_edge_history import mean_valid
VARIANTS=['baseline','drop_center','drop_weighted_center','drop050','drop025','normalizer_vector','normalizer_diag1','normalizer_diag4']

def estimate(mean,coarse,previous_coarse,previous_mean,ids,delta,grid,sizes,eligible,variant):
 if not variant.startswith('cumulant'):return old_estimate(mean,coarse,previous_coarse,previous_mean,ids,delta,grid,sizes,eligible,'centered050')
 if delta is None:return old_estimate(mean,coarse,None,None,ids,None,grid,sizes,eligible,'centered050')
 from .previous_logit_delta import correction
 h,g,p,r=mean.shape;n=len(grid);valid=ids>=0;safe=ids.clamp_min(0).long();block=grid[safe];ix=block[None,:,None].expand(h,-1,p,-1)
 changed=correction(delta,ids,grid,mean,previous_mean,torch.zeros_like(coarse),'centered050')
 count=torch.zeros(g,r,device=ids.device).scatter_add_(1,block,valid.float())
 var=torch.zeros_like(mean).scatter_add_(3,ix,changed.square().masked_fill(~valid[None,:,None],0.))/count.clamp_min(1)[None,:,None]
 fraction={'cumulant025':.25,'cumulant050':.5,'cumulant100':1.}[variant]
 changed=changed-.5*fraction*var.gather(3,ix)
 scores=mean[...,grid].clone();updated=mean.gather(3,ix)+changed;tomb=safe.masked_fill(~valid,n)[None,:,None].expand(h,-1,p,-1)
 scores=torch.cat((scores,scores.new_zeros(h,g,p,1)),-1).scatter_(3,tomb,updated.masked_fill(~valid[None,:,None],0.))[...,:n]
 return mean_valid(scores.softmax(-1),eligible).contiguous(),torch.zeros_like(coarse)

def aggregate(value,labels,f):
 shape=list(value.shape);shape[1]=f;out=value.new_zeros(shape)
 ix=labels[None].expand(value.shape[:2])
 if value.ndim==3:ix=ix[...,None].expand_as(value)
 return out.scatter_add_(1,ix,value)

def modeled_errors(prior,labels,reps,z,qualified,variant):
 """State losses after History-union-Rep eligibility; exact support error is0."""
 f=len(reps);dist=(z-z[reps[labels]]).square().sum(-1);norm=z.square().sum(-1)
 if variant in ['power15','power2']:
  w=prior.pow(1.5 if variant=='power15' else 2.)*prior.shape[1]**(.5 if variant=='power15' else 1.)
 else:w=prior
 drop=aggregate(w*norm,labels,f);rep=aggregate(w*dist,labels,f);kept=aggregate(w*dist*qualified,labels,f);hist=(rep-kept).clamp_min(0)
 if variant=='history_half':hist=(rep-.5*kept).clamp_min(0)
 if variant=='mass_bound':
  mass=aggregate(prior,labels,f);left=aggregate(prior*(~qualified),labels,f)
  drop=drop*mass;rep=rep*mass;hist=hist*left
 if variant.startswith('vector'):
  total=aggregate(prior[...,None]*z[None],labels,f);mass=aggregate(prior,labels,f)
  delta=total-mass[...,None]*z[reps][None]
  selected=aggregate((prior*qualified)[...,None]*(z-z[reps[labels]])[None],labels,f)
  drop=total.square().sum(-1);rep=delta.square().sum(-1);hist=(delta-selected).square().sum(-1)
  if variant!='vector':
   lam=1. if variant=='vector_diag1' else 4.;w=prior.square()
   drop=drop+lam*aggregate(w*norm,labels,f);rep=rep+lam*aggregate(w*dist,labels,f);hist=hist+lam*aggregate(w*dist*(~qualified),labels,f)
 if variant in ['drop050','drop025']:drop=drop*(.5 if variant=='drop050' else .25)
 if variant=='drop_center':drop=aggregate(prior*(z-z.mean(0)).square().sum(-1),labels,f)
 if variant=='drop_weighted_center' or variant.startswith('normalizer'):
  total=aggregate(prior[...,None]*z[None],labels,f);mass=aggregate(prior,labels,f);center=total.sum(1)/prior.sum(1).clamp_min(1e-30)[:,None]
  cnorm=(norm[None]-2*torch.einsum('nd,gd->gn',z,center)+center.square().sum(-1)[:,None]).clamp_min(0)
  if variant=='drop_weighted_center':drop=aggregate(prior*cnorm,labels,f)
  else:
   own=prior[:,reps];h_mass=aggregate(prior*qualified,labels,f);h_total=aggregate((prior*qualified)[...,None]*z[None],labels,f);count=aggregate(qualified.float(),labels,f);pop=aggregate(torch.ones_like(prior),labels,f)
   rep_mass=(pop*own*mass).clamp_min(0).sqrt()
   remain=(mass-h_mass+own).clamp_min(0);hist_rep_mass=((pop-count+1)*own*remain).clamp_min(0).sqrt()
   rvec=total-rep_mass[...,None]*z[reps][None]-(mass-rep_mass)[...,None]*center[:,None]
   hvec=total-h_total+(own-hist_rep_mass)[...,None]*z[reps][None]-(mass-h_mass+own-hist_rep_mass)[...,None]*center[:,None]
   dvec=total-mass[...,None]*center[:,None]
   drop=dvec.square().sum(-1);rep=rvec.square().sum(-1);hist=hvec.square().sum(-1)
   if variant!='normalizer_vector':
    lam=1. if variant=='normalizer_diag1' else 4.;w=prior.square()
    drop=drop+lam*aggregate(w*cnorm,labels,f);rep=rep+lam*aggregate(w*dist,labels,f);hist=hist+lam*aggregate(w*dist*(~qualified),labels,f)
 return torch.stack((drop,rep,hist,torch.zeros_like(rep)),-1)

def select_capture(prior,old,reps,labels,pop,geo,cap,z,variant):
 from . import current_probe_states as kernels,reuse_inference_ops as rule
 from .rep_history_loss import _repair,_weights_A
 from saf3r.triton.reuse_compact import budget
 g,n=prior.shape;f=len(reps);pi=torch.empty(g,f,device=prior.device);errors=torch.empty(g,f,4,device=prior.device);costs=torch.empty_like(errors);obj=torch.empty_like(errors)
 kernels._base[(g,triton.cdiv(f,4))](prior,geo['members'],geo['norm2'],geo['rep_distance'],pop,pi,errors,costs,obj,n,f,geo['probe_member_width'],4,num_warps=4,enable_fp_fusion=False)
 context=rule.gate(prior,pi,pop,old,labels,reps);context['remove_rep']=torch.zeros_like(context['remove_rep']);kr=torch.zeros_like(pi);hm=torch.zeros_like(pi)
 kernels._historical[(g,triton.cdiv(old.shape[1],256))](old,prior,labels,geo['rep_distance'],context['qualified'],kr,hm,n,f,old.shape[1],256,num_warps=4,enable_fp_fusion=False)
 amount=torch.empty_like(context['eligible_count'])
 _repair[(g,triton.cdiv(f,256))](errors,costs,obj,pi,kr,kr,context['eligible_count'],context['observed_rep'],reps,pop,context['qualified'],amount,n,f,256,1,num_warps=4,enable_fp_fusion=False)
 context['eligible_count']=amount
 if variant not in ['baseline','cumulant025','cumulant050','cumulant100']:
  errors=modeled_errors(prior,labels,reps,z,context['qualified'],variant);errors[:,:,2].masked_fill_(amount==0,float('inf'));obj=errors-errors[:,:,1:2]
 selected,_=budget(obj,costs,errors[:,:,0].amax(1)+1e-12,cap,16);selected=rule.decode(selected)
 ids,valid,record=rule.pack(selected,None,reps,labels,pop,cap,context);weights=torch.empty_like(ids,dtype=torch.float32)
 _weights_A[(g,triton.cdiv(cap,256))](ids,labels,reps,selected,pop,amount,weights,n,f,cap,256,num_warps=4)
 if context['pack_error'].any():raise RuntimeError('Support/budget failed')
 return dict(prior=prior,pi=pi,errors=errors,costs=costs,selected=selected,ids=ids,valid=valid,weights=weights,context=context)
