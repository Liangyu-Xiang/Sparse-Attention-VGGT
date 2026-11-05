"""Same A distortion/support/budget with a fully observed current per-key prior."""
import torch,triton
import triton.language as tl

@triton.jit
def _base(P,MEM,NORM,DR,POP,PI,E,COST,OBJ,N:tl.constexpr,F:tl.constexpr,W:tl.constexpr,C:tl.constexpr):
 g=tl.program_id(0);c=tl.program_id(1)*C+tl.arange(0,C);m=tl.arange(0,W)
 member=tl.load(MEM+c[:,None]*W+m[None,:],c[:,None]<F,-1);valid=(c[:,None]<F)&(member>=0);safe=tl.maximum(member,0)
 p=tl.load(P+g*N+safe,valid,0.);norm=tl.load(NORM+safe,valid,0.);dist=tl.load(DR+safe,valid,0.)
 mass=tl.sum(p,1);drop=tl.sum(p*norm,1);rep=tl.sum(p*dist,1);pop=tl.load(POP+c,c<F,0.);off=(g*F+c)*4
 tl.store(PI+g*F+c,mass,c<F)
 for s in tl.static_range(4):
  error=drop if s==0 else rep if s==1 else float('inf') if s==2 else 0.
  cost=0. if s==0 or s==2 else 1. if s==1 else pop
  tl.store(E+off+s,error,c<F);tl.store(COST+off+s,cost,c<F);tl.store(OBJ+off+s,error-rep,c<F)

@triton.jit
def _historical(IDS,P,LABEL,DR,QUAL,KR,HM,N:tl.constexpr,F:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B);key=tl.load(IDS+g*A+i,i<A,-1);valid=(i<A)&(key>=0);safe=tl.maximum(key,0)
 valid=valid&tl.load(QUAL+g*N+safe,valid,False);c=tl.load(LABEL+safe,valid,0);p=tl.load(P+g*N+safe,valid,0.);d=tl.load(DR+safe,valid,0.)
 tl.atomic_add(KR+g*F+c,p*d,valid,sem='relaxed');tl.atomic_add(HM+g*F+c,p,valid,sem='relaxed')

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
 if geometry['rep_history_execution']=='prior_residual':weights_A(ids,labels,reps,selected,prior,pi,hm,context['observed_rep'],weights)
 else:_weights_A[(g,triton.cdiv(cap,256))](ids,labels,reps,selected,population,new_amount,weights,n,f,cap,256,num_warps=4)
 context['mass_error']=torch.zeros(g,device=prior.device);record.update(rule.diagnostics(errors,selected,None,old_ids,labels,reps,context));record['history_cap_discarded']=torch.zeros(g,device=prior.device)
 return ids,real,weights,record
