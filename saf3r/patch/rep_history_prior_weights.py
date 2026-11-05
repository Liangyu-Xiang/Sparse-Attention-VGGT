"""History A variant: predict residual mass at Rep instead of uniform counts.

Inference adds no selected-edge QK. It uses the same existing per-key prior
as the A loss. Non-Rep history weights stay1. Rep's predicted own mass plus
unrepresented members go to Rep. Exact keys keep1; Drop emits nothing.
These are predicted-mass identities, not a guarantee for true Dense masses.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _weights(IDS,LABEL,REP,SEL,PRIOR,PI,HMASS,SEEN,OUT,
             N:tl.constexpr,F:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B)
 key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0);safe=tl.maximum(key,0)
 c=tl.load(LABEL+safe,valid,0);rep=tl.load(REP+c,valid,0)
 state=tl.load(SEL+g*F+c,valid,0);own=tl.load(PRIOR+g*N+rep,valid,1.)
 total=tl.load(PI+g*F+c,valid,0.);hm=tl.load(HMASS+g*F+c,valid,0.)
 seen=tl.load(SEEN+g*F+c,valid,False)
 remainder=total-hm+tl.where(seen,own,0.)
 mass=tl.where(state==1,total,remainder)
 weight=tl.where((key==rep)&((state==1)|(state==5)),tl.maximum(1.,tl.div_rn(mass,tl.maximum(own,1.e-30))),1.)
 tl.store(OUT+g*A+j,tl.where(valid,weight,0.),j<A)

def weights_A(ids,labels,reps,selected,prior,pi,history_mass,observed_rep,out):
 g,n=prior.shape;f=len(reps);a=ids.shape[1]
 _weights[(g,triton.cdiv(a,256))](ids,labels,reps,selected,prior,pi,history_mass,observed_rep,out,n,f,a,256,num_warps=4,enable_fp_fusion=False)
 return out

def reference(ids,labels,reps,selected,prior,pi,qualified):
 """Independent eager reference, also usable on CPU for analytical tests."""
 valid=ids>=0;safe=ids.clamp_min(0).long();c=labels[safe];rep=reps[c]
 states=selected.gather(1,c);p=prior.gather(1,rep)
 history_mass=torch.zeros_like(pi).scatter_add_(1,labels[None].expand_as(prior),prior*qualified)
 observed=qualified[:,reps]
 residual=pi-history_mass+prior[:,reps]*observed
 mass=torch.where(states==1,pi.gather(1,c),residual.gather(1,c))
 ratio=(mass/p.clamp_min(1.e-30)).clamp_min(1.)
 return torch.where(valid,torch.where((safe==rep)&((states==1)|(states==5)),ratio,1.),0.)
