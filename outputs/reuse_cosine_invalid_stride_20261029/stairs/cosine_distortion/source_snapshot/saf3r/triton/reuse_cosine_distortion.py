"""Tokenwise exact cosine state losses and estimated-mass assignment, FP32."""
import torch
import triton
import triton.language as tl


@triton.jit
def _losses(Z,LABEL,REP,HIDS,PRIOR,E,P:tl.constexpr,F:tl.constexpr,D:tl.constexpr,L:tl.constexpr,BD:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);dd=tl.arange(0,BD)
    valid=j<P;c=tl.load(LABEL+j,valid,0);rep=tl.load(REP+c)
    z=tl.load(Z+j[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
    zr=tl.load(Z+rep[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
    dr=tl.minimum(2.,tl.maximum(0.,1-tl.sum(z*zr,1)));dr=tl.where(j==rep,0.,dr)
    a=tl.load(PRIOR+group*P+j,valid,0.)
    width=2*L+3;base=(group*F+c)*width
    tl.atomic_add(E+base+1,a*dr,valid,sem='relaxed')
    minimum=tl.full((B,),2.,tl.float32);chosen=tl.full((B,),False,tl.int1)
    for slot in tl.static_range(L):
        h=tl.load(HIDS+(group*F+c)*L+slot)
        zh=tl.load(Z+tl.maximum(h,0)[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
        d=tl.minimum(2.,tl.maximum(0.,1-tl.sum(z*zh,1)));d=tl.where(j==h,0.,d)
        minimum=tl.minimum(minimum,tl.where(h>=0,d,2.));chosen=chosen|((h>=0)&(j==h))
        tl.atomic_add(E+base+2+slot,a*minimum,valid,sem='relaxed')
        tl.atomic_add(E+base+2+L+slot,a*tl.where(chosen,0.,dr),valid,sem='relaxed')


@triton.jit
def _transfer(Z,LABEL,REP,HIDS,PRIOR,SELECTED,M,P:tl.constexpr,F:tl.constexpr,D:tl.constexpr,L:tl.constexpr,BD:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);dd=tl.arange(0,BD)
    valid=j<P;c=tl.load(LABEL+j,valid,0);s=tl.load(SELECTED+group*F+c)
    rep=tl.load(REP+c);history_only=(s>=2)&(s<=L+1);history_rep=(s>=L+2)&(s<=2*L+1)
    t=tl.where(history_only,s-1,tl.where(history_rep,s-L-1,0))
    z=tl.load(Z+j[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
    minimum=tl.full((B,),float('inf'),tl.float32);nearest=rep;destination=rep
    for slot in tl.static_range(L):
        h=tl.load(HIDS+(group*F+c)*L+slot);use=(slot<t)&(h>=0)
        zh=tl.load(Z+tl.maximum(h,0)[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
        d=tl.minimum(2.,tl.maximum(0.,1-tl.sum(z*zh,1)));d=tl.where(j==h,-1.,d)
        better=use&(d<minimum);nearest=tl.where(better,h,nearest);minimum=tl.where(better,d,minimum)
        destination=tl.where(history_rep&use&(j==h),h,destination)
    destination=tl.where(history_only,nearest,destination)
    destination=tl.where(s==2*L+2,j,destination)
    a=tl.load(PRIOR+group*P+j,valid,0.)
    tl.atomic_add(M+group*P+destination,a,valid&(s!=0),sem='relaxed')


def distortion_losses(z,labels,reps,hids,prior,clusters):
    g=hids.shape[0];l=hids.shape[-1];p,d=z.shape
    errors=torch.zeros((g,clusters,2*l+3),device=z.device,dtype=torch.float32)
    _losses[(g,triton.cdiv(p,64))](z,labels,reps,hids,prior,errors,p,clusters,d,l,triton.next_power_of_2(d),64,num_warps=4)
    return errors


def transferred_masses(z,labels,reps,hids,prior,selected):
    g,f=selected.shape;p,d=z.shape;l=hids.shape[-1]
    mass=torch.zeros((g,p),device=z.device,dtype=torch.float32)
    _transfer[(g,triton.cdiv(p,64))](z,labels,reps,hids,prior,selected,mass,p,f,d,l,triton.next_power_of_2(d),64,num_warps=4)
    return mass
