"""Only the needed Rep loss; identical cached squared-L2 distances and p*dr."""
import torch,triton
import triton.language as tl

@triton.jit
def _rep(DR,LABEL,PRIOR,OUT,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);valid=j<N
    c=tl.load(LABEL+j,valid,0);p=tl.load(PRIOR+g*N+j,valid,0.);dr=tl.load(DR+j,valid,0.)
    tl.atomic_add(OUT+g*F+c,p*dr,valid,sem='relaxed')

def rep_loss(prior,labels,geometry,clusters):
    g,n=prior.shape;out=torch.zeros((g,clusters),device=prior.device)
    _rep[(g,triton.cdiv(n,256))](geometry['rep_distance'],labels,prior,out,n,clusters,256,num_warps=4)
    return out
