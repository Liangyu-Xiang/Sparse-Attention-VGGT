"""Complete thresholded History loss in the original cached squared-L2 geometry."""
import torch,triton
import triton.language as tl

@triton.jit
def _history(DIST,MEM,LABEL,QUAL,PRIOR,LOSS,OWNER,N:tl.constexpr,F:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);col=tl.arange(0,W)
    valid=j<N;c=tl.load(LABEL+j,valid,0)
    member=tl.load(MEM+c[:,None]*W+col[None,:],valid[:,None],-1)
    qualified=tl.load(QUAL+g*N+tl.maximum(member,0),valid[:,None]&(member>=0),False)
    distance=tl.load(DIST+j[:,None]*W+col[None,:],valid[:,None],float('inf'))
    distance=tl.where(qualified,distance,float('inf'))
    minimum=tl.min(distance,1)
    owner=tl.min(tl.where(qualified&(distance==minimum[:,None]),member,2147483647),1)
    self_selected=tl.load(QUAL+g*N+j,valid,False)
    owner=tl.where(self_selected,j,owner)
    p=tl.load(PRIOR+g*N+j,valid,0.)
    has=owner<2147483647
    tl.atomic_add(LOSS+g*F+c,tl.where(has,p*minimum,0.),valid&has,sem='relaxed')
    tl.store(OWNER+g*N+j,tl.where(has,owner,-1),valid)


def history_loss(prior,labels,geometry,qualified,clusters):
    g,n=prior.shape;loss=torch.zeros((g,clusters),device=prior.device)
    owner=torch.empty((g,n),device=prior.device,dtype=torch.int32)
    _history[(g,triton.cdiv(n,64))](geometry['distances'],geometry['members'],labels,qualified,prior,loss,owner,n,clusters,geometry['width'],64,num_warps=4,enable_fp_fusion=False)
    return loss,owner

@triton.jit
def _emit(FLAGS,PREFIX,IDS,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);valid=j<N
    keep=tl.load(FLAGS+g*N+j,valid,0)>0;position=tl.load(PREFIX+g*N+j,valid,0)-1
    tl.store(IDS+g*A+position,j,valid&keep&(position<A))
