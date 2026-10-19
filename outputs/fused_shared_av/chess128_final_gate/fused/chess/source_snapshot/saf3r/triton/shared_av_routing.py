"""FP32 per-head/per-probe coarse softmax, streamed reduction into shared scores.

Avoid materializing FP32 H*G*4*C logits AND probability tensors. Input is the
BF16 Tensor-core coarse GEMM output; no Q/K head averaging is performed.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _route(S,POP,O,G:tl.constexpr,C:tl.constexpr,HP:tl.constexpr,SCALE:tl.constexpr,BC:tl.constexpr):
    group=tl.program_id(0);col=tl.arange(0,BC)
    logpop=tl.load(POP+col,col<C,0)
    acc=tl.zeros((BC,),tl.float32)
    for hp in range(HP):
        head,probe=hp//4,hp%4
        z=tl.load(S+((head*G+group)*4+probe)*C+col,col<C,0).to(tl.float32)*SCALE+logpop
        z=tl.where(col<C,z,-float('inf'))
        w=tl.exp(z-tl.max(z,0));w=w/tl.sum(w,0)
        acc+=w
    tl.store(O+group*C+col,acc/HP,col<C)


def routing_probabilities(scores,log_population,groups,scale):
    out=torch.empty((groups,scores.shape[-1]),device=scores.device,dtype=torch.float32)
    _route[(groups,)](scores,log_population,out,groups,scores.shape[-1],scores.shape[0]*4,scale,
        triton.next_power_of_2(scores.shape[-1]),num_warps=8)
    return out
