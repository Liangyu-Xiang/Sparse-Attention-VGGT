"""GPU whole-region expansion with exact budget and residual representative costs."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fill(O,SEEN,TABLE,ORDER,COUNTS,BEFORE,NP:tl.constexpr,KA:tl.constexpr,
          H:tl.constexpr,P:tl.constexpr,CAP:tl.constexpr,BC:tl.constexpr):
    group=tl.program_id(0);slot=tl.program_id(1)
    before=tl.load(BEFORE+group*P+slot)
    available=tl.load(COUNTS+group*P+slot)
    take=tl.minimum(available,KA-H-before)
    if take>0:
        parent=tl.load(ORDER+group*P+slot)
        col=tl.arange(0,BC)
        ids=tl.load(TABLE+parent*CAP+col,col<CAP,-1)
        historical=tl.load(SEEN+group*NP+tl.maximum(ids,0),(col<CAP)&(ids>=0),0)
        valid=(col<CAP)&(ids>=0)&(historical==0)
        rank=tl.cumsum(valid.to(tl.int32),0)-1
        bucket=tl.maximum(rank,0)*take//tl.maximum(available,1)
        previous=tl.maximum(rank-1,0)*take//tl.maximum(available,1)
        chosen=valid&((rank==0)|(bucket>previous))
        tl.store(O+group*KA+H+before+bucket,ids,chosen)


def fill_attention(output,seen,table,order,counts,before,history_used):
    if not output.is_contiguous():raise ValueError('Contiguous support buffer required')
    _fill[(len(output),len(table))](output,seen,table,order,counts,before,seen.shape[1],output.shape[1],
        history_used,len(table),table.shape[1],triton.next_power_of_2(table.shape[1]),num_warps=4)


@triton.jit
def _cost(X,CENTERS,BINS,VALID,OUT,F:tl.constexpr,K:tl.constexpr,D:tl.constexpr,
          BD:tl.constexpr,BF:tl.constexpr):
    group=tl.program_id(0);fine=tl.program_id(1)*BF+tl.arange(0,BF)
    dd=tl.arange(0,BD)
    eligible=tl.load(VALID+group*F+fine,fine<F,0)!=0
    bins=tl.load(BINS+group*F+fine,fine<F,0)
    x=tl.load(X+fine[:,None]*D+dd[None,:],(fine[:,None]<F)&(dd[None,:]<D),0)
    center=tl.load(CENTERS+(group*K+bins[:,None])*D+dd[None,:],
        (fine[:,None]<F)&eligible[:,None]&(dd[None,:]<D),0)
    cost=tl.sum((x-center)*(x-center),1)
    tl.store(OUT+group*F+fine,tl.where(eligible,cost,float('inf')),fine<F)


def representative_costs(fine_values,centers,bins,uncovered):
    groups,nfine=bins.shape;d=fine_values.shape[-1]
    result=torch.empty((groups,nfine),device=fine_values.device,dtype=torch.float32)
    _cost[(groups,triton.cdiv(nfine,32))](fine_values,centers,bins,uncovered,result,nfine,
        centers.shape[1],d,triton.next_power_of_2(d),32,num_warps=4,enable_fp_fusion=False)
    return result
