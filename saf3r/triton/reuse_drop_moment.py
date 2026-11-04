"""Exact weighted RMS-Value moment norm, without a [group, cluster, D] bank."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["COUNT"])
def _moment(PRIOR,Z,MEM,CLUSTERS,OUT,N:tl.constexpr,F:tl.constexpr,D:tl.constexpr,
            WIDTH:tl.constexpr,COUNT,C:tl.constexpr,M:tl.constexpr,BD:tl.constexpr):
    group=tl.program_id(0);ci=tl.program_id(1)*C+tl.arange(0,C)
    m=tl.arange(0,M);d=tl.arange(0,BD)
    cluster=tl.load(CLUSTERS+ci,ci<COUNT,0)
    member=tl.load(MEM+cluster[:,None]*WIDTH+m[None,:],ci[:,None]<COUNT,-1)
    valid=(ci[:,None]<COUNT)&(member>=0)
    weight=tl.load(PRIOR+group*N+tl.maximum(member,0),valid,0.)
    value=tl.load(Z+tl.maximum(member,0)[:,:,None]*D+d[None,None,:],valid[:,:,None]&(d[None,None,:]<D),0.)
    moment=tl.sum(weight[:,:,None]*value,1)
    norm=tl.sqrt(tl.sum(moment*moment,1))
    tl.store(OUT+group*F+cluster,norm,ci<COUNT)


def prepare(population):
    """O(F) once per layer; bins change work layout, never cluster membership."""
    bins=[];lower=0
    width=triton.next_power_of_2(int(population.max()))
    if width>512:raise ValueError('Compact geometry supports at most 512 members')
    for size in [1,2,4,8,16,32,64,128,256,512]:
        if size>width:break
        ids=torch.where(((population>lower)|((population==0)&(lower==0)))&(population<=size))[0].int().contiguous()
        if ids.numel():bins.append((size,ids))
        lower=size
    return bins


def drop_norm(prior,z,geometry,clusters,bins):
    g,n=prior.shape;d=z.shape[1]
    out=torch.empty((g,clusters),device=prior.device,dtype=torch.float32)
    for size,ids in bins:
        c=min(32,max(1,256//size))
        _moment[(g,triton.cdiv(len(ids),c))](prior,z,geometry['members'],ids,out,
            n,clusters,d,geometry['width'],len(ids),c,size,triton.next_power_of_2(d),
            num_warps=4,enable_fp_fusion=False)
    return out
