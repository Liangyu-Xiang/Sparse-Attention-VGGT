"""Same RMS Drop, weighted L2 Rep/History losses, owners and state order."""
import torch,triton
import triton.language as tl

@triton.jit(do_not_specialize=['COUNT'])
def _states(PRIOR,Z,MEM,DIST,DR,QUAL,REMOVE,AMOUNT,ALLOW,POP,CLUSTERS,ERROR,COST,OBJECTIVE,OWNER,
            N:tl.constexpr,F:tl.constexpr,D:tl.constexpr,W:tl.constexpr,COUNT,C:tl.constexpr,M:tl.constexpr,BD:tl.constexpr,COUNT_PTR:tl.constexpr):
    if COUNT_PTR:COUNT=tl.load(COUNT)
    g=tl.program_id(0);ci=tl.program_id(1)*C+tl.arange(0,C);m=tl.arange(0,M);j=tl.arange(0,M);d=tl.arange(0,BD)
    c=tl.load(CLUSTERS+ci,ci<COUNT,0)
    member=tl.load(MEM+c[:,None]*W+m[None,:],ci[:,None]<COUNT,-1);valid=(ci[:,None]<COUNT)&(member>=0)
    weight=tl.load(PRIOR+g*N+tl.maximum(member,0),valid,0.)
    repdist=tl.load(DR+tl.maximum(member,0),valid,0.);rep=tl.sum(weight*repdist,1)
    values=tl.load(Z+tl.maximum(member,0)[:,:,None]*D+d[None,None,:],valid[:,:,None]&(d[None,None,:]<D),0.)
    moment=tl.sum(weight[:,:,None]*values,1);drop=tl.sqrt(tl.sum(moment*moment,1))
    candidates=tl.load(MEM+c[:,None]*W+j[None,:],ci[:,None]<COUNT,-1)
    qualified=tl.load(QUAL+g*N+tl.maximum(candidates,0),(ci[:,None]<COUNT)&(candidates>=0),False)
    distance=tl.load(DIST+tl.maximum(member,0)[:,:,None]*W+j[None,None,:],valid[:,:,None]&qualified[:,None,:],float('inf'))
    distance=tl.where(qualified[:,None,:],distance,float('inf'))
    minimum=tl.min(distance,2)
    owner=tl.min(tl.where(qualified[:,None,:]&(distance==minimum[:,:,None]),candidates[:,None,:],2147483647),2)
    own=tl.load(QUAL+g*N+tl.maximum(member,0),valid,False);owner=tl.where(own,member,owner)
    has=valid&(owner<2147483647);hist=tl.sum(tl.where(has,weight*minimum,0.),1)
    tl.store(OWNER+g*N+member,tl.where(has,owner,-1),valid)
    remove=tl.load(REMOVE+g*F+c,ci<COUNT,False);amount=tl.load(AMOUNT+g*F+c,ci<COUNT,0);allowed=tl.load(ALLOW+g*F+c,ci<COUNT,False);pop=tl.load(POP+c,ci<COUNT,0.)
    rep_error=tl.where(remove,float('inf'),rep);hist_error=tl.where(amount>0,hist,float('inf'));exact=tl.where(allowed,0.,float('inf'))
    base=(g*F+c)*4
    tl.store(ERROR+base,drop,ci<COUNT);tl.store(ERROR+base+1,rep_error,ci<COUNT);tl.store(ERROR+base+2,hist_error,ci<COUNT);tl.store(ERROR+base+3,exact,ci<COUNT)
    tl.store(COST+base,0.,ci<COUNT);tl.store(COST+base+1,1.,ci<COUNT);tl.store(COST+base+2,amount.to(tl.float32),ci<COUNT);tl.store(COST+base+3,pop,ci<COUNT)
    tl.store(OBJECTIVE+base,drop-rep,ci<COUNT);tl.store(OBJECTIVE+base+1,rep_error-rep,ci<COUNT);tl.store(OBJECTIVE+base+2,hist_error-rep,ci<COUNT);tl.store(OBJECTIVE+base+3,exact-rep,ci<COUNT)

def states(prior,z,reps,labels,population,allowed,geometry,context):
    g,n=prior.shape;f=len(reps);d=z.shape[-1]
    # Exact eligibility comes from a transposed sparse-MM result in real inference.
    # zeros_like preserves that layout; Triton below uses row-major indexing.
    allowed=allowed.contiguous()
    errors=torch.empty((g,f,4),device=prior.device);costs=torch.empty_like(errors);objective=torch.empty_like(errors);owner=torch.empty((g,n),device=prior.device,dtype=torch.int32)
    for index,(size,clusters) in enumerate(geometry['drop_bins']):
        c=min(16,max(1,128//size))
        pointer='drop_bin_counts' in geometry
        count=geometry['drop_bin_counts'][index] if pointer else len(clusters)
        _states[(g,triton.cdiv(len(clusters),c))](prior,z,geometry['members'],geometry['distances'],geometry['rep_distance'],context['qualified'],context['remove_rep'],context['eligible_count'],allowed,population,clusters,errors,costs,objective,owner,n,f,d,geometry['width'],count,c,size,triton.next_power_of_2(d),pointer,num_warps=8 if size>=16 else 4,enable_fp_fusion=False)
    geometry['objective']=objective;context['owners']=owner
    return errors,costs
