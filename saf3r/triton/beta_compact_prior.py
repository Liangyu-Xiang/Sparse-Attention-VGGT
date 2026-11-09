"""Compact Unified-Beta prior and cluster-statistics kernels."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _base(BLOCK, GRID, MEMBERS, ENERGY, DISTANCE, OUT, R:tl.constexpr, F:tl.constexpr, W:tl.constexpr, C:tl.constexpr):
    group=tl.program_id(0)
    cluster=tl.program_id(1)*C+tl.arange(0,C)
    lane=tl.arange(0,W)
    key=tl.load(MEMBERS+cluster[:,None]*W+lane[None,:],cluster[:,None]<F,-1)
    valid=key>=0
    region=tl.load(GRID+key,valid,0)
    weight=tl.load(BLOCK+group*R+region,valid,0.)
    energy=tl.load(ENERGY+key,valid,0.)
    distance=tl.load(DISTANCE+key,valid,0.)
    tl.store(OUT+(group*F+cluster)*3,tl.sum(weight,1),cluster<F)
    tl.store(OUT+(group*F+cluster)*3+1,tl.sum(weight*energy,1),cluster<F)
    tl.store(OUT+(group*F+cluster)*3+2,tl.sum(weight*distance,1),cluster<F)


@triton.jit
def _add_history(IDS, DELTA, LABELS, ENERGY, DISTANCE, OUT, F:tl.constexpr, A:tl.constexpr, B:tl.constexpr):
    group=tl.program_id(0)
    index=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+group*A+index,index<A,-1)
    valid=key>=0
    cluster=tl.load(LABELS+key,valid,0)
    delta=tl.load(DELTA+group*A+index,valid,0.)
    energy=tl.load(ENERGY+key,valid,0.)
    distance=tl.load(DISTANCE+key,valid,0.)
    target=OUT+(group*F+cluster)*3
    tl.atomic_add(target,delta,valid,sem='relaxed')
    tl.atomic_add(target+1,delta*energy,valid,sem='relaxed')
    tl.atomic_add(target+2,delta*distance,valid,sem='relaxed')


def cluster_statistics(prior, members, labels, energy, distance):
    groups, regions=prior.block_weight.shape
    clusters, width=members.shape
    output=torch.empty((groups,clusters,3),device=members.device,dtype=torch.float32)
    _base[(groups,triton.cdiv(clusters,4))](prior.block_weight,prior.grid,members,energy,distance,output,
                                            regions,clusters,width,4,num_warps=4,enable_fp_fusion=False)
    if prior.delta is not None:
        keys=prior.ids.shape[1]
        _add_history[(groups,triton.cdiv(keys,256))](prior.ids,prior.delta,labels,energy,distance,output,
                                                    clusters,keys,256,num_warps=4,enable_fp_fusion=False)
    return output


@triton.jit
def _rep_prior(BLOCK, GRID, REPS, OUT, R:tl.constexpr, F:tl.constexpr, B:tl.constexpr):
    group=tl.program_id(0)
    cluster=tl.program_id(1)*B+tl.arange(0,B)
    rep=tl.load(REPS+cluster,cluster<F,0)
    region=tl.load(GRID+rep,cluster<F,0)
    weight=tl.load(BLOCK+group*R+region,cluster<F,0.)
    tl.store(OUT+group*F+cluster,weight,cluster<F)


@triton.jit
def _gate(IDS, DELTA, BLOCK, GRID, PI, POP, LABEL, REP, DR, REP_PRIOR, COUNT, SEEN,
          AMOUNT, QUAL, KR, HM, N:tl.constexpr, R:tl.constexpr, F:tl.constexpr, A:tl.constexpr, B:tl.constexpr,TILED_COUNT:tl.constexpr):
    group=tl.program_id(0)
    slot=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+group*A+slot,slot<A,-1)
    valid=key>=0
    cluster=tl.load(LABEL+key,valid,0)
    rep=tl.load(REP+cluster,valid,-1)
    region=tl.load(GRID+key,valid,0)
    own=tl.load(BLOCK+group*R+region,valid,0.)+tl.load(DELTA+group*A+slot,valid,0.)
    offset=group*F+cluster
    mass=tl.load(PI+offset,valid,0.)
    population=tl.load(POP+cluster,valid,1.)
    is_rep=valid&(key==rep)
    eligible=valid&((own>tl.div_rn(mass,population))|is_rep)
    tl.store(REP_PRIOR+offset,own,is_rep)
    if TILED_COUNT:
        tile=tl.program_id(1)
        tl.store(COUNT+group*tl.cdiv(A,B)+tile,tl.sum(valid.to(tl.int32),0))
        # Packed history has unique Key IDs within each query group.
        tl.store(SEEN+offset,1,is_rep)
    else:
        tl.atomic_add(COUNT+offset,1,valid,sem='relaxed')
        tl.atomic_max(SEEN+offset,1,is_rep,sem='relaxed')
    tl.atomic_add(AMOUNT+offset,1,eligible,sem='relaxed')
    tl.store(QUAL+group*N+key,eligible,valid)
    distance=tl.load(DR+key,eligible,0.)
    tl.atomic_add(KR+offset,own*distance,eligible,sem='relaxed')
    tl.atomic_add(HM+offset,own,eligible,sem='relaxed')


def gate(prior, pi, population, labels, reps, distance, *, tiled_count=False):
    groups, regions=prior.block_weight.shape
    clusters=len(reps)
    keys=len(labels)
    count=torch.zeros((groups,clusters),device=pi.device,dtype=torch.int32)
    seen=torch.zeros_like(count)
    if tiled_count and prior.delta is not None:
        count=torch.empty((groups,triton.cdiv(prior.ids.shape[1],256)),device=pi.device,dtype=torch.int32)
    amount=torch.zeros_like(seen)
    qualified=torch.zeros((groups,keys),device=pi.device,dtype=torch.bool)
    residual=torch.zeros_like(pi)
    history_mass=torch.zeros_like(pi)
    rep_prior=torch.empty_like(pi)
    _rep_prior[(groups,triton.cdiv(clusters,256))](prior.block_weight,prior.grid,reps,rep_prior,regions,clusters,256)
    if prior.delta is not None:
        history_keys=prior.ids.shape[1]
        _gate[(groups,triton.cdiv(history_keys,256))](
            prior.ids,prior.delta,prior.block_weight,prior.grid,pi,population,labels,reps,distance,
            rep_prior,count,seen,amount,qualified,residual,history_mass,keys,regions,clusters,history_keys,256,tiled_count,
            num_warps=4,enable_fp_fusion=False)
    observed=seen>0
    return dict(qualified=qualified,eligible_count=amount,observed_rep=observed,
                remove_rep=torch.zeros_like(observed),history_count=count),residual,history_mass,rep_prior


@triton.jit
def _prior_weights(IDS, LABEL, REP, SEL, REP_PRIOR, PI, HMASS, SEEN, OUT,
                   F:tl.constexpr, A:tl.constexpr, B:tl.constexpr):
    group=tl.program_id(0)
    slot=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+group*A+slot,slot<A,-1)
    valid=key>=0
    cluster=tl.load(LABEL+key,valid,0)
    rep=tl.load(REP+cluster,valid,-1)
    offset=group*F+cluster
    state=tl.load(SEL+offset,valid,0)
    own=tl.load(REP_PRIOR+offset,valid,1.)
    total=tl.load(PI+offset,valid,0.)
    history=tl.load(HMASS+offset,valid,0.)
    seen=tl.load(SEEN+offset,valid,False)
    mass=tl.where(state==1,total,total-history+tl.where(seen,own,0.))
    weight=tl.where((key==rep)&((state==1)|(state==5)),
                    tl.maximum(1.,tl.div_rn(mass,tl.maximum(own,1.e-30))),1.)
    tl.store(OUT+group*A+slot,tl.where(valid,weight,0.),slot<A)


def weights(ids, labels, reps, selected, rep_prior, pi, history_mass, observed_rep, output):
    groups, keys=ids.shape
    clusters=len(reps)
    _prior_weights[(groups,triton.cdiv(keys,256))](
        ids,labels,reps,selected,rep_prior,pi,history_mass,observed_rep,output,clusters,keys,256,
        num_warps=4,enable_fp_fusion=False)


@triton.jit
def _expand_reference(BASE,GRID,OUT,N:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0)
    key=tl.program_id(1)*B+tl.arange(0,B)
    region=tl.load(GRID+key,key<N,0)
    value=tl.load(BASE+group*R+region,key<N,0.)
    tl.store(OUT+group*(N+1)+key,value,key<=N)


@triton.jit
def _override_reference(IDS,VALUES,OUT,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0)
    slot=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+group*A+slot,slot<A,-1)
    value=tl.load(VALUES+group*A+slot,slot<A,0.)
    tl.store(OUT+group*(N+1)+key,value,key>=0)


@triton.jit
def _apply_reference_lift(IDS,HISTORY,GRID,BASE,PREVIOUS,BETA,OUT,
                          N:tl.constexpr,R:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0)
    slot=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+group*A+slot,slot<A,-1)
    valid=key>=0
    region=tl.load(GRID+key,valid,0)
    history=tl.load(HISTORY+group*A+slot,valid,0.)
    previous=tl.load(PREVIOUS+group*R+region,valid,1.)
    beta=tl.load(BETA+group*R+region,valid,0.)
    base=tl.load(BASE+group*R+region,valid,0.)
    ratio=tl.div_rn(history,tl.maximum(previous,1.e-30))
    residual=libdevice.log(tl.maximum(ratio,1.e-30))
    lift=libdevice.exp((.5*beta)*residual)
    tl.store(OUT+group*(N+1)+key,base*lift,valid)
