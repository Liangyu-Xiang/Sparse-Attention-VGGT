"""Exact compact reuse operations. No state pruning or approximate allocator.

Value geometry is independent of query groups: cache within-cluster cosine
costs once per layer, then contract the EXISTING arbitrary per-token prior.
This preserves cross-grid priors, clamp/self conventions and all H-only states.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _geometry(Z,MEM,LABEL,RANK,REP,DIST,DR,P:tl.constexpr,D:tl.constexpr,M:tl.constexpr,BD:tl.constexpr,B:tl.constexpr):
    pair=tl.program_id(0)*B+tl.arange(0,B);j=pair//M;col=pair%M;dd=tl.arange(0,BD)
    valid=j<P;c=tl.load(LABEL+j,valid,0);other=tl.load(MEM+c*M+col,valid,-1)
    z=tl.load(Z+j[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0.)
    zo=tl.load(Z+tl.maximum(other,0)[:,None]*D+dd[None,:],valid[:,None]&(other[:,None]>=0)&(dd[None,:]<D),0.)
    distance=tl.minimum(2.,tl.maximum(0.,1.-tl.sum(z*zo,1)))
    distance=tl.where(j==other,0.,distance)
    tl.store(DIST+pair,distance,valid)
    rep=tl.load(REP+c,valid,0)
    tl.store(DR+j,distance,valid&(other==rep))


def prepare_geometry(z,labels,reps,population):
    p,d=z.shape;f=len(reps)
    width=triton.next_power_of_2(int(population.max()))
    # A pathological oversized cluster falls back to the original kernels;
    # this is an implementation-memory guard, never a state/quality change.
    if width>256:return None
    order=labels.argsort(stable=True)
    starts=population.long().cumsum(0)-population.long()
    rank=torch.empty(p,device=z.device,dtype=torch.int32)
    rank[order]=(torch.arange(p,device=z.device)-starts[labels[order]]).int()
    members=torch.full((f,width),-1,device=z.device,dtype=torch.int32)
    members[labels,rank.long()]=torch.arange(p,device=z.device,dtype=torch.int32)
    distances=torch.empty((p,width),device=z.device,dtype=torch.float32)
    rep_distance=torch.empty(p,device=z.device,dtype=torch.float32)
    _geometry[(triton.cdiv(p*width,128),)](z,members,labels,rank,reps,distances,rep_distance,p,d,width,triton.next_power_of_2(d),128,num_warps=4)
    return dict(members=members,rank=rank,distances=distances,rep_distance=rep_distance,width=width)


@triton.jit
def _top(POS,PRIOR,MEM,REP,OUT,DISCARD,P:tl.constexpr,F:tl.constexpr,M:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);c=tl.program_id(1)*B+tl.arange(0,B);col=tl.arange(0,M)
    members=tl.load(MEM+c[:,None]*M+col[None,:],c[:,None]<F,-1)
    pos=tl.load(POS+group*(P+1)+tl.maximum(members,0),(c[:,None]<F)&(members>=0),-1)
    rep=tl.load(REP+c,c<F,0)
    mass=tl.load(PRIOR+group*P+tl.maximum(members,0),(c[:,None]<F)&(members>=0),0.)
    valid=(members>=0)&(pos>=0)&(members!=rep[:,None])&(mass>0)&(c[:,None]<F)
    total=tl.sum(valid.to(tl.int32),1)
    scores=tl.where(valid,mass,-1.)
    for slot in tl.static_range(H):
        best=tl.max(scores,1)
        position=tl.min(tl.where((scores==best[:,None])&(best[:,None]>0),pos,2147483647),1)
        winner=(pos==position[:,None])&(scores==best[:,None])&(best[:,None]>0)
        key=tl.max(tl.where(winner,members,-1),1)
        tl.store(OUT+(group*F+c)*H+slot,key,c<F)
        scores=tl.where(winner,-1.,scores)
    tl.store(DISCARD+group*F+c,tl.maximum(total-H,0),c<F)


def top_history(ids,prior,reps,geometry,limit):
    g,p=prior.shape;f=len(reps)
    positions=torch.full((g,p+1),-1,device=prior.device,dtype=torch.int32)
    source=torch.arange(ids.shape[1],device=ids.device,dtype=torch.int32)[None].expand_as(ids)
    positions.scatter_(1,ids.long().masked_fill(ids<0,p),source.masked_fill(ids<0,-1))
    hids=torch.empty((g,f,limit),device=ids.device,dtype=torch.int32)
    discard=torch.empty((g,f),device=ids.device,dtype=torch.int32)
    _top[(g,triton.cdiv(f,32))](positions,prior,geometry['members'],reps,hids,discard,p,f,geometry['width'],limit,32,num_warps=4)
    return hids,discard.sum(1)


@triton.jit
def _loss(DIST,DR,RANK,LABEL,HIDS,PRIOR,E,P:tl.constexpr,F:tl.constexpr,M:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);valid=j<P
    c=tl.load(LABEL+j,valid,0);a=tl.load(PRIOR+group*P+j,valid,0.);dr=tl.load(DR+j,valid,0.)
    base=(group*F+c)*(2*H+3)
    tl.atomic_add(E+base+1,a*dr,valid,sem='relaxed')
    minimum=tl.full((B,),2.,tl.float32);chosen=tl.full((B,),False,tl.int1)
    for slot in tl.static_range(H):
        h=tl.load(HIDS+(group*F+c)*H+slot,valid,-1)
        rank=tl.load(RANK+tl.maximum(h,0),valid&(h>=0),0)
        d=tl.load(DIST+j*M+rank,valid,2.)
        minimum=tl.minimum(minimum,tl.where(h>=0,d,2.));chosen=chosen|(j==h)
        tl.atomic_add(E+base+2+slot,a*minimum,valid,sem='relaxed')
        tl.atomic_add(E+base+2+H+slot,a*tl.where(chosen,0.,dr),valid,sem='relaxed')


def losses(labels,hids,prior,geometry,clusters):
    g,p=prior.shape;h=hids.shape[-1]
    errors=torch.zeros((g,clusters,2*h+3),device=prior.device)
    _loss[(g,triton.cdiv(p,256))](geometry['distances'],geometry['rep_distance'],geometry['rank'],labels,hids,prior,errors,p,clusters,geometry['width'],h,256,num_warps=4)
    return errors


@triton.jit
def _multi_transfer(DIST,RANK,LABEL,IDS,PRIOR,SEL,MASS,P:tl.constexpr,F:tl.constexpr,M:tl.constexpr,H:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);valid=j<P
    c=tl.load(LABEL+j,valid,0);s=tl.load(SEL+group*F+c,valid,0)
    active=valid&(s>=3)&(s<=H+1)
    if tl.sum(active.to(tl.int32),0)>0:
        nearest=tl.full((B,),0,tl.int32);minimum=tl.full((B,),float('inf'),tl.float32)
        for slot in tl.static_range(H):
            h=tl.load(IDS+(group*F+c)*H+slot,active,-1)
            rank=tl.load(RANK+tl.maximum(h,0),active&(h>=0),0)
            d=tl.load(DIST+j*M+rank,active,2.)
            d=tl.where(j==h,-1.,d)
            use=active&(slot<s-1)&(h>=0);better=use&(d<minimum)
            nearest=tl.where(better,h,nearest);minimum=tl.where(better,d,minimum)
        a=tl.load(PRIOR+group*P+j,active,0.)
        tl.atomic_add(MASS+group*P+nearest,a,active,sem='relaxed')


@triton.jit
def _analytic_transfer(KEYS,LABEL,REP,HIDS,PRIOR,PI,SEL,MASS,W,MEM,P:tl.constexpr,F:tl.constexpr,A:tl.constexpr,H:tl.constexpr,M:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(KEYS+group*A+col,col<A,-1);valid=(col<A)&(key>=0)
    c=tl.load(LABEL+tl.maximum(key,0),valid,0);rep=tl.load(REP+c,valid,0)
    state=tl.load(SEL+group*F+c,valid,0)
    own=tl.load(PRIOR+group*P+tl.maximum(key,0),valid,0.)
    pi=tl.load(PI+group*F+c,valid,0.)
    hr=(state>=H+2)&(state<=2*H+1);total=tl.full((B,),0.,tl.float32)
    for slot in tl.static_range(H):
        hist=tl.load(HIDS+(group*F+c)*H+slot,valid&hr,-1)
        a=tl.load(PRIOR+group*P+tl.maximum(hist,0),valid&hr&(hist>=0),0.)
        total+=tl.where(slot<state-H-1,a,0.)
    assigned=tl.where((state==1)|(state==2),pi,own)
    residual=pi-total
    unstable=valid&hr&(key==rep)&(residual<.01*pi)
    if tl.sum(unstable.to(tl.int32),0)>0:
        col_member=tl.arange(0,M)
        members=tl.load(MEM+c[:,None]*M+col_member[None,:],unstable[:,None],-1)
        remaining=members>=0
        for slot in tl.static_range(H):
            hist=tl.load(HIDS+(group*F+c)*H+slot,unstable,-1)
            remaining=remaining&~((members==hist[:,None])&(slot<state[:,None]-H-1))
        values=tl.load(PRIOR+group*P+tl.maximum(members,0),unstable[:,None]&remaining,0.)
        residual=tl.where(unstable,tl.sum(values,1),residual)
    assigned=tl.where(hr&(key==rep),residual,assigned)
    multiple=(state>=3)&(state<=H+1)
    multi_mass=tl.load(MASS+group*P+tl.maximum(key,0),valid&multiple,0.)
    assigned=tl.where(multiple,multi_mass,assigned)
    weight=tl.maximum(1.,assigned/tl.maximum(own,1.e-30))
    tl.store(W+group*A+col,tl.where(valid,weight,0.),col<A)


def transfer(selected,labels,reps,hids,prior,pi,packed,geometry):
    g,p=prior.shape;f=len(reps);h=hids.shape[-1]
    mass=torch.zeros_like(prior)
    _multi_transfer[(g,triton.cdiv(p,256))](geometry['distances'],geometry['rank'],labels,hids,prior,selected,mass,p,f,geometry['width'],h,256,num_warps=4)
    weights=torch.empty(packed.shape,device=prior.device,dtype=torch.float32)
    _analytic_transfer[(g,triton.cdiv(packed.shape[1],256))](packed,labels,reps,hids,prior,pi,selected,mass,weights,geometry['members'],p,f,packed.shape[1],h,geometry['width'],256,num_warps=4)
    return weights


@triton.jit
def _choose(E,C,DUAL,PART,SEL,F:tl.constexpr,S:tl.constexpr,T:tl.constexpr,FINAL:tl.constexpr,B:tl.constexpr,BS:tl.constexpr):
    group=tl.program_id(0);tile=tl.program_id(1)
    c=tile*B+tl.arange(0,B);s=tl.arange(0,BS)
    value=tl.load(E+(group*F+c[:,None])*S+s[None,:],(c[:,None]<F)&(s[None,:]<S),float('inf'))
    cost=tl.load(C+(group*F+c[:,None])*S+s[None,:],(c[:,None]<F)&(s[None,:]<S),0.)
    lo=tl.load(DUAL+group*2);hi=tl.load(DUAL+group*2+1)
    lam=hi if FINAL else (lo+hi)*.5
    objective=value+lam*cost
    winner=tl.argmin(objective,1,tie_break_left=True)
    chosen=tl.sum(tl.where(s[None,:]==winner[:,None],cost,0.),1)
    tl.store(PART+group*T+tile,tl.sum(tl.where(c<F,chosen,0.),0))
    if FINAL:tl.store(SEL+group*F+c,winner,c<F)


@triton.jit
def _dual(PART,DUAL,T:tl.constexpr,BUDGET:tl.constexpr,BT:tl.constexpr):
    group=tl.program_id(0);t=tl.arange(0,BT)
    count=tl.sum(tl.load(PART+group*T+t,t<T,0.),0)
    lo=tl.load(DUAL+group*2);hi=tl.load(DUAL+group*2+1);mid=(lo+hi)*.5
    tl.store(DUAL+group*2,tl.where(count>BUDGET,mid,lo))
    tl.store(DUAL+group*2+1,tl.where(count>BUDGET,hi,mid))


def budget(errors,costs,upper,cap,iterations):
    g,f,s=errors.shape;tiles=triton.cdiv(f,128)
    dual=torch.stack((torch.zeros_like(upper),upper),1).contiguous()
    partial=torch.empty((g,tiles),device=errors.device);selected=torch.empty((g,f),device=errors.device,dtype=torch.int64)
    for _ in range(iterations):
        _choose[(g,tiles)](errors,costs,dual,partial,selected,f,s,tiles,False,128,triton.next_power_of_2(s),num_warps=4,enable_fp_fusion=False)
        _dual[(g,)](partial,dual,tiles,cap,triton.next_power_of_2(tiles),num_warps=4,enable_fp_fusion=False)
    _choose[(g,tiles)](errors,costs,dual,partial,selected,f,s,tiles,True,128,triton.next_power_of_2(s),num_warps=4,enable_fp_fusion=False)
    return selected,partial.sum(1)


@triton.jit
def _support_raw(SEL,HIDS,REP,LABEL,POP,group,col,F:tl.constexpr,P:tl.constexpr,H:tl.constexpr):
    is_h=col<F*H;is_rep=(col>=F*H)&(col<F*(H+1));is_member=(col>=F*(H+1))&(col<F*(H+1)+P)
    cluster=tl.where(is_h,col//H,col-F*H)
    member=col-F*(H+1)
    member_cluster=tl.load(LABEL+tl.maximum(member,0),is_member,0)
    cluster=tl.where(is_member,member_cluster,cluster)
    safe=tl.maximum(0,tl.minimum(cluster,F-1))
    state=tl.load(SEL+group*F+safe)
    t=tl.where((state>=2)&(state<=H+1),state-1,tl.where((state>=H+2)&(state<=2*H+1),state-H-1,0))
    history=tl.load(HIDS+(group*F+safe)*H+col%H,is_h,-1)
    rep=tl.load(REP+safe,is_rep,-1)
    valid=(is_h&(col%H<t)&(history>=0))|(is_rep&((state==1)|((state>=H+2)&(state<=2*H+1))))|(is_member&(state==2*H+2))
    key=tl.where(is_h,history,tl.where(is_rep,rep,member))
    return key,valid


@triton.jit
def _pack(SEL,HIDS,REP,LABEL,POP,COUNTS,OUT,F:tl.constexpr,P:tl.constexpr,H:tl.constexpr,A:tl.constexpr,T:tl.constexpr,WRITE:tl.constexpr,B:tl.constexpr):
    group=tl.program_id(0);tile=tl.program_id(1);col=tile*B+tl.arange(0,B)
    key,valid=_support_raw(SEL,HIDS,REP,LABEL,POP,group,col,F,P,H)
    if WRITE:
        before=tl.load(COUNTS+group*T+tile-1,tile>0,0)
        pos=before+tl.cumsum(valid.to(tl.int32),0)-1
        tl.store(OUT+group*A+pos,key,valid&(pos<A))
    else:tl.store(COUNTS+group*T+tile,tl.sum(valid.to(tl.int32),0))


def pack(selected,hids,reps,labels,population,cap):
    g,f=selected.shape;h=hids.shape[-1];p=len(labels);tiles=triton.cdiv(f*(h+1)+p,1024)
    counts=torch.empty((g,tiles),device=selected.device,dtype=torch.int32)
    ids=torch.full((g,cap),-1,device=selected.device,dtype=torch.int32)
    _pack[(g,tiles)](selected,hids,reps,labels,population,counts,ids,f,p,h,cap,tiles,False,1024,num_warps=4)
    prefix=counts.cumsum(1,dtype=torch.int32).contiguous()
    _pack[(g,tiles)](selected,hids,reps,labels,population,prefix,ids,f,p,h,cap,tiles,True,1024,num_warps=4)
    hr=(selected>=h+2)&(selected<=2*h+1)
    t=torch.where((selected>=2)&(selected<=h+1),selected-1,torch.where(hr,selected-h-1,0))
    rep=(selected==1)|hr
    record=dict(reuse_edges=t.sum(1),representatives=rep.sum(1),exact_edges=((selected==2*h+2)*population[None]).sum(1),dropped_clusters=(selected==0).sum(1),counts=counts.sum(1))
    return ids,ids>=0,record
