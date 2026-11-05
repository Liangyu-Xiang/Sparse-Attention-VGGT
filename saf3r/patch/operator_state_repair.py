"""Opt-in consistent state losses and additional real-key supports; no Dense QK."""
import torch
import triton
import triton.language as tl


def prepare(v,k,module,patches,specials,labels,reps,population,z,config):
    from .state_loss_reference import output_sketch
    y=output_sketch(v,module,patches,config.operator_feature_dim) if config.operator_loss!='legacy' else torch.zeros((v.shape[1],config.operator_feature_dim),device=v.device)
    width=max(64,triton.next_power_of_2(int(population.max())))
    if width>512:raise ValueError('Operator repair supports at most512 members per cluster')
    order=labels.argsort(stable=True);starts=population.long().cumsum(0)-population.long()
    members=torch.full((len(reps),width),-1,device=v.device,dtype=torch.int32)
    members[labels[order],torch.arange(len(labels),device=v.device)-starts[labels[order]]]=order.int()
    alternate=reps
    if config.partial_key_states:
        channels=torch.tensor([0,17,33,49],device=k.device)
        ks=k[:,patches][:,:,channels].permute(1,0,2).reshape(len(patches),-1).float()
        distance=(ks-ks[reps[labels]]).square().sum(-1);distance[reps]=-1.
        maxima=torch.full((len(reps),),-2.,device=k.device).scatter_reduce_(0,labels,distance,reduce='amax')
        ids=torch.arange(len(labels),device=k.device)
        candidates=ids.masked_fill((distance!=maxima[labels])|(ids==reps[labels]),len(labels))
        alternate=torch.full_like(reps,len(labels)).scatter_reduce_(0,labels,candidates,reduce='amin')
        alternate=torch.where(population>1,alternate,reps).contiguous()
    pair=(z[reps[labels]]+z[alternate[labels]])*.5
    return dict(width=width,members=members,shared_value_loss=True,operator_states=True,
        operator_loss=config.operator_loss,operator_partial=config.partial_key_states,
        operator_mass_penalty=config.operator_mass_penalty,operator_alternate=alternate,
        operator_weighting=config.operator_weighting,
        loss_patch_values=y[patches].contiguous(),loss_special_values=y[specials].contiguous(),
        operator_drop=z.square().sum(-1).contiguous(),
        operator_rep=(z-z[reps[labels]]).square().sum(-1).contiguous(),
        operator_alt=(z-z[alternate[labels]]).square().sum(-1).contiguous(),
        operator_pair=(z-pair).square().sum(-1).contiguous())


@triton.jit
def _statistics(P,LAB,MEM,Y,QUAL,DD,RR,AA,PP,OUT,
                N:tl.constexpr,F:tl.constexpr,W:tl.constexpr,D:tl.constexpr,C:tl.constexpr,M:tl.constexpr,
                CLUSTERS=None,COUNT=0,USE_IDS:tl.constexpr=False):
    g=tl.program_id(0);index=tl.program_id(1)*C+tl.arange(0,C);m=tl.arange(0,M);d=tl.arange(0,D)
    if USE_IDS:
        live=index<COUNT;c=tl.load(CLUSTERS+index,live,0)
    else:
        c=index;live=c<F
    key=tl.load(MEM+c[:,None]*W+m[None,:],live[:,None],-1)
    valid=live[:,None]&(key>=0);safe=tl.maximum(key,0)
    p=tl.load(P+g*N+safe,valid,0.)
    y=tl.load(Y+safe[:,:,None]*D+d[None,None,:],valid[:,:,None],0.)
    h=tl.load(QUAL+g*N+safe,valid,False);hp=tl.where(h,p,0.)
    moment=tl.sum(p[:,:,None]*y,1);hmoment=tl.sum(hp[:,:,None]*y,1)
    L:tl.constexpr=2*D+7;at=(g*F+c)*L
    tl.store(OUT+at[:,None]+d[None,:],moment,live[:,None])
    tl.store(OUT+at[:,None]+D+d[None,:],hmoment,live[:,None])
    tl.store(OUT+at+2*D,tl.sum(hp,1),live)
    drop=tl.load(DD+safe,valid,0.);rep=tl.load(RR+safe,valid,0.)
    alt=tl.load(AA+safe,valid,0.);pair=tl.load(PP+safe,valid,0.)
    tl.store(OUT+at+2*D+1,tl.sum(p*drop,1),live)
    tl.store(OUT+at+2*D+2,tl.sum(p*rep,1),live)
    tl.store(OUT+at+2*D+3,tl.sum(tl.where(h,0.,p*drop),1),live)
    tl.store(OUT+at+2*D+4,tl.sum(p*alt,1),live)
    tl.store(OUT+at+2*D+5,tl.sum(p*pair,1),live)
    tl.store(OUT+at+2*D+6,tl.sum(p*tl.sum(y*y,2),1),live)


@triton.jit
def _statistics_capacity(P,LAB,MEM,Y,QUAL,DD,RR,AA,PP,OUT,CLUSTERS,COUNT,
                         N:tl.constexpr,F:tl.constexpr,W:tl.constexpr,D:tl.constexpr,
                         C:tl.constexpr,M:tl.constexpr,COUNT_PTR:tl.constexpr):
    if COUNT_PTR:count=tl.load(COUNT)
    else:count=COUNT
    offset=0;step=tl.num_programs(1)*C
    while offset+tl.program_id(1)*C<count:
        _statistics(P,LAB,MEM,Y,QUAL,DD,RR,AA,PP,OUT,N,F,W,D,C,M,CLUSTERS+offset,count-offset,True)
        offset+=step


@triton.jit
def _losses(STATS,P,PI,POP,REP,ALT,Y,HC,ALLOW,MU,TOTAL,VAR,E,COST,OBJ,
            N:tl.constexpr,F:tl.constexpr,D:tl.constexpr,S:tl.constexpr,
            MOMENT:tl.constexpr,PENALTY:tl.constexpr,B:tl.constexpr,
            CALIBRATED:tl.constexpr=False,WEIGHTS=None):
    g=tl.program_id(0);c=tl.program_id(1)*B+tl.arange(0,B);valid=c<F;d=tl.arange(0,D)
    L:tl.constexpr=2*D+7;at=(g*F+c)*L
    m=tl.load(STATS+at[:,None]+d[None,:],valid[:,None],0.)
    h=tl.load(STATS+at[:,None]+D+d[None,:],valid[:,None],0.)
    hm=tl.load(STATS+at+2*D,valid,0.);pi=tl.load(PI+g*F+c,valid,0.)
    pop=tl.load(POP+c,valid,1.);hc=tl.load(HC+g*F+c,valid,0.)
    rep=tl.load(REP+c,valid,0);alt=tl.load(ALT+c,valid,0)
    pr=tl.load(P+g*N+rep,valid,0.);pa=tl.load(P+g*N+alt,valid,0.)
    yr=tl.load(Y+rep[:,None]*D+d[None,:],valid[:,None],0.)
    ya=tl.load(Y+alt[:,None]*D+d[None,:],valid[:,None],0.)
    mu=tl.load(MU+g*D+d);total=tl.load(TOTAL+g);variance=tl.load(VAR+g)
    for s in tl.static_range(S):
        if s==0:
            mass=tl.full((B,),0.,tl.float32);a=tl.full((B,D),0.,tl.float32);cost=tl.full((B,),0.,tl.float32)
        elif s==1:
            mass=pop*pr;a=mass[:,None]*yr;cost=tl.full((B,),1.,tl.float32)
        elif s==2:
            mass=pop/tl.maximum(hc,1.)*hm;a=pop[:,None]/tl.maximum(hc[:,None],1.)*h;cost=hc.to(tl.float32)
        elif s==3:
            mass=pi;a=m;cost=pop
        elif s==4:
            mass=pop*pa;a=mass[:,None]*ya;cost=tl.full((B,),1.,tl.float32)
        else:
            mass=pop*.5*(pr+pa);a=pop[:,None]*.5*(pr[:,None]*yr+pa[:,None]*ya);cost=tl.full((B,),2.,tl.float32)
        weight=tl.where(s==0,0.,tl.where(s==3,1.,pop/tl.maximum(cost,1.)))
        if CALIBRATED:
            if s!=0 and s!=3:
                ratio=tl.minimum(4.,tl.maximum(.25,pi/tl.maximum(mass,1.e-30)))
                mass=mass*ratio;a=a*ratio[:,None];weight=weight*ratio
        if MOMENT:
            delta=m-a-(pi-mass)[:,None]*mu[None,:]
            denominator=tl.maximum(total-pi+mass,1.e-8)
            error=(tl.sum(delta*delta,1)+PENALTY*variance*(pi-mass)*(pi-mass))/(denominator*denominator)
        else:
            if s==3:error=tl.full((B,),0.,tl.float32)
            else:
                slot=1 if s==0 else 2 if s==1 else 3 if s==2 else 4 if s==4 else 5
                error=tl.load(STATS+at+2*D+slot,valid,0.)
        ok=valid
        if s==2:ok=ok&(hc>0)
        if s==3:ok=ok&tl.load(ALLOW+g*F+c,valid,False)
        if s>=4:ok=ok&(pop>1)&(rep!=alt)
        error=tl.where(ok,tl.maximum(error,0.),float('inf'))
        index=(g*F+c)*S+s
        tl.store(E+index,error,valid);tl.store(COST+index,cost,valid);tl.store(OBJ+index,error,valid)
        if CALIBRATED:tl.store(WEIGHTS+index,weight,valid)


@triton.jit
def _qualify(SELECTED,LABEL,REP,ALT,OLD_QUAL,QUAL,HC,NEW_HC,PACKED,
             N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);valid=j<N
    c=tl.load(LABEL+j,valid,0);s=tl.load(SELECTED+g*F+c,valid,0)
    rep=tl.load(REP+c,valid,0);alt=tl.load(ALT+c,valid,0)
    old=tl.load(OLD_QUAL+g*N+j,valid,False)
    keep=tl.where(s==4,j==alt,tl.where(s==5,(j==rep)|(j==alt),old))
    tl.store(QUAL+g*N+j,keep,valid)
    c=tl.program_id(1)*B+tl.arange(0,B);valid=c<F
    s=tl.load(SELECTED+g*F+c,valid,0);hc=tl.load(HC+g*F+c,valid,0)
    mapped=tl.where(s==0,0,tl.where(s==1,1,tl.where(s==3,10,5)))
    tl.store(PACKED+g*F+c,mapped,valid)
    tl.store(NEW_HC+g*F+c,tl.where(s==4,1,tl.where(s==5,2,hc)),valid)


@triton.jit
def _emit_weights(IDS,LABEL,SELECTED,WEIGHTS,OUT,N:tl.constexpr,F:tl.constexpr,A:tl.constexpr,S:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    c=tl.load(LABEL+tl.maximum(key,0),valid,0);state=tl.load(SELECTED+g*F+c,valid,0)
    w=tl.load(WEIGHTS+(g*F+c)*S+state,valid,0.)
    tl.store(OUT+g*A+col,tl.where(valid,w,0.),col<A)


def select(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations,special_p):
    from . import reuse_inference_ops as rule
    from .shared_value_repair import execution_weights
    from saf3r.triton.reuse_compact import budget
    context=rule.gate(prior,pi,population,old_ids,labels,reps)
    context['remove_rep']=torch.zeros_like(context['remove_rep'])
    g,n=prior.shape;f=len(reps);d=geometry['loss_patch_values'].shape[-1]
    stats=torch.empty((g,f,2*d+7),device=prior.device)
    for index,(size,clusters) in enumerate(geometry['drop_bins']):
        use_ptr='drop_bin_counts' in geometry
        count_ptr=geometry['drop_bin_counts'][index] if use_ptr else len(clusters)
        if not len(clusters):continue
        c=min(16,max(1,128//size))
        programs=min(triton.cdiv(len(clusters),c),max(8,triton.cdiv(1024,g)))
        _statistics_capacity[(g,programs)](prior,labels,geometry['members'],geometry['loss_patch_values'],context['qualified'],
            geometry['operator_drop'],geometry['operator_rep'],geometry['operator_alt'],geometry['operator_pair'],stats,
            clusters,count_ptr,n,f,geometry['width'],d,c,size,use_ptr,num_warps=4,enable_fp_fusion=False)
    total=pi.sum(1)+special_p.sum(1)
    mu=(stats[:,:,:d].sum(1)+special_p@geometry['loss_special_values'])/total[:,None].clamp_min(1e-30)
    second=stats[:,:,2*d+6].sum(1)+special_p@geometry['loss_special_values'].square().sum(-1)
    variance=(second/total.clamp_min(1e-30)-mu.square().sum(-1)).clamp_min(0)
    count=6 if geometry['operator_partial'] else 4
    errors=torch.empty((g,f,count),device=prior.device);costs=torch.empty_like(errors);objective=torch.empty_like(errors)
    calibrated=geometry['operator_weighting']=='calibrated'
    state_weights=torch.empty_like(errors) if calibrated else None
    _losses[(g,triton.cdiv(f,32))](stats,prior,pi,population,reps,geometry['operator_alternate'],geometry['loss_patch_values'],
        context['eligible_count'],allowed,mu,total,variance,errors,costs,objective,n,f,d,count,
        geometry['operator_loss']=='normalized_moment',geometry['operator_mass_penalty'],32,calibrated,state_weights,num_warps=4,enable_fp_fusion=False)
    selected,_=budget(objective,costs,errors[:,:,0].amax(1)+1e-12,cap,iterations)
    qual=torch.empty_like(context['qualified']);amount=torch.empty_like(context['eligible_count']);packed=torch.empty_like(selected)
    _qualify[(g,triton.cdiv(n,256))](selected,labels,reps,geometry['operator_alternate'],context['qualified'],qual,
        context['eligible_count'],amount,packed,n,f,256,num_warps=4)
    original_hc=context['eligible_count'];context=dict(context,qualified=qual,eligible_count=amount)
    ids,real,record=rule.pack(packed,None,reps,labels,population,cap,context)
    if calibrated:
        weights=torch.empty(ids.shape,device=ids.device)
        _emit_weights[(g,triton.cdiv(ids.shape[1],256))](ids,labels,selected,state_weights,weights,n,f,ids.shape[1],count,256,num_warps=4)
    else:weights=execution_weights('cluster_population',packed,ids,reps,labels,population,prior,pi,amount)
    context['mass_error']=torch.zeros(g,device=prior.device)
    record.update(rule.diagnostics(errors,packed,None,old_ids,labels,reps,context))
    record['history_cap_discarded']=torch.zeros(g,device=prior.device)
    record['history_state_clusters']=(selected==2).sum(1)
    record['alternate_rep_clusters']=(selected==4).sum(1)
    record['partial_pair_clusters']=(selected==5).sum(1)
    record['reuse_edges']=(original_hc*(selected==2)).sum(1)
    record['state_5_clusters']=(selected==2).sum(1)
    return ids,real,weights,record
