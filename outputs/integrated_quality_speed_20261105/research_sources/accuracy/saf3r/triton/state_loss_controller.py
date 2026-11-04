"""QK-free four-state statistics for uniform unobserved cluster members."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate(IDS,PRIOR,PI,POP,LABEL,REP,OBS,QUAL,COUNT,AMOUNT,SEEN,
          N:tl.constexpr,F:tl.constexpr,A:tl.constexpr,ALL:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    c=tl.load(LABEL+tl.maximum(key,0),valid,0);rep=tl.load(REP+c,valid,-1)
    eligible=valid
    if not ALL:
        p=tl.load(PRIOR+g*N+tl.maximum(key,0),valid,0.)
        mass=tl.load(PI+g*F+c,valid,0.);pop=tl.load(POP+c,valid,1.)
        eligible=valid&((p>tl.div_rn(mass,pop))|(key==rep))
    tl.store(OBS+g*N+key,True,valid);tl.store(QUAL+g*N+key,eligible,valid)
    tl.atomic_add(COUNT+g*F+c,1,valid,sem='relaxed')
    tl.atomic_add(AMOUNT+g*F+c,1,eligible,sem='relaxed')
    tl.atomic_max(SEEN+g*F+c,1,valid&(key==rep),sem='relaxed')


def gate(prior,pi,old_ids,labels,reps,population,all_observed):
    g,n=prior.shape;f=len(reps)
    observed=torch.zeros((g,n),device=prior.device,dtype=torch.bool)
    qualified=torch.zeros_like(observed)
    count=torch.zeros((g,f),device=prior.device,dtype=torch.int32)
    amount=torch.zeros_like(count);seen=torch.zeros_like(count)
    _gate[(g,triton.cdiv(old_ids.shape[1],256))](old_ids,prior,pi,population,labels,reps,
        observed,qualified,count,amount,seen,n,f,old_ids.shape[1],all_observed,256,num_warps=4,enable_fp_fusion=False)
    return dict(observed=observed,qualified=qualified,history_count=count,eligible_count=amount,
        observed_rep=seen>0,remove_rep=torch.zeros_like(count,dtype=torch.bool))


@triton.jit(do_not_specialize=['COUNT'])
def _stats(PRIOR,PI,POP,REP,MEMBERS,Y,OBS,QUAL,CLUSTERS,OUT,
           N:tl.constexpr,F:tl.constexpr,W:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,
           RESIDUAL:tl.constexpr,COUNT,C:tl.constexpr,M:tl.constexpr,COUNT_PTR:tl.constexpr):
    if COUNT_PTR:COUNT=tl.load(COUNT)
    g=tl.program_id(0);index=tl.program_id(1)*C+tl.arange(0,C)
    c=tl.load(CLUSTERS+index,index<COUNT,0)
    m=tl.arange(0,M);d=tl.arange(0,D)
    key=tl.load(MEMBERS+c[:,None]*W+m[None,:],index[:,None]<COUNT,-1)
    valid=(index[:,None]<COUNT)&(key>=0);safe=tl.maximum(key,0)
    p=tl.load(PRIOR+g*N+safe,valid,0.)
    known=tl.load(OBS+g*N+safe,valid,False)&(g%STRIDE==0)
    qualified=tl.load(QUAL+g*N+safe,valid,False)&valid
    rep=tl.load(REP+c,index<COUNT,0)
    history=qualified
    if RESIDUAL:history=qualified&(key!=rep[:,None])
    mass=tl.load(PI+g*F+c,index<COUNT,0.);pop=tl.load(POP+c,index<COUNT,1.)
    known_mass=tl.sum(tl.where(known,p,0.),1);known_count=tl.sum(known.to(tl.int32),1)
    uniform=tl.maximum(0.,mass-known_mass)/tl.maximum(1.,pop-known_count)
    prob=tl.where(valid,tl.where(known,p,uniform[:,None]),0.)
    y=tl.load(Y+safe[:,:,None]*D+d[None,None,:],valid[:,:,None],0.)
    moment=tl.sum(prob[:,:,None]*y,1)
    hp=tl.where(history,prob,0.)
    hm=tl.sum(hp,1);hvec=tl.sum(hp[:,:,None]*y,1)
    norm=tl.sum(y*y,2)
    a=tl.sum(prob*norm,1);ah=tl.sum(hp*norm,1)
    hc=tl.sum(history.to(tl.int32),1);available=tl.sum(qualified.to(tl.int32),1)
    L:tl.constexpr=2*D+6
    offset=(g*F+c)*L
    tl.store(OUT+offset[:,None]+d[None,:],moment,index[:,None]<COUNT)
    tl.store(OUT+offset[:,None]+D+d[None,:],hvec,index[:,None]<COUNT)
    tl.store(OUT+offset+2*D,a,index<COUNT);tl.store(OUT+offset+2*D+1,ah,index<COUNT)
    tl.store(OUT+offset+2*D+2,hm,index<COUNT);tl.store(OUT+offset+2*D+3,uniform,index<COUNT)
    tl.store(OUT+offset+2*D+4,hc.to(tl.float32),index<COUNT)
    tl.store(OUT+offset+2*D+5,available.to(tl.float32),index<COUNT)


@triton.jit
def _loss(STATS,PI,POP,REP,Y,MU,ALLOW,ERROR,COST,OBJECTIVE,
          F:tl.constexpr,D:tl.constexpr,MODE:tl.constexpr,RESIDUAL:tl.constexpr,C:tl.constexpr):
    g=tl.program_id(0);c=tl.program_id(1)*C+tl.arange(0,C);valid=c<F;d=tl.arange(0,D)
    L:tl.constexpr=2*D+6;offset=(g*F+c)*L
    moment=tl.load(STATS+offset[:,None]+d[None,:],valid[:,None],0.)
    hvec=tl.load(STATS+offset[:,None]+D+d[None,:],valid[:,None],0.)
    a=tl.load(STATS+offset+2*D,valid,0.);ah=tl.load(STATS+offset+2*D+1,valid,0.)
    hm=tl.load(STATS+offset+2*D+2,valid,0.);hc=tl.load(STATS+offset+2*D+4,valid,0.)
    available=tl.load(STATS+offset+2*D+5,valid,0.)
    pi=tl.load(PI+g*F+c,valid,0.);pop=tl.load(POP+c,valid,0.)
    rep=tl.load(REP+c,valid,0);yr=tl.load(Y+rep[:,None]*D+d[None,:],valid[:,None],0.)
    mu=tl.load(MU+g*D+d)
    er=tl.maximum(0.,a-2*tl.sum(yr*moment,1)+pi*tl.sum(yr*yr,1))
    centered=tl.maximum(0.,a-2*tl.sum(mu[None,:]*moment,1)+pi*tl.sum(mu*mu,0))
    if RESIDUAL:
        eh=tl.maximum(0.,er-(ah-2*tl.sum(yr*hvec,1)+hm*tl.sum(yr*yr,1)))
        htarget=(pi-hm)[:,None]*yr+hvec
        hcost=hc+1
    else:
        hy=hvec/tl.maximum(hm[:,None],1.e-30)
        remainder=tl.maximum(pi-hm,0.)
        eh=tl.maximum(0.,a-ah-2*tl.sum(hy*(moment-hvec),1)+remainder*tl.sum(hy*hy,1))
        htarget=pi[:,None]*hy
        hcost=hc
    drop=a;rep_error=er;hist_error=eh
    if MODE==1:drop=centered
    elif MODE==2:
        drop=centered
        if not RESIDUAL:
            hist_error=tl.maximum(0.,a-ah-2*tl.sum(hy*(moment-hvec),1)+remainder*ah/tl.maximum(hm,1.e-30))
    elif MODE==3 or MODE==4:
        delta_d=moment-pi[:,None]*mu[None,:]
        delta_r=moment-pi[:,None]*yr;delta_h=moment-htarget
        drop=tl.sum(delta_d*delta_d,1)
        rep_error=tl.sum(delta_r*delta_r,1)
        hist_error=tl.sum(delta_h*delta_h,1)
        if MODE==4:
            drop+=.1*pi*centered;rep_error+=.1*pi*er;hist_error+=.1*pi*eh
    hist_ok=available>0
    if not RESIDUAL:hist_ok=hist_ok&(hm>1.e-30)
    hist_error=tl.where(hist_ok,hist_error,float('inf'))
    allow=tl.load(ALLOW+g*F+c,valid,False)
    exact_error=tl.where(allow,0.,float('inf'))
    at=(g*F+c)*4
    tl.store(ERROR+at,drop,valid);tl.store(ERROR+at+1,rep_error,valid)
    tl.store(ERROR+at+2,hist_error,valid);tl.store(ERROR+at+3,exact_error,valid)
    tl.store(COST+at,0.,valid);tl.store(COST+at+1,1.,valid)
    tl.store(COST+at+2,hcost,valid);tl.store(COST+at+3,pop,valid)
    tl.store(OBJECTIVE+at,drop-rep_error,valid);tl.store(OBJECTIVE+at+1,0.,valid)
    tl.store(OBJECTIVE+at+2,hist_error-rep_error,valid);tl.store(OBJECTIVE+at+3,exact_error-rep_error,valid)


@triton.jit
def _weights(IDS,LABEL,REP,SELECTED,PI,PRIOR,OBS,POP,STATS,RATIO,OUT,REUSED,
             N:tl.constexpr,F:tl.constexpr,A:tl.constexpr,D:tl.constexpr,STRIDE:tl.constexpr,
             RESIDUAL:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);tile=tl.program_id(1);col=tile*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0);safe=tl.maximum(key,0)
    c=tl.load(LABEL+safe,valid,0);rep=tl.load(REP+c,valid,0)
    state=tl.load(SELECTED+g*F+c,valid,0);pi=tl.load(PI+g*F+c,valid,0.)
    observed=tl.load(OBS+g*N+safe,valid,False)
    L:tl.constexpr=2*D+6;off=(g*F+c)*L
    hm=tl.load(STATS+off+2*D+2,valid,0.);uniform=tl.load(STATS+off+2*D+3,valid,0.)
    hc=tl.load(STATS+off+2*D+4,valid,0.);pop=tl.load(POP+c,valid,1.)
    weight=tl.full((B,),1.,tl.float32)
    if g%STRIDE!=0:
        if RESIDUAL:
            weight=tl.where((key==rep)&(state==5),pop-hc,tl.where(state==1,pop,1.))
        else:
            ratio=tl.load(RATIO+pop.to(tl.int32)*513+tl.maximum(hc,1.).to(tl.int32),valid,1.)
            weight=tl.where(state==1,pop,tl.where(state==5,ratio,1.))
    else:
        p=tl.load(PRIOR+g*N+safe,valid,0.);own=tl.where(observed,p,uniform)
        if RESIDUAL:
            assigned=tl.where(state==5,pi-hm,pi)
            calibrated=tl.div_rn(assigned,tl.maximum(own,1.e-30))
            weight=tl.where((key==rep)&((state==1)|(state==5)),calibrated,1.)
        else:
            rep_w=tl.div_rn(pi,tl.maximum(own,1.e-30));history_w=tl.div_rn(pi,tl.maximum(hm,1.e-30))
            weight=tl.where(state==1,rep_w,tl.where(state==5,history_w,1.))
    tl.store(OUT+g*A+col,tl.where(valid,weight,0.),col<A)
    tl.store(REUSED+g*tl.cdiv(A,B)+tile,tl.sum((observed&valid).to(tl.int32),0))


def statistics(prior,pi,labels,reps,population,geometry,context):
    g,n=prior.shape;f=len(reps);d=geometry['loss_patch_values'].shape[-1]
    result=torch.empty((g,f,2*d+6),device=prior.device)
    residual=geometry['loss_operator']=='residual_rep'
    for i,(size,clusters) in enumerate(geometry['drop_bins']):
        use_ptr='drop_bin_counts' in geometry
        count=geometry['drop_bin_counts'][i] if use_ptr else len(clusters)
        c=min(16,max(1,128//size))
        _stats[(g,triton.cdiv(len(clusters),c))](prior,pi,population,reps,geometry['members'],geometry['loss_patch_values'],
            context['observed'],context['qualified'],clusters,result,n,f,geometry['width'],d,
            geometry['loss_history_stride'],residual,count,c,size,use_ptr,num_warps=4,enable_fp_fusion=False)
    return result


def state_losses(prior,pi,allowed,population,reps,geometry,stats,special_p):
    g,f=pi.shape;d=geometry['loss_patch_values'].shape[-1]
    total=pi.sum(1)+special_p.sum(1)
    mu=(stats[:,:,:d].sum(1)+special_p@geometry['loss_special_values'])/total[:,None].clamp_min(1e-30)
    errors=torch.empty((g,f,4),device=prior.device);costs=torch.empty_like(errors);objective=torch.empty_like(errors)
    names={'group_centroid':0,'group_centered':1,'group_mixture':2,'group_moment':3,'group_moment_variance':4,
        'residual_energy':0,'centered_residual_energy':1,'transport_moment':3}
    _loss[(g,triton.cdiv(f,32))](stats,pi,population,reps,geometry['loss_patch_values'],mu,allowed,errors,costs,objective,
        f,d,names[geometry['state_loss_variant']],geometry['loss_operator']=='residual_rep',32,num_warps=4,enable_fp_fusion=False)
    return errors,costs,objective


def execution_weights(selected,ids,prior,pi,labels,reps,population,geometry,context,stats):
    from .shared_value_repair import ratio_table
    g,a=ids.shape;d=geometry['loss_patch_values'].shape[-1]
    out=torch.empty((g,a),device=ids.device)
    partial=torch.empty((g,triton.cdiv(a,256)),device=ids.device,dtype=torch.int32)
    _weights[(g,triton.cdiv(a,256))](ids,labels,reps,selected,pi,prior,context['observed'],population,stats,
        ratio_table(ids.device),out,partial,prior.shape[1],len(reps),a,d,geometry['loss_history_stride'],
        geometry['loss_operator']=='residual_rep',256,num_warps=4,enable_fp_fusion=False)
    return out,partial.sum(1)
