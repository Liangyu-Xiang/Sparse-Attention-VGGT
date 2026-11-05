"""Opt-in A/B History losses; shared Rep statistics and original four-state budget.

A: History support H union Rep. Exact historical non-Rep keys have weight1;
Rep weight is population-(support_count-1). This is a representation proxy,
not a normalized attention-output loss. B changes only singleton-H loss;
its support and cluster_population weights remain the original ones.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _corrections(IDS,P,U,GRID,LABEL,TD,TR,QUAL,AMOUNT,D,R,KD,KR,DH,PH,
                 N:tl.constexpr,F:tl.constexpr,REGIONS:tl.constexpr,A:tl.constexpr,B:tl.constexpr,
                 MODE:tl.constexpr,DIAG:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0);safe=tl.maximum(key,0)
    c=tl.load(LABEL+safe,valid,0);grid=tl.load(GRID+safe,valid,0)
    p=tl.load(P+g*N+safe,valid,0.);u=tl.load(U+g*REGIONS+grid,valid,0.);delta=p-u
    a=tl.load(TD+safe,valid,0.);b=tl.load(TR+safe,valid,0.)
    qualified=valid&tl.load(QUAL+g*N+safe,valid,False)
    tl.atomic_add(D+g*F+c,delta*a,valid&(delta!=0.),sem='relaxed')
    tl.atomic_add(R+g*F+c,delta*b,valid&(delta!=0.),sem='relaxed')
    if MODE!=1:tl.atomic_add(KD+g*F+c,p*a,qualified,sem='relaxed')
    if MODE!=0:tl.atomic_add(KR+g*F+c,p*b,qualified,sem='relaxed')
    if MODE==2:
        singleton=qualified&(tl.load(AMOUNT+g*F+c,valid,0)==1)
        tl.atomic_add(DH+g*F+c,b,singleton,sem='relaxed')
    if DIAG:tl.atomic_add(PH+g*F+c,p,qualified,sem='relaxed')

@triton.jit
def _repair(E,COST,OBJ,PI,KR,DH,OLD_M,SEEN,REP,POP,QUAL,NEW_M,
            N:tl.constexpr,F:tl.constexpr,B:tl.constexpr,MODE:tl.constexpr):
    g=tl.program_id(0);c=tl.program_id(1)*B+tl.arange(0,B);valid=c<F;offset=(g*F+c)*4
    rep=tl.load(E+offset+1,valid,0.);m=tl.load(OLD_M+g*F+c,valid,0)
    kept=tl.load(KR+g*F+c,valid,0.)
    if MODE==1:
        seen=tl.load(SEEN+g*F+c,valid,False)
        amount=m+((m>0)&~seen).to(tl.int32)
        hist=tl.where(m==tl.load(POP+c,valid,0.),0.,tl.maximum(0.,rep-kept))
        r=tl.load(REP+c,valid,0)
        # QUAL is a fresh gate result, shared by no other selectable state.
        tl.store(QUAL+g*N+r,True,valid&(m>0))
    else:
        amount=m;pi=tl.load(PI+g*F+c,valid,0.);distance=tl.load(DH+g*F+c,valid,0.)
        old=tl.load(E+offset+2,valid,0.)
        hist=tl.where(m==1,tl.maximum(0.,rep+pi*distance-2.*kept),old)
    hist=tl.where(m>0,hist,float('inf'))
    tl.store(NEW_M+g*F+c,amount,valid)
    tl.store(E+offset+2,hist,valid);tl.store(COST+offset+2,amount.to(tl.float32),valid)
    tl.store(OBJ+offset+2,hist-rep,valid)

@triton.jit
def _weights_A(IDS,LABEL,REP,SEL,POP,AMOUNT,OUT,N:tl.constexpr,F:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    c=tl.load(LABEL+tl.maximum(key,0),valid,0);rep=tl.load(REP+c,valid,0)
    sel=tl.load(SEL+g*F+c,valid,0);pop=tl.load(POP+c,valid,1.)
    count=tl.load(AMOUNT+g*F+c,valid,0)
    weight=tl.where(sel==1,pop,tl.where((sel==5)&(key==rep),pop-count+1.,1.))
    tl.store(OUT+g*A+col,tl.where(valid,weight,0.),col<A)


def select(prior,pi,old_ids,allowed,z,reps,labels,population,geometry,cap,iterations,uniform):
    from . import reuse_inference_ops as rule
    from .shared_value_repair import execution_weights
    from saf3r.triton import reuse_shared_value_states as shared
    from saf3r.triton.reuse_compact import budget
    mode={'legacy':0,'A':1,'B':2}[geometry['rep_history_loss']];diag=geometry.get('rep_history_diagnostics',False)
    calibrated=geometry.get('rep_history_execution','population')=='prior_residual'
    context=rule.gate(prior,pi,population,old_ids,labels,reps)
    context['remove_rep']=torch.zeros_like(context['remove_rep'])
    original_amount=context['eligible_count'];g,n=prior.shape;f=len(reps)
    d=torch.zeros((g,f),device=prior.device);r=torch.zeros_like(d);kd=torch.zeros_like(d)
    kr=torch.zeros_like(d);dh=torch.zeros_like(d);ph=torch.zeros_like(d)
    _corrections[(g,triton.cdiv(old_ids.shape[1],256))](old_ids,prior,uniform,geometry['shared_grid'],labels,
        geometry['shared_token_drop'],geometry['shared_token_rep'],context['qualified'],original_amount,
        d,r,kd,kr,dh,ph,n,f,uniform.shape[1],old_ids.shape[1],256,mode,diag or calibrated,num_warps=4,enable_fp_fusion=False)
    errors,costs=shared.states(uniform,allowed,population,geometry,context,(d,r,kd))
    amount=original_amount
    if mode:
        amount=torch.empty_like(original_amount)
        _repair[(g,triton.cdiv(f,256))](errors,costs,geometry['objective'],pi,kr,dh,original_amount,
            context['observed_rep'],reps,population,context['qualified'],amount,n,f,256,mode,num_warps=4,enable_fp_fusion=False)
        context['eligible_count']=amount
    selected,_=budget(geometry['objective'],costs,errors[:,:,0].amax(1)+1e-12,cap,iterations)
    selected=rule.decode(selected)
    ids,real,record=rule.pack(selected,None,reps,labels,population,cap,context)
    if mode==1:
        weights=torch.empty(ids.shape,device=ids.device)
        if calibrated:
            from .rep_history_prior_weights import weights_A
            weights_A(ids,labels,reps,selected,prior,pi,ph,context['observed_rep'],weights)
        else:
            _weights_A[(g,triton.cdiv(ids.shape[1],256))](ids,labels,reps,selected,population,amount,weights,n,f,ids.shape[1],256,num_warps=4)
    else:weights=execution_weights('cluster_population',selected,ids,reps,labels,population,prior,pi,amount)
    context['mass_error']=torch.zeros(g,device=prior.device)
    record.update(rule.diagnostics(errors,selected,None,old_ids,labels,reps,context))
    record['history_cap_discarded']=torch.zeros(g,device=prior.device)
    if diag:
        chosen=selected==5;input_rep=chosen&context['observed_rep']
        support_rep=chosen if mode==1 else input_rep
        # A adds Rep only when absent from original H; its prior mass is explicit.
        support_mass=ph
        if mode==1:
            support_mass=ph+prior[:,reps]*(~context['observed_rep'])*(original_amount>0)
        record.update(history_chosen_clusters=chosen.sum(1),history_input_keys=(original_amount*chosen).sum(1),
            history_support_keys=(amount*chosen).sum(1),history_input_rep_clusters=input_rep.sum(1),
            history_input_only_rep_clusters=(input_rep&(original_amount==1)).sum(1),
            history_support_rep_clusters=support_rep.sum(1),history_support_only_rep_clusters=(support_rep&(amount==1)).sum(1),
            history_selected_input_probability=(ph*chosen).sum(1),history_selected_support_probability=(support_mass*chosen).sum(1),
            history_selected_cluster_probability=(pi*chosen).sum(1),
            drop_key_cost=torch.zeros(g,device=prior.device),rep_key_cost=(selected==1).sum(1),
            history_key_cost=(amount*chosen).sum(1),exact_key_cost=(population[None]*(selected==10)).sum(1),
            singleton_history_candidates=(original_amount==1).sum(1))
    return ids,real,weights,record
