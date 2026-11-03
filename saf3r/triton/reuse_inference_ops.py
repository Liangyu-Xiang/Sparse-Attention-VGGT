"""Fused indexing/masks for the unchanged four-state algorithm."""
import triton
import triton.language as tl

@triton.jit
def _gate(IDS,PRIOR,PI,POP,LABEL,REP,COUNT,SEEN,AMOUNT,QUAL,N:tl.constexpr,F:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    c=tl.load(LABEL+tl.maximum(key,0),valid,0);rep=tl.load(REP+c,valid,-1)
    own=tl.load(PRIOR+g*N+tl.maximum(key,0),valid,0.)
    mass=tl.load(PI+g*F+c,valid,0.);pop=tl.load(POP+c,valid,1.)
    eligible=valid&((own>tl.div_rn(mass,pop))|(key==rep))
    tl.atomic_add(COUNT+g*F+c,1,valid,sem='relaxed')
    tl.atomic_max(SEEN+g*F+c,1,valid&(key==rep),sem='relaxed')
    tl.atomic_add(AMOUNT+g*F+c,1,eligible,sem='relaxed')
    tl.store(QUAL+g*N+key,eligible,valid)

@triton.jit
def _flags(SEL,LABEL,REP,QUAL,FLAGS,N:tl.constexpr,F:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);valid=j<N
    c=tl.load(LABEL+j,valid,0);state=tl.load(SEL+g*F+c,valid,0)
    rep=tl.load(REP+c,valid,-1);qualified=tl.load(QUAL+g*N+j,valid,False)
    flag=(state==10)|((state==1)&(j==rep))|((state==5)&qualified)
    tl.store(FLAGS+g*N+j,flag.to(tl.int32),valid)

@triton.jit
def _validate(SEL,AMOUNT,POP,REMOVE,PART,INVALID,F:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);tile=tl.program_id(1);c=tile*B+tl.arange(0,B);valid=c<F
    state=tl.load(SEL+g*F+c,valid,0);amount=tl.load(AMOUNT+g*F+c,valid,0);pop=tl.load(POP+c,valid,0.).to(tl.int32)
    remove=tl.load(REMOVE+g*F+c,valid,False)
    expected=tl.where(state==1,1,tl.where(state==5,amount,tl.where(state==10,pop,0)))
    bad=((state==1)&remove)|~((state==0)|(state==1)|(state==5)|(state==10))
    tl.store(PART+g*T+tile,tl.sum(tl.where(valid,expected,0),0))
    tl.store(INVALID+g*T+tile,tl.sum((valid&bad).to(tl.int32),0))

@triton.jit
def _mass(SEL,LABEL,REP,OWNER,PRIOR,MASS,COVER,N:tl.constexpr,F:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);tile=tl.program_id(1);j=tile*B+tl.arange(0,B);valid=j<N
    c=tl.load(LABEL+j,valid,0);state=tl.load(SEL+g*F+c,valid,0);rep=tl.load(REP+c,valid,0);owner=tl.load(OWNER+g*N+j,valid,-1)
    dest=tl.where(state==10,j,tl.where(state==1,rep,tl.where(state==5,owner,-1)))
    covered=valid&(dest>=0);p=tl.load(PRIOR+g*N+j,valid,0.)
    tl.atomic_add(MASS+g*(N+1)+tl.maximum(dest,0),p,covered,sem='relaxed')
    tl.store(COVER+g*T+tile,tl.sum(tl.where(covered,p,0.),0))

@triton.jit
def _weights(IDS,PRIOR,MASS,OUT,ASSIGNED,N:tl.constexpr,A:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);tile=tl.program_id(1);col=tile*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    own=tl.load(PRIOR+g*N+tl.maximum(key,0),valid,1.)
    mass=tl.load(MASS+g*(N+1)+tl.maximum(key,0),valid,0.)
    w=tl.maximum(tl.div_rn(mass,tl.maximum(own,1.e-30)),1.)
    tl.store(OUT+g*A+col,tl.where(valid,w,0.),col<A)
    tl.store(ASSIGNED+g*T+tile,tl.sum(mass,0))
