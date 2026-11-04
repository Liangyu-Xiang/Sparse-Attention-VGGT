"""Apply one-Probe cached conditionals only to their own sampled Query Groups."""
import triton
import triton.language as tl


@triton.jit
def _predict(CACHE,IDS,GRID,SIZE,CUR,PREV,PRIOR,
 A:tl.constexpr,R:tl.constexpr,N:tl.constexpr,STRIDE:tl.constexpr,B:tl.constexpr,EPS:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);sg=tl.program_id(1)
    key=tl.load(IDS+sg*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0);size=tl.load(SIZE+region,valid,1.)
    current=tl.load(CUR+sg*R+region,valid,0.);previous=tl.load(PREV+sg*R+region,valid,0.)
    ratio=(current+EPS)/(previous+EPS)
    beta=tl.minimum(1.,tl.where(ratio<1.,ratio,1./(ratio*ratio)))
    logp=tl.load(CACHE+sg*A+col,valid,-float('inf')).to(tl.float32)
    value=current*(beta*tl.exp(logp)+(1.-beta)/size)
    tl.store(PRIOR+sg*STRIDE*N+key,value,valid)


def predict(cache,ids,grid,sizes,current,previous,prior,stride,eps):
    _predict[(triton.cdiv(ids.shape[1],256),len(ids))](cache,ids,grid,sizes,current,previous,prior,ids.shape[1],len(sizes),len(grid),stride,256,eps,num_warps=4)
