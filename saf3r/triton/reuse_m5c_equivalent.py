"""M5c fusion with unchanged probabilities and explicit head reduction."""
import torch
import triton
import triton.language as tl


@triton.jit
def _edges(Q,K,IDS,ACTUAL,POS,GRID,SIZES,CUR,PREV,CORR,LOGZ,W,OUT,
           N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,A:tl.constexpr,R:tl.constexpr,
           SCALE:tl.constexpr,EPS:tl.constexpr,BD:tl.constexpr,T:tl.constexpr):
    tile=tl.program_id(0);g=tl.program_id(1);h=tl.program_id(2)
    p=tl.arange(0,16);a=tile*T+tl.arange(0,T);d=tl.arange(0,BD)
    key=tl.load(IDS+g*A+a,a<A,-1);actual=tl.load(ACTUAL+g*A+a,a<A,-1)
    valid=(a<A)&(key>=0)
    qi=tl.load(POS+g*4+p,p<4,0)
    q=tl.load(Q+h*N*D+qi[:,None]*D+d[None,:],(p[:,None]<4)&(d[None,:]<D),0.)
    k=tl.load(K+h*N*D+tl.maximum(actual,0)[None,:]*D+d[:,None],valid[None,:]&(d[:,None]<D),0.)
    raw=tl.dot(q,k,allow_tf32=False)*SCALE
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    size=tl.load(SIZES+region,valid,1.)
    ix=((g*H+h)*4+p[:,None])*R+region[None,:];mask=(p[:,None]<4)&valid[None,:]
    cur=tl.load(CUR+ix,mask,0.);prev=tl.load(PREV+ix,mask,0.);corr=tl.load(CORR+ix,mask,1.)
    ratio=(cur+EPS)/(prev+EPS)
    confidence=tl.minimum(1.,tl.maximum(0.,tl.where(ratio<1.,ratio,1./(ratio*ratio))))
    logz=tl.load(LOGZ+(g*H+h)*4+p,p<4,0.)
    probability=tl.exp(raw-logz[:,None])
    conditional=probability/tl.maximum(corr,EPS)
    predicted=cur*(confidence*conditional+(1.-confidence)/size[None,:])
    weight=tl.load(W+g*4+p,p<4,0.)
    value=tl.sum(tl.where(mask,predicted*weight[:,None],0.),0)
    tl.store(OUT+(g*H+h)*A+a,value,a<A)


@triton.jit
def _write(EDGE,IDS,PRIOR,A:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);a=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+a,a<A,-1);value=tl.load(EDGE+g*A+a,a<A,0.)
    tl.store(PRIOR+g*P+key,value,(a<A)&(key>=0))


def predict(q,k,ids,actual,obs,old,grid,sizes,prior,scale,eps):
    g,a=ids.shape;h,n,d=q.shape
    # Preserve separate per-head probe sums, then use the same Torch head mean.
    # No nondeterministic atomic summation across heads.
    per_head=torch.empty((g,h,a),device=q.device,dtype=torch.float32)
    _edges[(triton.cdiv(a,64),g,h)](q,k,ids,actual,obs['positions'],grid,sizes,
        obs['coarse_grid_probability'].contiguous(),old['previous_coarse_grid_probability'].contiguous(),
        old['previous_corrected_grid_probability'].contiguous(),old['history_dense_log_normalizer'].contiguous(),
        obs['weights'].contiguous(),per_head,n,h,d,a,len(sizes),scale,eps,triton.next_power_of_2(d),64,
        num_warps=4,enable_fp_fusion=False)
    edge=per_head.mean(1)
    _write[(g,triton.cdiv(a,256))](edge,ids,prior,a,len(grid),256,num_warps=4)
    return edge


@triton.jit
def _correct(RAW,LOGZ,IDS,GRID,CORRECTED,A:tl.constexpr,H:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    a=tl.program_id(0)*B+tl.arange(0,B);hp=tl.program_id(1);g=tl.program_id(2)
    key=tl.load(IDS+g*A+a,a<A,-1);valid=(a<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    raw=tl.load(RAW+(g*H*4+hp)*A+a,valid,-float('inf'))
    logz=tl.load(LOGZ+g*H*4+hp)
    probability=tl.exp(raw-logz)
    tl.atomic_add(CORRECTED+(g*H*4+hp)*R+region,probability,valid,sem='relaxed')


def correct(raw,logz,ids,grid,base):
    g,h,_,a=raw.shape
    corrected=base.contiguous()
    _correct[(triton.cdiv(a,256),h*4,g)](raw,logz,ids,grid,corrected,a,h,base.shape[-1],256,num_warps=4)
    return corrected
