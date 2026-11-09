"""Apply cached 2D RoPE and materialize head-major Q/K in one launch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(Q,K,V,POS,COS,SIN,QOUT,KOUT,VOUT,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
            QH:tl.constexpr,QN:tl.constexpr,KH:tl.constexpr,KN:tl.constexpr,VH:tl.constexpr,VN:tl.constexpr,B:tl.constexpr,COPY_VALUE:tl.constexpr):
    token=tl.program_id(0)*B+tl.arange(0,B)
    head=tl.program_id(1)
    feature=tl.arange(0,D)
    axis=feature//(D//2)
    local=feature%(D//2)
    partner=axis*(D//2)+(local+D//4)%(D//2)
    pos=tl.load(POS+2*token[:,None]+axis[None,:],token[:,None]<N,0)
    table=pos*(D//2)+local[None,:]
    cosine=tl.load(COS+table).to(tl.float32)
    sine=tl.load(SIN+table).to(tl.float32)
    sign=tl.where(local<D//4,-1.,1.)
    valid=token[:,None]<N
    dst=(head*N+token[:,None])*D+feature[None,:]
    if COPY_VALUE:
        value=tl.load(V+head*VH+token[:,None]*VN+feature[None,:],valid,0)
        tl.store(VOUT+dst,value,valid)
    for which in tl.static_range(2):
        if which==0:
            src=Q+head*QH+token[:,None]*QN
            out=QOUT
        else:
            src=K+head*KH+token[:,None]*KN
            out=KOUT
        dtype=src.dtype.element_ty
        value=tl.load(src+feature[None,:],valid,0).to(tl.float32)
        rotated=tl.load(src+partner[None,:],valid,0).to(tl.float32)*sign[None,:]
        first=(value*cosine).to(dtype).to(tl.float32)
        second=(rotated*sine).to(dtype).to(tl.float32)
        tl.store(out+dst,(first+second).to(dtype),valid)


def apply(q,k,positions,rope,*,v=None):
    h,n,d=q.shape
    cosine,sine=rope._compute_frequency_components(
        d//2,int(positions.max())+1,q.device,q.dtype)
    qo=torch.empty((h,n,d),device=q.device,dtype=q.dtype if v is None else v.dtype)
    ko=torch.empty_like(qo)
    vo=torch.empty_like(qo) if v is not None else qo
    src_v=q if v is None else v
    _rotate[(triton.cdiv(n,32),h)](
        q,k,src_v,positions.contiguous(),cosine,sine,qo,ko,vo,n,h,d,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),src_v.stride(0),src_v.stride(1),32,v is not None,
        num_warps=4,enable_fp_fusion=False)
    return (qo,ko) if v is None else (qo,ko,vo)


@triton.jit
def _moments(SRC, VALID, B:tl.constexpr, MODE:tl.constexpr):
    # Match the reference CUDA LayerNorm reduction for 64 aligned features:
    # 16 vectors of four values, followed by a descending warp reduction.
    lane=tl.arange(0,16)
    mean=tl.full((B,16),0.,tl.float32)
    m2=tl.full((B,16),0.,tl.float32)
    for j in tl.static_range(4):
        value=tl.load(SRC+lane[None,:]*4+j,VALID,0.).to(tl.float32)
        delta=value-mean
        mean=tl.fma(delta,1./(j+1),mean)
        m2=tl.fma(delta,value-mean,m2)
    count=4.
    for step in tl.static_range(4):
        offset=8>>step
        index=tl.broadcast_to(((lane+offset)%16)[None,:],(B,16))
        other_mean=tl.gather(mean,index,1)
        other_m2=tl.gather(m2,index,1)
        delta=mean-other_mean
        if MODE==2:
            mean=tl.fma(.5,mean,.5*other_mean)
        else:
            mean=tl.fma(.5,other_mean,.5*mean)
        m2=tl.fma(delta*delta*count,.5,m2+other_m2)
        count*=2.
    return tl.sum(tl.where(lane[None,:]==0,mean,0.),1),tl.sum(tl.where(lane[None,:]==0,m2,0.),1)/64.


@triton.jit
def _norm_rotate(Q,K,V,QW,QB,KW,KB,POS,COS,SIN,QOUT,KOUT,VOUT,N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
            QH:tl.constexpr,QN:tl.constexpr,KH:tl.constexpr,KN:tl.constexpr,VH:tl.constexpr,VN:tl.constexpr,B:tl.constexpr,COPY_VALUE:tl.constexpr,EPSQ:tl.constexpr,EPSK:tl.constexpr,NORM_BF16:tl.constexpr,WELFORD:tl.constexpr):
    token=tl.program_id(0)*B+tl.arange(0,B)
    head=tl.program_id(1)
    feature=tl.arange(0,D)
    axis=feature//(D//2)
    local=feature%(D//2)
    partner=axis*(D//2)+(local+D//4)%(D//2)
    pos=tl.load(POS+2*token[:,None]+axis[None,:],token[:,None]<N,0)
    table=pos*(D//2)+local[None,:]
    cosine=tl.load(COS+table).to(tl.float32)
    sine=tl.load(SIN+table).to(tl.float32)
    sign=tl.where(local<D//4,-1.,1.)
    valid=token[:,None]<N
    dst=(head*N+token[:,None])*D+feature[None,:]
    if COPY_VALUE:
        value=tl.load(V+head*VH+token[:,None]*VN+feature[None,:],valid,0)
        tl.store(VOUT+dst,value,valid)
    for which in tl.static_range(2):
        if which==0:
            src=Q+head*QH+token[:,None]*QN
            out=QOUT
            weight,bias,eps=QW,QB,EPSQ
        else:
            src=K+head*KH+token[:,None]*KN
            out=KOUT
            weight,bias,eps=KW,KB,EPSK
        dtype=tl.bfloat16 if NORM_BF16 else tl.float32
        value=tl.load(src+feature[None,:],valid,0).to(tl.float32)
        if WELFORD:
            center,variance=_moments(src,valid,B,WELFORD)
        else:
            center=tl.sum(value,1)/D
            residual=value-center[:,None]
            variance=tl.sum(residual*residual,1)/D
        residual=value-center[:,None]
        inv=tl.rsqrt(variance+eps)
        rotated=tl.load(src+partner[None,:],valid,0).to(tl.float32)
        w=tl.load(weight+feature).to(tl.float32)
        b=tl.load(bias+feature).to(tl.float32)
        rw=tl.load(weight+partner).to(tl.float32)
        rb=tl.load(bias+partner).to(tl.float32)
        value=tl.fma(residual*inv[:,None],w[None,:],b[None,:]).to(dtype).to(tl.float32)
        rotated=tl.fma((rotated-center[:,None])*inv[:,None],rw[None,:],rb[None,:]).to(dtype).to(tl.float32)*sign[None,:]
        first=(value*cosine).to(dtype).to(tl.float32)
        second=(rotated*sine).to(dtype).to(tl.float32)
        tl.store(out+dst,(first+second).to(dtype),valid)



def apply_norm(q,k,v,positions,rope,q_norm,k_norm,normalized_dtype,*,welford=0):
    h,n,d=q.shape
    cosine,sine=rope._compute_frequency_components(
        d//2,int(positions.max())+1,q.device,normalized_dtype)
    qo=torch.empty((h,n,d),device=q.device,dtype=v.dtype)
    ko=torch.empty_like(qo)
    vo=torch.empty_like(qo)
    _norm_rotate[(triton.cdiv(n,32),h)](
        q,k,v,q_norm.weight,q_norm.bias,k_norm.weight,k_norm.bias,
        positions.contiguous(),cosine,sine,qo,ko,vo,n,h,d,
        q.stride(0),q.stride(1),k.stride(0),k.stride(1),v.stride(0),v.stride(1),
        32,True,q_norm.eps,k_norm.eps,normalized_dtype==torch.bfloat16,welford,num_warps=4,enable_fp_fusion=False)
    return qo,ko,vo
