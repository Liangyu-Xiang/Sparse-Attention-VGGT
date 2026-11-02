"""Calibrate emitted logits and consume cached per-head/probe conditionals."""
import torch,triton
import triton.language as tl


@triton.jit
def _row_lse(X,Y,A:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.arange(0,B)
    x=tl.load(X+row*A+i,i<A,-float('inf'))
    maximum=tl.max(x,0);safe=tl.where(maximum==-float('inf'),0.,maximum)
    total=tl.sum(tl.exp(x-safe),0)
    tl.store(Y+row,tl.where(total>0,safe+tl.log(total),-float('inf')))


def logsumexp(raw):
    out=torch.empty(raw.shape[:-1],device=raw.device,dtype=torch.float32)
    b=triton.next_power_of_2(raw.shape[-1])
    _row_lse[(out.numel(),)](raw,out,raw.shape[-1],b,num_warps=16 if b>4096 else 4)
    return out


@triton.jit
def _segment(a,fa,b,fb):
    return tl.where(fb,b,a+b),fa|fb


@triton.jit
def _calibrate(RAW,LOGZ,IDS,GRID,OUT,A:tl.constexpr,H:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);hp=tl.program_id(1);g=tl.program_id(2)
    index=col
    key=tl.load(IDS+g*A+index,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    segment=tl.where(valid,region,-1)
    prev=col-1
    prevkey=tl.load(IDS+g*A+prev,(col>0)&(col<A),-1)
    prevregion=tl.load(GRID+tl.maximum(prevkey,0),(col>0)&(col<A)&(prevkey>=0),0)
    prevsegment=tl.where((col>0)&(prevkey>=0),prevregion,-1)
    flag=(col==0)|(segment!=prevsegment)
    raw=tl.load(RAW+(g*H*4+hp)*A+index,valid,-float('inf'))
    logz=tl.load(LOGZ+g*H*4+hp)
    value=tl.exp(raw-logz)
    total,_=tl.associative_scan((value,flag),0,_segment)
    nxt=col+1
    nextkey=tl.load(IDS+g*A+nxt,col+1<A,-1)
    nextregion=tl.load(GRID+tl.maximum(nextkey,0),(col+1<A)&(nextkey>=0),0)
    nextsegment=tl.where((col+1<A)&(nextkey>=0),nextregion,-1)
    last=((col%B)==B-1)|(col==A-1)|(segment!=nextsegment)
    tl.atomic_add(OUT+(g*H*4+hp)*R+region,total,valid&last,sem='relaxed')


@triton.jit
def _pack_bits(encoded,CACHE,row,tile,A:tl.constexpr,B:tl.constexpr):
    byte=tl.arange(0,triton.next_power_of_2(B*14//8))
    bit=byte*8;index=bit//14;shift=bit%14
    lo=tl.gather(encoded,tl.minimum(index,B-1),0).to(tl.uint32)
    hi=tl.where(index+1<B,tl.gather(encoded,tl.minimum(index+1,B-1),0),0).to(tl.uint32)
    value=(lo>>shift)|(hi<<(14-shift))
    offset=tile*(B*14//8)+byte;stride=(A*14+7)//8
    tl.store(CACHE+row*stride+offset,value.to(tl.uint8),(byte<B*14//8)&(offset<stride))


@triton.jit
def _pack_half(HALF,CACHE,A:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);tile=tl.program_id(1);col=tile*B+tl.arange(0,B)
    value=tl.load(HALF+row*A+col,col<A,0.)
    _pack_bits(value.to(tl.uint16,bitcast=True),CACHE,row,tile,A,B)


def pack_half(half):
    a=half.shape[-1];out=torch.empty((*half.shape[:-1],(a*14+7)//8),device=half.device,dtype=torch.uint8)
    _pack_half[(half.numel()//a,triton.cdiv(a,256))](half,out,a,256,num_warps=4)
    return out


@triton.jit
def _store(RAW,LOGZ,IDS,GRID,CORRECTED,CACHE,A:tl.constexpr,H:tl.constexpr,R:tl.constexpr,B:tl.constexpr,EPS:tl.constexpr,PACKED:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);hp=tl.program_id(1);g=tl.program_id(2)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    raw=tl.load(RAW+(g*H*4+hp)*A+col,valid,-float('inf'))
    logz=tl.load(LOGZ+g*H*4+hp)
    corrected=tl.load(CORRECTED+(g*H*4+hp)*R+region,valid,1.)
    conditional=tl.exp(raw-logz)/tl.maximum(corrected,EPS)
    conditional=tl.where(valid,conditional,0.)
    if PACKED:
        encoded=conditional.to(tl.float16).to(tl.uint16,bitcast=True)
        _pack_bits(encoded,CACHE,g*H*4+hp,tl.program_id(0),A,B)
    else:
        tl.store(CACHE+(g*H*4+hp)*A+col,conditional,col<A)


def calibrate(raw,logz,ids,grid,base,regions):
    g,h,_,a=raw.shape
    corrected=base.contiguous()
    _calibrate[(triton.cdiv(a,256),h*4,g)](raw,logz,ids,grid,corrected,a,h,base.shape[-1],256,num_warps=4)
    return corrected


def store(raw,logz,ids,grid,corrected,cache,eps):
    g,h,_,a=raw.shape
    _store[(triton.cdiv(a,256),h*4,g)](raw,logz,ids,grid,corrected,cache,a,h,corrected.shape[-1],256,eps,cache.dtype==torch.uint8,num_warps=4,enable_fp_fusion=False)


@triton.jit
def _load_cached(CACHE,row,col,mask,A:tl.constexpr,PACKED:tl.constexpr):
    if PACKED:
        stride=(A*14+7)//8;bit=col*14;byte=bit//8;shift=bit%8
        lo=tl.load(CACHE+row*stride+byte,mask,0).to(tl.uint32)
        mid=tl.load(CACHE+row*stride+byte+1,mask&(byte+1<stride),0).to(tl.uint32)
        hi=tl.load(CACHE+row*stride+byte+2,mask&(byte+2<stride),0).to(tl.uint32)
        word=((lo|(mid<<8)|(hi<<16))>>shift)&0x3fff
        return word.to(tl.uint16).to(tl.float16,bitcast=True).to(tl.float32)
    else:
        return tl.load(CACHE+row*A+col,mask,0.).to(tl.float32)


@triton.jit
def _decode(CACHE,OUT,A:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    value=_load_cached(CACHE,row,col,col<A,A,True)
    tl.store(OUT+row*A+col,value,col<A)


def decode(cache,a):
    if cache.dtype!=torch.uint8:return cache.float()
    out=torch.empty((*cache.shape[:-1],a),device=cache.device,dtype=torch.float32)
    _decode[(out.numel()//a,triton.cdiv(a,256))](cache,out,a,256,num_warps=4)
    return out


@triton.jit
def _predict(CACHE,IDS,GRID,SIZE,CUR,PREV,WEIGHTS,OUT,A:tl.constexpr,H:tl.constexpr,R:tl.constexpr,B:tl.constexpr,EPS:tl.constexpr,PACKED:tl.constexpr):
    col=tl.program_id(0)*B+tl.arange(0,B);g=tl.program_id(1);h=tl.program_id(2);p=tl.arange(0,4)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    region=tl.load(GRID+tl.maximum(key,0),valid,0);size=tl.load(SIZE+region,valid,1.)
    ix=((g*H+h)*4+p[:,None])*R+region[None,:]
    current=tl.load(CUR+ix,valid[None,:],0.);previous=tl.load(PREV+ix,valid[None,:],0.)
    ratio=(current+EPS)/(previous+EPS)
    confidence=tl.minimum(1.,tl.maximum(0.,tl.where(ratio<1.,ratio,1./(ratio*ratio))))
    condition=_load_cached(CACHE,((g*H+h)*4+p[:,None]),col[None,:],valid[None,:],A,PACKED)
    predicted=current*(confidence*condition+(1.-confidence)/size[None,:])
    weight=tl.load(WEIGHTS+g*4+p)
    value=tl.sum(tl.where(valid[None,:],predicted*weight[:,None],0.),0)
    tl.store(OUT+(g*H+h)*A+col,value,col<A)


from .reuse_m5c_equivalent import _write


def predict(cache,ids,grid,sizes,current,previous,weights,prior,eps):
    g,h,_,_=cache.shape;a=ids.shape[1]
    per_head=torch.empty((g,h,a),device=prior.device,dtype=torch.float32)
    _predict[(triton.cdiv(a,256),g,h)](cache,ids,grid,sizes,current.contiguous(),previous,
        weights.contiguous(),per_head,a,h,len(sizes),256,eps,cache.dtype==torch.uint8,num_warps=4,enable_fp_fusion=False)
    edge=per_head.mean(1)
    _write[(g,triton.cdiv(a,256))](edge,ids,prior,a,len(grid),256,num_warps=4)
    return edge
