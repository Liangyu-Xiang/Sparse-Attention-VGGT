"""Forced PyTorch FlashAttention AV with unchanged sparse support/bias and v8 evidence.

Padding duplicates Special0 with split probability mass, avoiding nonfinite
Flash inputs. Three low-precision residual channels encode each FP32 Key bias; this preserves
original Q/K coordinates and scale, with separately measured numerical error.
"""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
import triton
import triton.language as tl

@triton.jit
def _statistics(Q,K,IDS,BIAS,ORDER,OFFSETS,SPECIAL,LSE,PROBE,LOGGRID,
               H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
               REGIONS:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr,EVIDENCE:tl.constexpr):
    group=tl.program_id(0);head=tl.program_id(1)
    row=tl.arange(0,32);rr=group*32+row;dd=tl.arange(0,BD);cc=tl.arange(0,64)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<R)&(dd[None,:]<D),0)
    maximum=tl.full((32,),-float('inf'),tl.float32);den=tl.zeros((32,),tl.float32)
    for bank in tl.static_range(2):
        for region in range(1 if bank==0 else REGIONS):
            if bank==0:
                begin=0;end=S
            else:
                begin=tl.load(OFFSETS+group*(REGIONS+1)+region)
                end=tl.load(OFFSETS+group*(REGIONS+1)+region+1)
            region_den=tl.zeros((32,),tl.float32);unit_den=tl.zeros((32,),tl.float32)
            for tile in range(tl.cdiv(end-begin,64)):
                col=begin+tile*64+cc
                if bank==0:
                    ids=tl.load(SPECIAL+col,col<end,0);valid=col<end;bias=tl.full((64,),0.,tl.float32)
                else:
                    ids=tl.load(IDS+group*A+col,col<end,-1);valid=(col<end)&(ids>=0)
                    bias=tl.load(BIAS+group*A+col,col<end,-float('inf'))
                k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
                qk=tl.dot(q,k,allow_tf32=False)
                z=tl.where(valid[None,:],qk*SCALE+bias[None,:],-float('inf'))
                if EVIDENCE and bank==1:
                    probe=tl.arange(0,4);pr=tl.minimum(probe*8+4,R-group*32-1)
                    raw=tl.where(valid[None,:],qk*SCALE,-float('inf'))
                    values=tl.gather(raw,tl.broadcast_to(pr[:,None],(4,64)),0)
                    logical=tl.load(ORDER+group*A+col,col<end,0)
                    tl.store(PROBE+((group*H+head)*4+probe[:,None])*A+logical[None,:],values,col[None,:]<end)
                mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
                correction=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
                den=den*correction+tl.sum(p,1)
                region_den=region_den*correction+tl.sum(p,1)
                unit_den=unit_den*correction+tl.sum(p*tl.where(valid,tl.exp(-bias),0.)[None,:],1)
                maximum=mx
            if bank==1:
                base=((group*H+head)*2)*REGIONS*32+region*32+row
                tl.store(LOGGRID+base,maximum+tl.log(region_den))
                tl.store(LOGGRID+base+REGIONS*32,maximum+tl.log(unit_den))
    tl.store(LSE+head*R+rr,maximum+tl.log(den),rr<R)

@triton.jit
def _reduce(LOGGRID,LSE,OUT,H:tl.constexpr,R:tl.constexpr,REGIONS:tl.constexpr,B:tl.constexpr,BH:tl.constexpr):
    group=tl.program_id(0);kind=tl.program_id(1);region=tl.program_id(2)*B+tl.arange(0,B)
    hr=tl.arange(0,BH*32);head=hr//32;row=hr%32;rr=group*32+row
    value=tl.load(LOGGRID+(((group*H+head[:,None])*2+kind)*REGIONS+region[None,:])*32+row[:,None],(head[:,None]<H)&(region[None,:]<REGIONS),-float('inf'))
    lse=tl.load(LSE+head*R+rr,(head<H)&(rr<R),0.)
    probability=tl.where((head[:,None]<H)&(rr[:,None]<R),tl.exp(value-lse[:,None]),0.)
    result=tl.sum(probability,0)/(H*tl.minimum(32,R-group*32))
    tl.store(OUT+(group*2+kind)*REGIONS+region,result,region<REGIONS)


@triton.jit
def _pack_kv(K,V,IDS,BIAS,SPECIAL,PADCOUNT,PK,PV,H:tl.constexpr,N:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,PAD:tl.constexpr,FIRST:tl.constexpr,SCALE:tl.constexpr,B:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0)*B+tl.arange(0,B);head=tl.program_id(1);local=tl.program_id(2);g=FIRST+local;d=tl.arange(0,BD)
    key=tl.load(SPECIAL+row,(row<S)&(row<A+S),0)
    patch=tl.load(IDS+g*A+row-S,(row>=S)&(row<A+S),-1);key=tl.where(row<S,key,patch)
    padded=(row<A+S)&(row>=S)&(key<0)
    special0=tl.load(SPECIAL)
    key=tl.where(padded,special0,key)
    valid=(row<A+S)&(key>=0)
    bias=tl.load(BIAS+g*A+row-S,(row>=S)&valid&~padded,0.)
    copies=tl.load(PADCOUNT+g).to(tl.float32)+1.
    bias=tl.where((row==0)|padded,-tl.log(copies),bias)
    residual=tl.where(valid,bias/SCALE,0.)
    p1=residual.to(PK.dtype.element_ty);residual=residual-p1.to(tl.float32)
    p2=residual.to(PK.dtype.element_ty);residual=residual-p2.to(tl.float32)
    p3=residual.to(PK.dtype.element_ty)
    kval=tl.load(K+(head*N+tl.maximum(key,0))[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0.)
    vval=tl.load(V+(head*N+tl.maximum(key,0))[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0.)
    kval=tl.where(d[None,:]==D,p1[:,None],kval)
    kval=tl.where(d[None,:]==D+1,p2[:,None],kval);kval=tl.where(d[None,:]==D+2,p3[:,None],kval)
    offset=((local*H+head)*(A+S)+row[:,None])*PAD+d[None,:]
    tl.store(PK+offset,kval,(row[:,None]<A+S)&(d[None,:]<PAD));tl.store(PV+offset,vval,(row[:,None]<A+S)&(d[None,:]<PAD))

@triton.jit
def _pack_q(Q,PQ,H:tl.constexpr,R:tl.constexpr,D:tl.constexpr,PAD:tl.constexpr,FIRST:tl.constexpr,QH:tl.constexpr,BD:tl.constexpr):
    local=tl.program_id(0);head=tl.program_id(1);row=tl.arange(0,32);rr=(FIRST+local)*32+row;d=tl.arange(0,BD)
    q=tl.load(Q+head*QH+rr[:,None]*D+d[None,:],(rr[:,None]<R)&(d[None,:]<D),0.)
    q=tl.where((d[None,:]>=D)&(d[None,:]<D+3),1.,q)
    tl.store(PQ+((local*H+head)*32+row[:,None])*PAD+d[None,:],q,d[None,:]<PAD)

@triton.jit
def _unpack(O,OUT,H:tl.constexpr,R:tl.constexpr,D:tl.constexpr,PAD:tl.constexpr,FIRST:tl.constexpr,BD:tl.constexpr):
    local=tl.program_id(0);head=tl.program_id(1);row=tl.arange(0,32);rr=(FIRST+local)*32+row;d=tl.arange(0,BD)
    x=tl.load(O+((local*H+head)*32+row[:,None])*PAD+d[None,:],d[None,:]<D,0.)
    tl.store(OUT+(head*R+rr[:,None])*D+d[None,:],x,(rr[:,None]<R)&(d[None,:]<D))

def weighted_attention(q,k,v,indices,log_weights,specials,scale,key_grid,regions,emit_evidence=True,evidence_layout=None,group_chunk=16,scope=None):
    if q.dtype not in (torch.float16,torch.bfloat16):raise ValueError('Forced FlashAttention requires FP16/BF16 Q/K/V')
    if group_chunk<1:raise ValueError('group_chunk must be positive')
    if len(specials)<1:raise ValueError('At least one Special is required for exact duplicate-padding normalization')
    h,r,d=q.shape;g,a=indices.shape
    if g!=triton.cdiv(r,32):raise ValueError('Invalid query group count')
    if not (q.stride(1)==d and q.stride(2)==1 and k.is_contiguous() and v.is_contiguous()):raise ValueError('Contiguous query rows and K/V required; strided head windows are supported')
    scope=scope or (lambda name:nullcontext())
    with scope('flash_support_sort'):
        bank=key_grid.masked_fill(indices<0,regions);order=bank.argsort(dim=1,stable=True)
        sorted_ids=indices.gather(1,order).contiguous();sorted_bias=log_weights.gather(1,order).contiguous();pad_count=(indices<0).sum(1,dtype=torch.int32).contiguous()
        counts=torch.zeros((g,regions+1),device=q.device,dtype=torch.int32);counts.scatter_add_(1,bank.long(),torch.ones_like(indices,dtype=torch.int32))
        offsets=F.pad(counts[:,:regions].cumsum(1,dtype=torch.int32),(1,0)).contiguous();order=order.int().contiguous()
        if emit_evidence and evidence_layout is not None:evidence_layout.update(selected_grid_order=order,selected_grid_offsets=offsets)
    output=torch.empty(q.shape,device=q.device,dtype=q.dtype);pad=triton.cdiv(d+3,8)*8
    for first in range(0,g,group_chunk):
        amount=min(group_chunk,g-first)
        with scope('flash_pack_qkv_bias'):
            pk=torch.empty((amount,h,a+len(specials),pad),device=q.device,dtype=q.dtype);pv=torch.empty_like(pk);pq=torch.empty((amount,h,32,pad),device=q.device,dtype=q.dtype)
            _pack_kv[(triton.cdiv(a+len(specials),16),h,amount)](k,v,sorted_ids,sorted_bias,specials,pad_count,pk,pv,h,k.shape[1],a,len(specials),d,pad,first,scale,16,triton.next_power_of_2(pad),num_warps=4,enable_fp_fusion=False)
            _pack_q[(amount,h)](q,pq,h,r,d,pad,first,q.stride(0),triton.next_power_of_2(pad),num_warps=4)
        with scope('flash_native_av'):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):result=F.scaled_dot_product_attention(pq,pk,pv,dropout_p=0.,is_causal=False,scale=scale)
        with scope('flash_unpack_output'):_unpack[(amount,h)](result,output,h,r,d,pad,first,triton.next_power_of_2(d),num_warps=4)
        del pq,pk,pv,result
    with scope('flash_v8_evidence_qk'):
        lse=torch.empty((h,r),device=q.device,dtype=torch.float32);probes=torch.full((g,h,4,a),-torch.inf,device=q.device) if emit_evidence else lse
        logs=torch.empty((g,h,2,regions,32),device=q.device,dtype=torch.float32);summary=torch.empty((g,2,regions),device=q.device,dtype=torch.float32)
        _statistics[(g,h)](q,k,sorted_ids,sorted_bias,order,offsets,specials,lse,probes,logs,h,r,a,len(specials),d,regions,q.stride(0),k.stride(0),scale,max(16,triton.next_power_of_2(d)),emit_evidence,num_warps=4,num_stages=1)
        _reduce[(g,2,triton.cdiv(regions,8))](logs,lse,summary,h,r,regions,8,triton.next_power_of_2(h),num_warps=4)
    return output,summary,probes if emit_evidence else None
