"""One shared, contiguous union K/V bank per window; identical selected masks.

All union sizes/offsets remain on GPU. No torch.unique/nonzero or scalar read.
Unselected union entries have -inf bias and never enter attention or history.
"""
import torch,triton
import triton.language as tl
from .reuse_single_qk_grid import _reduce

@triton.jit
def _union(IDS,FLAG,A:tl.constexpr,G:tl.constexpr,B:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B);key=tl.load(IDS+j,j<G*A,-1)
    tl.atomic_max(FLAG+tl.maximum(key,0),1,(j<G*A)&(key>=0),sem='relaxed')

@triton.jit
def _map(ORDER,FLAG,PREFIX,GRID,POS,COUNT,N:tl.constexpr,B:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B);key=tl.load(ORDER+j,j<N,0);keep=tl.load(FLAG+key,j<N,0)>0
    pos=tl.load(PREFIX+j,j<N,0)-1;region=tl.load(GRID+key,j<N,0)
    tl.store(POS+key,pos,(j<N)&keep)
    tl.atomic_add(COUNT+region,1,(j<N)&keep,sem='relaxed')

@triton.jit
def _gather(K,V,ORDER,FLAG,PREFIX,KB,VB,N:tl.constexpr,D:tl.constexpr,KH:tl.constexpr,B:tl.constexpr,BD:tl.constexpr):
    h=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);d=tl.arange(0,BD)
    key=tl.load(ORDER+j,j<N,0);keep=tl.load(FLAG+key,j<N,0)>0;pos=tl.load(PREFIX+j,j<N,0)-1
    kval=tl.load(K+h*KH+key[:,None]*D+d[None,:],((j<N)&keep)[:,None]&(d[None,:]<D),0)
    vval=tl.load(V+h*KH+key[:,None]*D+d[None,:],((j<N)&keep)[:,None]&(d[None,:]<D),0)
    tl.store(KB+(h*N+pos[:,None])*D+d[None,:],kval,((j<N)&keep)[:,None]&(d[None,:]<D))
    tl.store(VB+(h*N+pos[:,None])*D+d[None,:],vval,((j<N)&keep)[:,None]&(d[None,:]<D))

@triton.jit
def _bias(IDS,WEIGHT,POS,BIAS,LOGICAL,N:tl.constexpr,A:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B);key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0)
    pos=tl.load(POS+tl.maximum(key,0),valid,0);bias=tl.load(WEIGHT+g*A+j,valid,0.)
    tl.store(BIAS+g*N+pos,bias,valid);tl.store(LOGICAL+g*N+pos,j,valid)

@triton.jit
def _attention(Q,K,V,KB,VB,BIAS,LOGICAL,OFF,SPECIAL,PART,LSE,PROBE,LOGGRID,RAWGRID,
               G:tl.constexpr,H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,REGIONS:tl.constexpr,PARTS:tl.constexpr,PER:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,EVIDENCE:tl.constexpr):
    qb=tl.program_id(0);h=tl.program_id(1);part=tl.program_id(2)
    row=tl.arange(0,64);rr=qb*64+row;group=rr//32;dd=tl.arange(0,BD);cc=tl.arange(0,BK)
    q=tl.load(Q+h*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<R)&(dd[None,:]<D),0)
    maximum=tl.full((64,),-float('inf'),tl.float32);den=tl.zeros((64,),tl.float32);acc=tl.zeros((64,BD),tl.float32)
    for bank in tl.static_range(2):
        for region in range(0 if bank==0 else part*PER,(1 if part==0 else 0) if bank==0 else tl.minimum((part+1)*PER,REGIONS)):
            begin=0;end=S
            if bank==1:begin=tl.load(OFF+region);end=tl.load(OFF+region+1)
            region_den=tl.zeros((64,),tl.float32);unit_den=tl.zeros((64,),tl.float32)
            rawmax=tl.full((8,),-float('inf'),tl.float32);rawden=tl.zeros((8,),tl.float32)
            for tile in range(tl.cdiv(end-begin,BK)):
                col=begin+tile*BK+cc
                if bank==0:
                    ids=tl.load(SPECIAL+col,col<end,0)
                    kval=tl.load(K+h*KH+ids[None,:]*D+dd[:,None],(col[None,:]<end)&(dd[:,None]<D),0)
                    vval=tl.load(V+h*KH+ids[:,None]*D+dd[None,:],(col[:,None]<end)&(dd[None,:]<D),0)
                    bias=tl.full((64,BK),0.,tl.float32);active=tl.broadcast_to((col<end)[None,:],(64,BK))
                else:
                    kval=tl.load(KB+(h*N+col[None,:])*D+dd[:,None],(col[None,:]<end)&(dd[:,None]<D),0)
                    vval=tl.load(VB+(h*N+col[:,None])*D+dd[None,:],(col[:,None]<end)&(dd[None,:]<D),0)
                    bias=tl.load(BIAS+group[:,None]*N+col[None,:],(rr[:,None]<R)&(col[None,:]<end),-float('inf'))
                    active=(bias!=-float('inf'))&(col[None,:]<end)
                qk=tl.dot(q,kval,allow_tf32=False)*SCALE
                z=tl.where(active,qk+bias,-float('inf'))
                if EVIDENCE and bank==1:
                    pp=tl.arange(0,8);pg=qb*2+pp//4;pr=(pp//4)*32+tl.minimum((pp%4)*8+4,R-qb*64-(pp//4)*32-1)
                    raw=tl.where(active,qk,-float('inf'));value=tl.gather(raw,tl.broadcast_to(pr[:,None],(8,BK)),0)
                    mx=tl.maximum(rawmax,tl.max(value,1));safe=tl.where(mx==-float('inf'),0.,mx)
                    rawden=rawden*tl.exp(rawmax-safe)+tl.sum(tl.exp(value-safe[:,None]),1);rawmax=mx
                    logical=tl.load(LOGICAL+pg[:,None]*N+col[None,:],(pg[:,None]<G)&(col[None,:]<end),-1)
                    tl.store(PROBE+((pg[:,None]*H+h)*4+(pp%4)[:,None])*A+logical,value,(pg[:,None]<G)&(col[None,:]<end)&(logical>=0))
                mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx);correction=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
                acc=acc*correction[:,None]+tl.dot(p.to(vval.dtype),vval,allow_tf32=False)
                den=den*correction+tl.sum(p,1);region_den=region_den*correction+tl.sum(p,1)
                unit_den=unit_den*correction+tl.sum(p*tl.where(active,tl.exp(-bias),0.),1);maximum=mx
            if bank==1:
                base=(((group*H+h)*2)*REGIONS+region)*32+row%32
                tl.store(LOGGRID+base,maximum+tl.log(region_den),group<G)
                tl.store(LOGGRID+base+REGIONS*32,maximum+tl.log(unit_den),group<G)
                if EVIDENCE:
                    pp=tl.arange(0,8);pg=qb*2+pp//4
                    tl.store(RAWGRID+((pg*H+h)*4+pp%4)*REGIONS+region,rawmax+tl.log(rawden),pg<G)
    tl.store(PART+(((qb*H+h)*PARTS+part)*64+row[:,None])*BD+dd[None,:],tl.where(den[:,None]>0,acc/tl.maximum(den[:,None],1.e-30),0.),dd[None,:]<D)
    tl.store(LSE+((qb*H+h)*PARTS+part)*64+row,maximum+tl.log(den))

@triton.jit
def _merge(PART,LP,O,LSE,H:tl.constexpr,R:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,S:tl.constexpr,BS:tl.constexpr):
    qb=tl.program_id(0);h=tl.program_id(1);row=tl.program_id(2);ss=tl.arange(0,BS);dd=tl.arange(0,BD)
    z=tl.load(LP+((qb*H+h)*S+ss)*64+row,ss<S,-float('inf'));maximum=tl.max(z,0);w=tl.exp(z-maximum);den=tl.sum(w,0)
    value=tl.load(PART+(((qb*H+h)*S+ss[:,None])*64+row)*BD+dd[None,:],(ss[:,None]<S)&(dd[None,:]<D),0.)
    rr=qb*64+row;tl.store(O+h*R*D+rr*D+dd,tl.sum(value*w[:,None],0)/den,(rr<R)&(dd<D));tl.store(LSE+h*R+rr,maximum+tl.log(den),rr<R)

def weighted_attention(q,k,v,indices,log_weights,specials,scale,key_grid,regions,emit_evidence=True,evidence_layout=None,parts=8,bank_order=None,token_grid=None,block_keys=128):
    if q.dtype==torch.float32:block_keys=min(block_keys,64)
    h,r,d=q.shape;g,a=indices.shape;n=k.shape[1];device=q.device
    if token_grid is None:
        token_grid=torch.full((n+1,),regions,device=device,dtype=torch.int32)
        token_grid.scatter_(0,indices.long().masked_fill(indices<0,n).flatten(),key_grid.int().flatten())
        token_grid=token_grid[:n]
    if bank_order is None:bank_order=token_grid.argsort(stable=True).int()
    flags=torch.zeros(n,device=device,dtype=torch.int32);_union[(triton.cdiv(g*a,256),)](indices,flags,a,g,256,num_warps=4)
    prefix=flags[bank_order.long()].cumsum(0,dtype=torch.int32);pos=torch.empty(n,device=device,dtype=torch.int32);bank_counts=torch.zeros(regions,device=device,dtype=torch.int32)
    _map[(triton.cdiv(n,256),)](bank_order,flags,prefix,token_grid,pos,bank_counts,n,256,num_warps=4)
    off=torch.nn.functional.pad(bank_counts.cumsum(0,dtype=torch.int32),(1,0)).contiguous()
    kb=torch.empty_like(k);vb=torch.empty_like(v)
    _gather[(h,triton.cdiv(n,32))](k,v,bank_order,flags,prefix,kb,vb,n,d,k.stride(0),32,triton.next_power_of_2(d),num_warps=4)
    bias=torch.full((g,n),-torch.inf,device=device);logical=torch.full((g,n),-1,device=device,dtype=torch.int32)
    _bias[(g,triton.cdiv(a,256))](indices,log_weights,pos,bias,logical,n,a,256,num_warps=4)
    counts=torch.zeros((g,regions+1),device=device,dtype=torch.int32).scatter_add_(1,key_grid.masked_fill(indices<0,regions).long(),torch.ones_like(indices,dtype=torch.int32))
    offsets=torch.nn.functional.pad(counts[:,:regions].cumsum(1,dtype=torch.int32),(1,0)).contiguous()
    bd=max(16,triton.next_power_of_2(d));qb=triton.cdiv(r,64)
    output=torch.empty_like(q);lse=torch.empty((h,r),device=device);partial=torch.empty((qb,h,parts,64,bd),device=device);plse=torch.empty((qb,h,parts,64),device=device)
    probe=torch.full((g,h,4,a),-torch.inf,device=device) if emit_evidence else lse
    logs=torch.empty((g,h,2,regions,32),device=device);rawgrid=torch.empty((g,h,4,regions),device=device) if emit_evidence else lse;summary=torch.empty((g,2,regions),device=device)
    if emit_evidence and evidence_layout is not None:evidence_layout.update(selected_grid_order=None,selected_grid_offsets=offsets,grid_raw_logsum=rawgrid)
    _attention[(qb,h,parts)](q,k,v,kb,vb,bias,logical,off,specials,partial,plse,probe,logs,rawgrid,g,h,r,n,a,len(specials),d,regions,parts,triton.cdiv(regions,parts),q.stride(0),k.stride(0),scale,bd,block_keys,emit_evidence,num_warps=8,num_stages=1)
    _merge[(qb,h,64)](partial,plse,output,lse,h,r,d,bd,parts,triton.next_power_of_2(parts),num_warps=4)
    _reduce[(g,2,triton.cdiv(regions,8))](logs,lse,summary,h,r,regions,8,triton.next_power_of_2(h),num_warps=4)
    return output,summary,probe if emit_evidence else None
