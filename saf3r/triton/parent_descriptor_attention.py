"""Direct common-bank + remaining-history + whole-Parent online attention.

No expanded per-group Parent Token index tensor, no group×Token seen bitmap.
Only Parent descriptors, compact history, and shared member_bank are consumed.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _member(IDS,ENDS,STARTS,MEM,row,col,valid,P:tl.constexpr,SEARCH:tl.constexpr):
    lo=tl.full(col.shape,0,tl.int32);hi=tl.full(col.shape,P,tl.int32)
    for _ in range(SEARCH):
        mid=(lo+hi)//2
        end=tl.load(ENDS+row*P+mid,mid<P,2147483647)
        move=end<=col
        lo=tl.where(move,mid+1,lo);hi=tl.where(move,hi,mid)
    slot=tl.minimum(lo,P-1)
    parent=tl.load(IDS+row*P+slot,valid,0)
    previous=tl.load(ENDS+row*P+slot-1,valid&(slot>0),0)
    start=tl.load(STARTS+tl.maximum(parent,0),valid,0)
    return tl.load(MEM+start+col-previous,valid,-1)


@triton.jit
def _filter(H,SELECTED,SOURCE,O,COUNT,HC:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,B);cursor=tl.full((),0,tl.int32)
    for block in range(tl.cdiv(HC,B)):
        j=block*B+col;key=tl.load(H+row*HC+j,j<HC,-1)
        parent=tl.load(SOURCE+tl.maximum(key,0),(j<HC)&(key>=0),0)
        covered=tl.load(SELECTED+row*(P+1)+parent,(j<HC)&(key>=0),0)
        valid=(j<HC)&(key>=0)&(covered==0)
        within=tl.cumsum(valid.to(tl.int32),0)-1
        tl.store(O+row*HC+cursor+within,key,valid)
        cursor+=tl.sum(valid.to(tl.int32),0)
    tl.store(COUNT+row,cursor)


def remaining_history(history,parent_ids,patch_parent):
    groups,p=parent_ids.shape;hc=history.shape[1]
    selected=torch.zeros((groups,p+1),device=history.device,dtype=torch.uint8)
    selected.scatter_(1,torch.where(parent_ids>=0,parent_ids.long(),p),1)
    out=torch.full_like(history,-1,dtype=torch.int32)
    count=torch.zeros(groups,device=history.device,dtype=torch.int32)
    if hc:_filter[(groups,)](history,selected,patch_parent,out,count,hc,p,1024,num_warps=4)
    return out,count


@triton.jit
def _attention(Q,K,V,PK,PV,HIST,HC,IDS,ENDS,STARTS,SIZES,MEM,PATCH,ISVALUE,O,E,LSE,TC,
        N:tl.constexpr,R:tl.constexpr,D:tl.constexpr,PC:tl.constexpr,COMMON:tl.constexpr,VC:tl.constexpr,
        HCAP:tl.constexpr,PLIMIT:tl.constexpr,PTILES:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,PH:tl.constexpr,
        QSTART,QOFFSET,SCALE:tl.constexpr,BD:tl.constexpr,SLOTS:tl.constexpr,EVIDENCE:tl.constexpr,BPC:tl.constexpr):
    group,head=tl.program_id(0),tl.program_id(1)
    rr=group*32+tl.arange(0,32);dd=tl.arange(0,BD);cc=tl.arange(0,64)
    qr=(QSTART+rr+QOFFSET)%N
    qq=tl.load(Q+head*QH+qr[:,None]*D+dd[None,:],(rr[:,None]<R)&(dd[None,:]<D),0)
    maximum=tl.full((32,),-float('inf'),tl.float32);den=tl.zeros((32,),tl.float32)
    acc=tl.zeros((32,BD),tl.float32)
    hn=tl.load(HC+group)
    pp=tl.arange(0,BPC)
    pn=tl.sum((tl.load(IDS+group*PC+pp,pp<PC,-1)>=0).to(tl.int32),0)
    for bank in tl.static_range(3):
        if bank==0:
            size=COMMON;capacity=tl.cdiv(COMMON,64);base=0
        elif bank==1:
            size=hn;capacity=tl.cdiv(HCAP,64);base=tl.cdiv(COMMON,64)
        else:
            size=pn*PTILES;capacity=PLIMIT*PTILES;base=tl.cdiv(COMMON,64)+tl.cdiv(HCAP,64)
        for slot in range(size if bank==2 else tl.cdiv(size,64)):
            col=slot*64+cc
            if bank==0:
                valid=col<COMMON;patch=valid&(col<VC)
                kk=tl.load(PK+head*PH+col[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
                vv=tl.load(PV+head*PH+col[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
            else:
                if bank==1:local=tl.load(HIST+group*HCAP+col,col<hn,-1)
                else:
                    rank=slot//PTILES;tile=slot%PTILES
                    parent=tl.load(IDS+group*PC+rank)
                    start=tl.load(STARTS+parent);length=tl.load(SIZES+parent)
                    col=tile*64+cc
                    local=tl.load(MEM+start+col,col<length,-1)
                valid=(local>=0)
                if bank==1:valid=valid&(col<size)
                if bank==2:
                    duplicate=tl.load(ISVALUE+tl.maximum(local,0),valid,0)
                    valid=valid&(duplicate==0)
                global_id=tl.load(PATCH+tl.maximum(local,0),valid,0)
                kk=tl.load(K+head*KH+global_id[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
                vv=tl.load(V+head*KH+global_id[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
                patch=valid
            z=tl.where(valid[None,:],tl.dot(qq,kk)*SCALE,-float('inf'))
            new=tl.maximum(maximum,tl.max(z,1));safe=tl.where(new==-float('inf'),0.,new)
            correction=tl.exp(maximum-safe);w=tl.exp(z-safe[:,None])
            acc=acc*correction[:,None]+tl.dot(w.to(vv.dtype),vv)
            den=den*correction+tl.sum(w,1);maximum=new
            if EVIDENCE:
                mass=tl.sum(tl.where(patch[None,:],w,0.),1)
                logmass=tl.where(mass>0,tl.log(mass)+safe,-float('inf'))
                tl.store(E+(head*R+rr)*SLOTS+base+slot,logmass,rr<R)
                if head==0:tl.store(TC+group*SLOTS+base+slot,tl.sum(patch.to(tl.int32),0))
        if EVIDENCE:
            for slot in range(size if bank==2 else tl.cdiv(size,64),capacity):
                tl.store(E+(head*R+rr)*SLOTS+base+slot,-float('inf'),rr<R)
                if head==0:tl.store(TC+group*SLOTS+base+slot,0)
    tl.store(O+(head*R+rr[:,None])*D+dd[None,:],acc/den[:,None],(rr[:,None]<R)&(dd[None,:]<D))
    if EVIDENCE:tl.store(LSE+head*R+rr,maximum+tl.log(den),rr<R)


def descriptor_attention(q,k,v,pk,pv,history,history_counts,parent_ids,parent_ends,parent_starts,
        member_bank,patches,is_value,value_count,raw_capacity,*,query_start,query_rows,query_offset=0,scale=None,evidence=False,
        parent_sizes=None,parent_stride=None,parent_limit=None):
    h,n,d=q.shape;groups=len(parent_ids);pc=parent_ids.shape[1];hc=history.shape[1];common=pk.shape[1]
    if parent_sizes is None:raise ValueError('Parent sizes required for direct Parent-tile execution')
    pt=triton.cdiv(parent_stride,64);limit=pc if parent_limit is None else parent_limit
    slots=triton.cdiv(common,64)+triton.cdiv(hc,64)+limit*pt
    out=torch.empty((h,query_rows,d),device=q.device,dtype=q.dtype)
    e=torch.empty((h,query_rows,slots) if evidence else (0,),device=q.device,dtype=torch.float32)
    lse=torch.empty((h,query_rows) if evidence else (0,),device=q.device,dtype=torch.float32)
    counts=torch.empty((groups,slots) if evidence else (0,),device=q.device,dtype=torch.int32)
    _attention[(groups,h)](q,k,v,pk,pv,history,history_counts,parent_ids,parent_ends,parent_starts,parent_sizes,member_bank,
        patches,is_value,out,e,lse,counts,n,query_rows,d,pc,common,value_count,hc,limit,pt,
        q.stride(0),k.stride(0),pk.stride(0),query_start,query_offset,d**-.5 if scale is None else scale,
        max(16,triton.next_power_of_2(d)),slots,evidence,triton.next_power_of_2(pc),num_warps=4,num_stages=2)
    return out,e,lse,counts


@triton.jit
def _history(ORDER,BEFORE,HIST,HC,IDS,ENDS,STARTS,SIZES,MEM,COMMON,ISVALUE,OUT,
        P:tl.constexpr,S:tl.constexpr,HN:tl.constexpr,HCAP:tl.constexpr,CP:tl.constexpr,
        VC:tl.constexpr,PP:tl.constexpr,PTILES:tl.constexpr):
    group,chunk=tl.program_id(0),tl.program_id(1)
    rank=chunk*4+tl.arange(0,4);cc=tl.arange(0,64)
    begin=tl.load(BEFORE+group*S+rank,rank<S,HN)
    active=(rank<S)&(begin<HN)
    if tl.sum(active.to(tl.int32),0)>0:
        slot=tl.load(ORDER+group*S+rank,rank<S,0)
        common=slot<CP;historical=(slot>=CP)&(slot<CP+tl.cdiv(HCAP,64))
        col=slot[:,None]*64+cc[None,:]
        global_id=tl.load(COMMON+col,active[:,None]&common[:,None]&(col<VC),0)
        common_local=global_id-(global_id//(PP+5)+1)*5
        hcol=(slot[:,None]-CP)*64+cc[None,:];hn=tl.load(HC+group)
        hlocal=tl.load(HIST+group*HCAP+tl.maximum(hcol,0),active[:,None]&historical[:,None]&(hcol<hn),-1)
        rslot=slot-CP-tl.cdiv(HCAP,64)
        parent_rank=tl.maximum(rslot,0)//PTILES
        parent=tl.load(IDS+group*P+parent_rank,active&(rslot>=0)&(parent_rank<P),-1)
        start=tl.load(STARTS+tl.maximum(parent,0),parent>=0,0)
        size=tl.load(SIZES+tl.maximum(parent,0),parent>=0,0)
        rcol=(tl.maximum(rslot,0)%PTILES)[:,None]*64+cc[None,:]
        region=~common&~historical
        rvalid=active[:,None]&region[:,None]&(parent[:,None]>=0)&(rcol<size[:,None])
        rlocal=tl.load(MEM+start[:,None]+rcol,rvalid,-1)
        duplicate=tl.load(ISVALUE+tl.maximum(rlocal,0),rvalid&(rlocal>=0),0)
        valid=(active[:,None]&common[:,None]&(col<VC))|((hlocal>=0)&historical[:,None])|(rvalid&(rlocal>=0)&(duplicate==0))
        local=tl.where(common[:,None],common_local,tl.where(historical[:,None],hlocal,rlocal))
        within=tl.cumsum(valid.to(tl.int32),1)-1
        dest=begin[:,None]+within
        tl.store(OUT+group*HN+dest,local,valid&(dest<HN))


def descriptor_history(e,lse,tile_counts,common,history,history_counts,parent_ids,parent_ends,
        parent_starts,member_bank,is_value,value_count,history_count,patches_per_frame,*,parent_sizes,parent_stride):
    groups=len(parent_ids);slots=tile_counts.shape[1]
    mass=(e-lse[:,:,None]).exp().sum(0)
    mass=torch.nn.functional.pad(mass,(0,0,0,groups*32-len(mass))).reshape(groups,32,slots).sum(1)
    order=mass.masked_fill(tile_counts==0,-1).argsort(dim=1,descending=True,stable=True)
    counts=tile_counts.gather(1,order);before=counts.cumsum(1)-counts
    out=torch.empty((groups,history_count),device=e.device,dtype=torch.int32)
    _history[(groups,triton.cdiv(slots,4))](order,before,history,history_counts,parent_ids,parent_ends,parent_starts,parent_sizes,
        member_bank,common,is_value,out,parent_ids.shape[1],slots,history_count,history.shape[1],
        triton.cdiv(len(common),64),value_count,patches_per_frame,triton.cdiv(parent_stride,64),num_warps=4)
    return out
