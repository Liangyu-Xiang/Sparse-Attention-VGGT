"""Shared representative bank + indexed extras, exact partition normalizers.

Grid log normalizers (weighted and unit-key) replace a repeated selected-QK
pass. Logical support slots are retained independently of physical grid order.
"""
import torch
import triton
import triton.language as tl

BASE_QUERY_TILE=64
BASE_NUM_WARPS=8
BASE_SEGMENTED=True
BASE_SPLITS=8
BASE_SPLIT_TILE=16
BASE_SPLIT_WARPS=4


@triton.jit
def _route(IDS,WEIGHTS,LABEL,REPS,GRID,RBIAS,RSLOT,COUNTS,RANK,A:tl.constexpr,F:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
    c=tl.load(LABEL+tl.maximum(key,0),valid,0);rep=tl.load(REPS+c,valid,-1)
    is_rep=valid&(key==rep);weight=tl.load(WEIGHTS+g*A+col,valid,1.)
    tl.store(RBIAS+g*F+c,tl.log(weight),is_rep);tl.store(RSLOT+g*F+c,col,is_rep)
    region=tl.load(GRID+tl.maximum(key,0),valid,0)
    rank=tl.atomic_add(COUNTS+g*R+region,1,valid&~is_rep,sem='relaxed')
    tl.store(RANK+g*A+col,tl.where(valid&~is_rep,rank,-1),col<A)


@triton.jit
def _extras(IDS,PATCH,GRID,RANK,OFFSETS,EIDS,ESLOT,EREGION,A:tl.constexpr,R:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    rank=tl.load(RANK+g*A+col,col<A,-1);valid=(col<A)&(rank>=0)
    key=tl.load(IDS+g*A+col,valid,0)
    region=tl.load(GRID+key,valid,0);begin=tl.load(OFFSETS+g*(R+2)+region,valid,0)
    actual=tl.load(PATCH+key,valid,0);position=begin+rank
    tl.store(EIDS+g*A+position,actual,valid);tl.store(ESLOT+g*A+position,col,valid);tl.store(EREGION+g*A+position,region,valid)


def prepare_bank(k,v,reps,patches,specials,grid,regions):
    keys=torch.cat((patches[reps],specials)).long()
    key_grid=torch.cat((grid[reps],torch.full_like(specials,regions)))
    order=key_grid.argsort(stable=True)
    clusters=torch.cat((torch.arange(len(reps),device=k.device),torch.full_like(specials,-1)))[order].int().contiguous()
    count=torch.bincount(key_grid,minlength=regions+1).int()
    offsets=torch.cat((torch.zeros(1,device=k.device,dtype=torch.int32),count.cumsum(0,dtype=torch.int32))).contiguous()
    return dict(k=k[:,keys[order]].contiguous(),v=v[:,keys[order]].contiguous(),clusters=clusters,regions=key_grid[order].int().contiguous(),offsets=offsets)


def route(ids,weights,labels,reps,patches,grid,regions):
    g,a=ids.shape;f=len(reps)
    bias=torch.full((g,f),-torch.inf,device=ids.device);slots=torch.full((g,f),-1,device=ids.device,dtype=torch.int32)
    counts=torch.zeros((g,regions),device=ids.device,dtype=torch.int32);rank=torch.empty((g,a),device=ids.device,dtype=torch.int32)
    _route[(g,triton.cdiv(a,256))](ids,weights,labels,reps,grid,bias,slots,counts,rank,a,f,regions,256,num_warps=4)
    prefix=counts.cumsum(1,dtype=torch.int32)
    offsets=torch.cat((torch.zeros((g,1),device=ids.device,dtype=torch.int32),prefix,prefix[:,-1:]),1).contiguous()
    extras=torch.empty((g,a),device=ids.device,dtype=torch.int32);logical=torch.empty_like(extras);extra_grid=torch.empty_like(extras)
    _extras[(g,triton.cdiv(a,256))](ids,patches,grid,rank,offsets,extras,logical,extra_grid,a,regions,256,num_warps=4)
    return bias,slots,offsets,extras,logical,extra_grid


@triton.jit
def _bank(Q,K,V,OFFSETS,META,BIAS,SLOT,IDS,POOL,OUT,LSE,ZW,ZU,DELTA,
          BASE:tl.constexpr,H:tl.constexpr,NQ:tl.constexpr,NK:tl.constexpr,A:tl.constexpr,F:tl.constexpr,R:tl.constexpr,D:tl.constexpr,
          QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    block,head=tl.program_id(0),tl.program_id(1)
    rr=block*BM+tl.arange(0,BM);dd=tl.arange(0,BD);cc=tl.arange(0,BN)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<NQ)&(dd[None,:]<D),0)
    groups=block*(BM//32)+tl.arange(0,BM//32)
    nrows=tl.minimum(32,NQ-groups*32)
    qmean=tl.sum(tl.reshape(q.to(tl.float32),(BM//32,32,BD)),1)/nrows[:,None]
    maximum=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
    for region in range(R+1):
        if BASE:
            begin=tl.load(OFFSETS+region);end=tl.load(OFFSETS+region+1)
        else:
            begin=tl.load(OFFSETS+block*(R+2)+region);end=tl.load(OFFSETS+block*(R+2)+region+1)
        gm=tl.full((BM,),-float('inf'),tl.float32);gd=tl.zeros((BM,),tl.float32)
        um=tl.full((BM,),-float('inf'),tl.float32);ud=tl.zeros((BM,),tl.float32)
        if region<R:
            pooled=tl.load(POOL+head*R*D+region*D+dd,dd<D,0).to(tl.float32)
            reference=tl.sum(qmean*pooled[None,:],1)*SCALE
        else:reference=tl.full((BM//32,),0.,tl.float32)
        for tile in range(tl.cdiv(end-begin,BN)):
            col=begin+tile*BN+cc;valid=col<end
            if BASE:
                key=col
                cluster=tl.load(META+col,valid,-1)
                bias=tl.load(BIAS+(rr[:,None]//32)*F+tl.maximum(cluster,0)[None,:],(rr[:,None]<NQ)&valid[None,:],-float('inf'))
                bias=tl.where((cluster[None,:]<0)&valid[None,:],0.,bias)
                logical=tl.load(SLOT+groups[:,None]*F+tl.maximum(cluster,0)[None,:],(groups[:,None]*32<NQ)&valid[None,:]&(cluster[None,:]>=0),-1)
            else:
                key=tl.load(IDS+block*A+col,valid,0)
                position=tl.load(META+block*A+col,valid,0)
                bb=tl.load(BIAS+block*A+position,valid,-float('inf'))
                bias=tl.broadcast_to(bb[None,:],(BM,BN))
                logical=tl.reshape(tl.where(valid,position,-1),(1,BN))
            kk=tl.load(K+head*KH+key[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
            vv=tl.load(V+head*KH+key[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
            raw=tl.dot(q,kk,input_precision='ieee')*SCALE
            active=(rr[:,None]<NQ)&valid[None,:]&(bias!=-float('inf'))
            z=tl.where(active,raw+bias,-float('inf'));u=tl.where(active,raw,-float('inf'))
            mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
            alpha=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
            acc=acc*alpha[:,None]+tl.dot(p.to(vv.dtype),vv,input_precision='ieee')
            den=den*alpha+tl.sum(p,1);maximum=mx
            mm=tl.maximum(gm,tl.max(z,1));ms=tl.where(mm==-float('inf'),0.,mm)
            gd=gd*tl.exp(gm-ms)+tl.sum(tl.exp(z-ms[:,None]),1);gm=mm
            mmu=tl.maximum(um,tl.max(u,1));mus=tl.where(mmu==-float('inf'),0.,mmu)
            ud=ud*tl.exp(um-mus)+tl.sum(tl.exp(u-mus[:,None]),1);um=mmu
            if region<R:
                average=tl.sum(tl.reshape(tl.where(rr[:,None]<NQ,raw,0.),(BM//32,32,BN)),1)/nrows[:,None]
                delta=(average-reference[:,None])/H
                tl.atomic_add(DELTA+groups[:,None]*A+logical,delta,(groups[:,None]*32<NQ)&(logical>=0)&valid[None,:],sem='relaxed')
        if region<R:
            tl.store(ZW+(head*R+region)*NQ+rr,gm+tl.log(gd),rr<NQ)
            tl.store(ZU+(head*R+region)*NQ+rr,um+tl.log(ud),rr<NQ)
    tl.store(OUT+(head*NQ+rr[:,None])*D+dd[None,:],tl.where(den[:,None]>0,acc/tl.maximum(den[:,None],1.e-30),0.),(rr[:,None]<NQ)&(dd[None,:]<D))
    tl.store(LSE+head*NQ+rr,maximum+tl.log(den),rr<NQ)


@triton.jit
def _merge(OA,OB,LA,LB,O,L,H:tl.constexpr,N:tl.constexpr,D:tl.constexpr,B:tl.constexpr,BD:tl.constexpr):
    head=tl.program_id(0);row=tl.program_id(1)*B+tl.arange(0,B);dd=tl.arange(0,BD)
    a=tl.load(LA+head*N+row,row<N,-float('inf'));b=tl.load(LB+head*N+row,row<N,-float('inf'))
    maximum=tl.maximum(a,b);lse=maximum+tl.log(tl.exp(a-maximum)+tl.exp(b-maximum))
    aa=tl.load(OA+(head*N+row[:,None])*D+dd[None,:],(row[:,None]<N)&(dd[None,:]<D),0.)
    bb=tl.load(OB+(head*N+row[:,None])*D+dd[None,:],(row[:,None]<N)&(dd[None,:]<D),0.)
    value=tl.exp(a-lse)[:,None]*aa+tl.exp(b-lse)[:,None]*bb
    tl.store(O+(head*N+row[:,None])*D+dd[None,:],value,(row[:,None]<N)&(dd[None,:]<D))
    tl.store(L+head*N+row,lse,row<N)


@triton.jit
def _summarize(AW,BW,AU,BU,L,GW,GU,H:tl.constexpr,N:tl.constexpr,R:tl.constexpr,B:tl.constexpr,BH:tl.constexpr):
    group=tl.program_id(0);region=tl.program_id(1)*B+tl.arange(0,B)
    hq=tl.arange(0,BH);head=hq//32;row=group*32+hq%32;valid=(head<H)&(row<N)
    lse=tl.load(L+head*N+row,valid,0.)
    offset=(head[:,None]*R+region[None,:])*N+row[:,None];mask=valid[:,None]&(region[None,:]<R)
    a=tl.load(AW+offset,mask,-float('inf'));b=tl.load(BW+offset,mask,-float('inf'))
    u=tl.load(AU+offset,mask,-float('inf'));v=tl.load(BU+offset,mask,-float('inf'))
    weighted=tl.sum(tl.exp(a-lse[:,None])+tl.exp(b-lse[:,None]),0)/(H*tl.minimum(32,N-group*32))
    unit=tl.sum(tl.exp(u-lse[:,None])+tl.exp(v-lse[:,None]),0)/(H*tl.minimum(32,N-group*32))
    tl.store(GW+group*R+region,weighted,region<R);tl.store(GU+group*R+region,unit,region<R)


def attention(q,k,v,ids,weights,patches,labels,reps,specials,grid,pooled,bank,scale):
    h,n,d=q.shape;g,a=ids.shape;r=pooled.shape[1];f=len(reps)
    rbias,rslot,offsets,extras,logical,extra_grid=route(ids,weights,labels,reps,patches,grid,r)
    delta=torch.zeros((g,a),device=q.device)
    outputs=[torch.empty((BASE_SPLITS,h,n,d),device=q.device) if i==0 and BASE_SPLITS else torch.empty((h,n,d),device=q.device) for i in range(2)]
    lses=[torch.empty((BASE_SPLITS,h,n),device=q.device) if i==0 and BASE_SPLITS else torch.empty((h,n),device=q.device) for i in range(2)]
    stats=[torch.full((h,r,n),-torch.inf,device=q.device) for _ in range(4)]
    bd=max(16,triton.next_power_of_2(d))
    common=(h,n,bank['k'].shape[1],a,f,r,d)
    if BASE_SPLITS:
        _common_split[(triton.cdiv(n,BASE_SPLIT_TILE),h,BASE_SPLITS)](q,bank['k'],bank['v'],bank['offsets'],bank['clusters'],rbias,rslot,pooled,outputs[0],lses[0],stats[0],stats[2],delta,h,n,a,f,r,d,q.stride(0),bank['k'].stride(0),scale,BASE_SPLIT_TILE,bd,BASE_SPLITS,num_warps=BASE_SPLIT_WARPS)
    elif BASE_SEGMENTED:
        _common_bank[(triton.cdiv(n,BASE_QUERY_TILE),h)](q,bank['k'],bank['v'],bank['clusters'],bank['regions'],rbias,rslot,pooled,outputs[0],lses[0],stats[0],stats[2],delta,*common,q.stride(0),bank['k'].stride(0),scale,BASE_QUERY_TILE,64,bd,num_warps=BASE_NUM_WARPS)
    else:
        _bank[(triton.cdiv(n,BASE_QUERY_TILE),h)](q,bank['k'],bank['v'],bank['offsets'],bank['clusters'],rbias,rslot,extras,pooled,outputs[0],lses[0],stats[0],stats[2],delta,True,*common,q.stride(0),bank['k'].stride(0),scale,BASE_QUERY_TILE,64,bd,num_warps=BASE_NUM_WARPS)
    _extra_bank[(g,h)](q,k,v,offsets,logical,weights.log().contiguous(),extras,extra_grid,pooled,outputs[1],lses[1],stats[1],stats[3],delta,h,n,k.shape[1],a,r,d,q.stride(0),k.stride(0),scale,32,bd,num_warps=4)
    output=torch.empty_like(q);lse=torch.empty((h,n),device=q.device)
    if BASE_SPLITS:
        _merge_parts[(h,triton.cdiv(n,32))](*outputs,*lses,output,lse,h,n,d,BASE_SPLITS,32,bd,triton.next_power_of_2(BASE_SPLITS),num_warps=4)
    else:_merge[(h,triton.cdiv(n,32))](*outputs,*lses,output,lse,h,n,d,32,bd,num_warps=4)
    gw=torch.empty((g,r),device=q.device);gu=torch.empty_like(gw)
    _summarize[(g,triton.cdiv(r,8))](stats[0],stats[1],stats[2],stats[3],lse,gw,gu,h,n,r,8,triton.next_power_of_2(h*32),num_warps=4)
    total=gw.sum(1,keepdim=True).clamp_min(1.e-30)
    return output,delta,gw/total,gu/total


@triton.jit
def _sample(Q,K,V,IDS,W,SPECIAL,O,H:tl.constexpr,G:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,N:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr):
    g,head=tl.program_id(0),tl.program_id(1);dd=tl.arange(0,BD);rr=tl.arange(0,16);cc=tl.arange(0,64)
    q=tl.load(Q+(head*G+g)*D+dd,dd<D,0)
    qq=tl.broadcast_to(q[None,:],(16,BD));maximum=tl.full((16,),-float('inf'),tl.float32);den=tl.zeros((16,),tl.float32);acc=tl.zeros((16,BD),tl.float32)
    for bank in tl.static_range(2):
        for tile in range(tl.cdiv(S,64) if bank==0 else tl.cdiv(A,64)):
            col=tile*64+cc
            if bank==0:
                key=tl.load(SPECIAL+col,col<S,0);valid=col<S;bias=tl.full((64,),0.,tl.float32)
            else:
                key=tl.load(IDS+g*A+col,col<A,-1);valid=(col<A)&(key>=0)
                weight=tl.load(W+g*A+col,col<A,0.);bias=tl.log(weight)
            kk=tl.load(K+head*N*D+tl.maximum(key,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
            vv=tl.load(V+head*N*D+tl.maximum(key,0)[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
            z=tl.where(valid[None,:],tl.dot(qq,kk,input_precision='ieee')*SCALE+bias[None,:],-float('inf'))
            mx=tl.maximum(maximum,tl.max(z,1));p=tl.exp(z-mx[:,None]);alpha=tl.exp(maximum-mx)
            acc=acc*alpha[:,None]+tl.dot(p.to(vv.dtype),vv,input_precision='ieee');den=den*alpha+tl.sum(p,1);maximum=mx
    result=tl.sum(tl.where(rr[:,None]==0,acc/den[:,None],0.),0)
    tl.store(O+(head*G+g)*D+dd,result,dd<D)


def sample(q,k,v,actual_ids,weights,specials,scale):
    h,g,d=q.shape;output=torch.empty((h,g,d),device=q.device)
    _sample[(g,h)](q.contiguous(),k,v,actual_ids,weights,specials,output,h,g,actual_ids.shape[1],len(specials),d,k.shape[1],scale,max(16,triton.next_power_of_2(d)),num_warps=4)
    return output


@triton.jit
def _segment_add(ida,a,b,idb,c,d):
    same=ida==idb
    return idb,tl.where(same,a+c,c),tl.where(same,b+d,d)


@triton.jit
def _extra_bank(Q,K,V,OFFSETS,META,BIAS,IDS,REGION,POOL,OUT,LSE,ZW,ZU,DELTA,
                H:tl.constexpr,NQ:tl.constexpr,NK:tl.constexpr,A:tl.constexpr,R:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    group,head=tl.program_id(0),tl.program_id(1)
    rr=group*32+tl.arange(0,32);dd=tl.arange(0,BD);cc=tl.arange(0,BN)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<NQ)&(dd[None,:]<D),0)
    rows=tl.minimum(32,NQ-group*32);qmean=tl.sum(q.to(tl.float32),0)/rows
    total=tl.load(OFFSETS+group*(R+2)+R)
    maximum=tl.full((32,),-float('inf'),tl.float32);den=tl.zeros((32,),tl.float32);acc=tl.zeros((32,BD),tl.float32)
    for tile in range(tl.cdiv(total,BN)):
        col=tile*BN+cc;valid=col<total
        key=tl.load(IDS+group*A+col,valid,0);position=tl.load(META+group*A+col,valid,0)
        region=tl.load(REGION+group*A+col,valid,R)
        bias=tl.load(BIAS+group*A+position,valid,-float('inf'))
        kk=tl.load(K+head*KH+key[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
        vv=tl.load(V+head*KH+key[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
        raw=tl.dot(q,kk,input_precision='ieee')*SCALE
        active=(rr[:,None]<NQ)&valid[None,:]
        z=tl.where(active,raw+bias[None,:],-float('inf'));u=tl.where(active,raw,-float('inf'))
        mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
        p=tl.exp(z-safe[:,None]);alpha=tl.exp(maximum-safe)
        acc=acc*alpha[:,None]+tl.dot(p.to(vv.dtype),vv,input_precision='ieee');den=den*alpha+tl.sum(p,1);maximum=mx
        zm=safe;um=safe;pw=p
        inverse=tl.where(valid,tl.exp(-bias),0.)
        pu=p*inverse[None,:]
        ids=tl.broadcast_to(region[None,:],(32,BN))
        _,sw,su=tl.associative_scan((ids,pw,pu),1,_segment_add)
        next_region=tl.gather(region,tl.minimum(cc+1,BN-1),0)
        end=valid&((cc==BN-1)|(region!=next_region))
        offset=(head*R+region[None,:])*NQ+rr[:,None]
        mask=(rr[:,None]<NQ)&end[None,:]
        oldw=tl.load(ZW+offset,mask,-float('inf'));oldu=tl.load(ZU+offset,mask,-float('inf'))
        curw=zm[:,None]+tl.log(sw);curu=um[:,None]+tl.log(su)
        wm=tl.maximum(oldw,curw);uu=tl.maximum(oldu,curu)
        neww=wm+tl.log(tl.exp(oldw-wm)+tl.exp(curw-wm));newu=uu+tl.log(tl.exp(oldu-uu)+tl.exp(curu-uu))
        tl.store(ZW+offset,neww,mask);tl.store(ZU+offset,newu,mask)
        pooled=tl.load(POOL+head*R*D+tl.minimum(region,R-1)[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0).to(tl.float32)
        reference=tl.sum(pooled*qmean[None,:],1)*SCALE
        delta=(tl.sum(tl.where(rr[:,None]<NQ,raw,0.),0)/rows-reference)/H
        tl.atomic_add(DELTA+group*A+position,delta,valid,sem='relaxed')
        tl.debug_barrier()
    tl.store(OUT+(head*NQ+rr[:,None])*D+dd[None,:],tl.where(den[:,None]>0,acc/tl.maximum(den[:,None],1.e-30),0.),(rr[:,None]<NQ)&(dd[None,:]<D))
    tl.store(LSE+head*NQ+rr,maximum+tl.log(den),rr<NQ)


@triton.jit
def _common_bank(Q,K,V,CLUSTER,REGION,BIAS,SLOT,POOL,OUT,LSE,ZW,ZU,DELTA,
                H:tl.constexpr,NQ:tl.constexpr,NB:tl.constexpr,A:tl.constexpr,F:tl.constexpr,R:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr):
    block,head=tl.program_id(0),tl.program_id(1)
    rr=block*BM+tl.arange(0,BM);dd=tl.arange(0,BD);cc=tl.arange(0,BN)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<NQ)&(dd[None,:]<D),0)
    groups=block*(BM//32)+tl.arange(0,BM//32)
    rows=tl.maximum(1,tl.minimum(32,NQ-groups*32))
    qmean=tl.sum(tl.reshape(q.to(tl.float32),(BM//32,32,BD)),1)/rows[:,None]
    maximum=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
    for tile in range(tl.cdiv(NB,BN)):
        col=tile*BN+cc;valid=col<NB
        cluster=tl.load(CLUSTER+col,valid,-1);region=tl.load(REGION+col,valid,R+1)
        bias=tl.load(BIAS+(rr[:,None]//32)*F+tl.maximum(cluster,0)[None,:],(rr[:,None]<NQ)&valid[None,:],-float('inf'))
        bias=tl.where((cluster[None,:]<0)&valid[None,:],0.,bias)
        logical=tl.load(SLOT+groups[:,None]*F+tl.maximum(cluster,0)[None,:],(groups[:,None]*32<NQ)&valid[None,:]&(cluster[None,:]>=0),-1)
        kk=tl.load(K+head*KH+col[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
        vv=tl.load(V+head*KH+col[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
        raw=tl.dot(q,kk,input_precision='ieee')*SCALE
        active=(rr[:,None]<NQ)&valid[None,:]&(bias!=-float('inf'))
        z=tl.where(active,raw+bias,-float('inf'));u=tl.where(active,raw,-float('inf'))
        mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
        pw=tl.exp(z-safe[:,None]);alpha=tl.exp(maximum-safe)
        acc=acc*alpha[:,None]+tl.dot(pw.to(vv.dtype),vv,input_precision='ieee');den=den*alpha+tl.sum(pw,1);maximum=mx
        um=tl.max(u,1);um=tl.where(um==-float('inf'),0.,um);pu=tl.exp(u-um[:,None])
        rids=tl.broadcast_to(region[None,:],(BM,BN))
        _,sw,su=tl.associative_scan((rids,pw,pu),1,_segment_add)
        next_region=tl.gather(region,tl.minimum(cc+1,BN-1),0)
        end=valid&((cc==BN-1)|(region!=next_region))&(region<R)
        offset=(head*R+tl.minimum(region,R-1)[None,:])*NQ+rr[:,None]
        mask=(rr[:,None]<NQ)&end[None,:]
        oldw=tl.load(ZW+offset,mask,-float('inf'));oldu=tl.load(ZU+offset,mask,-float('inf'))
        curw=safe[:,None]+tl.log(sw);curu=um[:,None]+tl.log(su)
        wm=tl.maximum(oldw,curw);uu=tl.maximum(oldu,curu)
        ws=tl.where(wm==-float('inf'),0.,wm);us=tl.where(uu==-float('inf'),0.,uu)
        neww=ws+tl.log(tl.exp(oldw-ws)+tl.exp(curw-ws));newu=us+tl.log(tl.exp(oldu-us)+tl.exp(curu-us))
        tl.store(ZW+offset,neww,mask);tl.store(ZU+offset,newu,mask)
        pooled=tl.load(POOL+head*R*D+tl.minimum(region,R-1)[:,None]*D+dd[None,:],valid[:,None]&(cluster[:,None]>=0)&(dd[None,:]<D),0).to(tl.float32)
        reference=tl.sum(qmean[:,None,:]*pooled[None,:,:],2)*SCALE
        average=tl.sum(tl.reshape(tl.where(rr[:,None]<NQ,raw,0.),(BM//32,32,BN)),1)/rows[:,None]
        tl.atomic_add(DELTA+groups[:,None]*A+logical,(average-reference)/H,(groups[:,None]*32<NQ)&(logical>=0)&valid[None,:],sem='relaxed')
        tl.debug_barrier()
    tl.store(OUT+(head*NQ+rr[:,None])*D+dd[None,:],tl.where(den[:,None]>0,acc/tl.maximum(den[:,None],1.e-30),0.),(rr[:,None]<NQ)&(dd[None,:]<D))
    tl.store(LSE+head*NQ+rr,maximum+tl.log(den),rr<NQ)


@triton.jit
def _common_split(Q,K,V,OFFSETS,CLUSTER,BIAS,SLOT,POOL,OUT,LSE,ZW,ZU,DELTA,
                  H:tl.constexpr,NQ:tl.constexpr,A:tl.constexpr,F:tl.constexpr,R:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BD:tl.constexpr,PARTS:tl.constexpr):
    block,head,part=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    rr=block*BM+tl.arange(0,BM);dd=tl.arange(0,BD);cc=tl.arange(0,64);group=block*BM//32
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<NQ)&(dd[None,:]<D),0)
    group_rows=tl.minimum(32,NQ-group*32);qmean=tl.sum(q.to(tl.float32),0)/group_rows
    maximum=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
    chunk=tl.cdiv(R+1,PARTS)
    for region in range(part*chunk,tl.minimum((part+1)*chunk,R+1)):
        begin=tl.load(OFFSETS+region);end=tl.load(OFFSETS+region+1)
        if end>begin:
            gd=tl.zeros((BM,),tl.float32);ud=tl.zeros((BM,),tl.float32)
            if region<R:
                pooled=tl.load(POOL+head*R*D+region*D+dd,dd<D,0).to(tl.float32)
                reference=tl.sum(qmean*pooled,0)*SCALE
            else:reference=0.
            for tile in range(tl.cdiv(end-begin,64)):
                col=begin+tile*64+cc;valid=col<end
                cluster=tl.load(CLUSTER+col,valid,-1)
                bias=tl.load(BIAS+group*F+tl.maximum(cluster,0),valid,-float('inf'))
                bias=tl.where(valid&(cluster<0),0.,bias)
                logical=tl.load(SLOT+group*F+tl.maximum(cluster,0),valid&(cluster>=0),-1)
                kk=tl.load(K+head*KH+col[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
                vv=tl.load(V+head*KH+col[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
                raw=tl.dot(q,kk,input_precision='ieee')*SCALE
                active=(rr[:,None]<NQ)&valid[None,:]&(bias[None,:]!=-float('inf'))
                z=tl.where(active,raw+bias[None,:],-float('inf'))
                m=tl.maximum(maximum,tl.max(z,1));safe=tl.where(m==-float('inf'),0.,m)
                pp=tl.exp(z-safe[:,None]);alpha=tl.exp(maximum-safe)
                acc=acc*alpha[:,None]+tl.dot(pp.to(vv.dtype),vv,input_precision='ieee')
                step=tl.sum(pp,1);den=den*alpha+step;gd=gd*alpha+step
                inverse=tl.where(valid&(bias!=-float('inf')),tl.exp(-bias),0.)
                ud=ud*alpha+tl.sum(pp*inverse[None,:],1);maximum=m
                if region<R:
                    avg=tl.sum(tl.where(rr[:,None]<NQ,raw,0.),0)/group_rows
                    tl.atomic_add(DELTA+group*A+logical,(avg-reference)/H,valid&(logical>=0),sem='relaxed')
            if region<R:
                tl.store(ZW+(head*R+region)*NQ+rr,maximum+tl.log(gd),rr<NQ)
                tl.store(ZU+(head*R+region)*NQ+rr,maximum+tl.log(ud),rr<NQ)
    tl.store(OUT+((part*H+head)*NQ+rr[:,None])*D+dd[None,:],tl.where(den[:,None]>0,acc/tl.maximum(den[:,None],1.e-30),0.),(rr[:,None]<NQ)&(dd[None,:]<D))
    tl.store(LSE+(part*H+head)*NQ+rr,maximum+tl.log(den),rr<NQ)


@triton.jit
def _merge_parts(OA,OB,LA,LB,O,L,H:tl.constexpr,N:tl.constexpr,D:tl.constexpr,PARTS:tl.constexpr,B:tl.constexpr,BD:tl.constexpr,BP:tl.constexpr):
    head=tl.program_id(0);rr=tl.program_id(1)*B+tl.arange(0,B);dd=tl.arange(0,BD);part=tl.arange(0,BP)
    ll=tl.load(LA+(part[None,:]*H+head)*N+rr[:,None],(rr[:,None]<N)&(part[None,:]<PARTS),-float('inf'))
    other=tl.load(LB+head*N+rr,rr<N,-float('inf'))
    mx=tl.maximum(tl.max(ll,1),other)
    total=tl.sum(tl.exp(ll-mx[:,None]),1)+tl.exp(other-mx);lse=mx+tl.log(total)
    out=tl.zeros((B,BD),tl.float32)
    for i in tl.static_range(PARTS):
        value=tl.load(OA+((i*H+head)*N+rr[:,None])*D+dd[None,:],(rr[:,None]<N)&(dd[None,:]<D),0.)
        weight=tl.load(LA+(i*H+head)*N+rr,rr<N,-float('inf'))
        out+=value*tl.exp(weight-lse)[:,None]
    value=tl.load(OB+(head*N+rr[:,None])*D+dd[None,:],(rr[:,None]<N)&(dd[None,:]<D),0.)
    out+=value*tl.exp(other-lse)[:,None]
    tl.store(O+(head*N+rr[:,None])*D+dd[None,:],out,(rr[:,None]<N)&(dd[None,:]<D))
    tl.store(L+head*N+rr,lse,rr<N)
