"""Continuous64-Key tiles with fused per-grid segment statistics; one QK."""
import torch,triton
import triton.language as tl
from .reuse_split_grid_attention import _merge

@triton.jit
def _segment(a,b,fa,c,d,fb):
    return tl.where(fb,c,a+c),tl.where(fb,d,b+d),fa|fb

@triton.jit
def _attention(Q,K,V,IDS,BIAS,ORDER,GRID,SPECIAL,O,LSE,PROBE,LOGS,
               H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
               REGIONS:tl.constexpr,PARTS:tl.constexpr,TP:tl.constexpr,TILES:tl.constexpr,
               QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr,EVIDENCE:tl.constexpr):
    group=tl.program_id(0);head=tl.program_id(1);part=tl.program_id(2)
    row=tl.arange(0,32);rr=group*32+row;dd=tl.arange(0,BD);cc=tl.arange(0,64)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<R)&(dd[None,:]<D),0)
    maximum=tl.full((32,),-float('inf'),tl.float32);den=tl.zeros((32,),tl.float32);acc=tl.zeros((32,BD),tl.float32)
    for bank in tl.static_range(2):
        for tile in range(0 if bank==0 else part*TP,(tl.cdiv(S,64) if part==0 else 0) if bank==0 else tl.minimum((part+1)*TP,TILES)):
            col=tile*64+cc
            if bank==0:
                ids=tl.load(SPECIAL+col,col<S,0);valid=col<S;bias=tl.full((64,),0.,tl.float32)
            else:
                ids=tl.load(IDS+group*A+col,col<A,-1);valid=(col<A)&(ids>=0)
                bias=tl.load(BIAS+group*A+col,col<A,-float('inf'))
            k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
            v=tl.load(V+head*KH+tl.maximum(ids,0)[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
            qk=tl.dot(q,k,allow_tf32=False)
            z=tl.where(valid[None,:],qk*SCALE+bias[None,:],-float('inf'))
            if EVIDENCE and bank==1:
                probe=tl.arange(0,4);pr=tl.minimum(probe*8+4,R-group*32-1)
                raw=tl.where(valid[None,:],qk*SCALE,-float('inf'))
                values=tl.gather(raw,tl.broadcast_to(pr[:,None],(4,64)),0)
                logical=tl.load(ORDER+group*A+col,col<A,0)
                tl.store(PROBE+((group*H+head)*4+probe[:,None])*A+logical[None,:],values,col[None,:]<A)
            mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
            correction=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
            acc=acc*correction[:,None]+tl.dot(p.to(v.dtype),v,allow_tf32=False)
            den=den*correction+tl.sum(p,1);maximum=mx
            if bank==1:
                region=tl.load(GRID+group*A+col,valid,-1)
                prev=tl.load(GRID+group*A+col-1,valid&(cc>0),-2)
                nxt=tl.load(GRID+group*A+col+1,(col+1<A)&(cc<63),-2)
                flag=(cc==0)|(region!=prev)
                unbiased=p*tl.where(valid,tl.exp(-bias),0.)[None,:]
                weighted,unit,_=tl.associative_scan((p,unbiased,tl.broadcast_to(flag[None,:],(32,64))),1,_segment)
                last=valid&((cc==63)|(region!=nxt)|(col==A-1))
                offset=(((group*H+head)*2)*(REGIONS+TILES)+region[None,:]+tile)*32+row[:,None]
                tl.store(LOGS+offset,safe[:,None]+tl.log(weighted),last[None,:])
                tl.store(LOGS+offset+(REGIONS+TILES)*32,safe[:,None]+tl.log(unit),last[None,:])
    tl.store(O+(((group*H+head)*PARTS+part)*32+row[:,None])*BD+dd[None,:],tl.where(den[:,None]>0,acc/tl.maximum(den[:,None],1.e-30),0.),dd[None,:]<D)
    tl.store(LSE+((group*H+head)*PARTS+part)*32+row,maximum+tl.log(den))

@triton.jit
def _summary(LOGS,LSE,OFF,OUT,H:tl.constexpr,R:tl.constexpr,REGIONS:tl.constexpr,TILES:tl.constexpr,BH:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);kind=tl.program_id(1);region=tl.program_id(2)*B+tl.arange(0,B)
    hr=tl.arange(0,BH*32);head=hr//32;row=hr%32;rr=g*32+row
    begin=tl.load(OFF+g*(REGIONS+1)+region,region<REGIONS,0)
    end=tl.load(OFF+g*(REGIONS+1)+region+1,region<REGIONS,0)
    lse=tl.load(LSE+head*R+rr,(head<H)&(rr<R),0.)
    probability=tl.zeros((BH*32,B),tl.float32)
    for tile in range(tl.min(begin//64,0),tl.max(tl.cdiv(end,64),0)):
        valid=(region<REGIONS)&(begin<end)&(tile>=begin//64)&(tile<tl.cdiv(end,64))
        value=tl.load(LOGS+(((g*H+head[:,None])*2+kind)*(REGIONS+TILES)+region[None,:]+tile)*32+row[:,None],(head[:,None]<H)&valid[None,:],-float('inf'))
        probability+=tl.where((head[:,None]<H)&(rr[:,None]<R),tl.exp(value-lse[:,None]),0.)
    result=tl.sum(probability,0)/(H*tl.minimum(32,R-g*32))
    tl.store(OUT+(g*2+kind)*REGIONS+region,result,region<REGIONS)

def weighted_attention(q,k,v,indices,log_weights,specials,scale,key_grid,regions,emit_evidence=True,evidence_layout=None,parts=8):
    h,r,d=q.shape;g,a=indices.shape
    bank=key_grid.masked_fill(indices<0,regions);order=bank.argsort(dim=1,stable=True)
    ids=indices.gather(1,order).contiguous();bias=log_weights.gather(1,order).contiguous();sg=bank.gather(1,order).int().contiguous()
    counts=torch.zeros((g,regions+1),device=q.device,dtype=torch.int32);counts.scatter_add_(1,bank.long(),torch.ones_like(indices,dtype=torch.int32))
    offsets=torch.nn.functional.pad(counts[:,:regions].cumsum(1,dtype=torch.int32),(1,0)).contiguous();order=order.int().contiguous()
    if emit_evidence and evidence_layout is not None:evidence_layout.update(selected_grid_order=order,selected_grid_offsets=offsets)
    bd=max(16,triton.next_power_of_2(d));tiles=triton.cdiv(a,64)
    output=torch.empty_like(q);lse=torch.empty((h,r),device=q.device);partial=torch.empty((g,h,parts,32,bd),device=q.device);plse=torch.empty((g,h,parts,32),device=q.device)
    probe=torch.full((g,h,4,a),-torch.inf,device=q.device) if emit_evidence else lse
    logs=torch.empty((g,h,2,regions+tiles,32),device=q.device);summary=torch.empty((g,2,regions),device=q.device)
    _attention[(g,h,parts)](q,k,v,ids,bias,order,sg,specials,partial,plse,probe,logs,h,r,a,len(specials),d,regions,parts,triton.cdiv(tiles,parts),tiles,q.stride(0),k.stride(0),scale,bd,emit_evidence,num_warps=4)
    _merge[(g,h,32)](partial,plse,output,lse,h,r,d,bd,parts,triton.next_power_of_2(parts),num_warps=4)
    _summary[(g,2,triton.cdiv(regions,4))](logs,lse,offsets,summary,h,r,regions,tiles,triton.next_power_of_2(h),4,num_warps=4)
    return output,summary,probe if emit_evidence else None
