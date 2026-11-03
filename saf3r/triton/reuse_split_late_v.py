"""Single-QK sparse attention with exact all-query grid probability summaries.

Keys are physically grouped by canonical grid; probe logits are written in the
original logical support order. Only per-grid row statistics are materialized.
"""
import torch,triton
import triton.language as tl

@triton.jit
def _attention(Q,K,V,IDS,BIAS,ORDER,OFFSETS,SPECIAL,O,LSE,PROBE,LOGGRID,RAWGRID,
               H:tl.constexpr,R:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
               REGIONS:tl.constexpr,PARTS:tl.constexpr,PER:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr,EVIDENCE:tl.constexpr,BK:tl.constexpr,STATIC_PROBES:tl.constexpr):
    group=tl.program_id(0);head=tl.program_id(1);part=tl.program_id(2)
    row=tl.arange(0,32);rr=group*32+row;dd=tl.arange(0,BD);cc=tl.arange(0,BK)
    q=tl.load(Q+head*QH+rr[:,None]*D+dd[None,:],(rr[:,None]<R)&(dd[None,:]<D),0)
    maximum=tl.full((32,),-float('inf'),tl.float32);den=tl.zeros((32,),tl.float32)
    acc=tl.zeros((32,BD),tl.float32)
    for bank in tl.static_range(2):
        for region in range(0 if bank==0 else part*PER, (1 if part==0 else 0) if bank==0 else tl.minimum((part+1)*PER,REGIONS)):
            if bank==0:
                begin=0;end=S
            else:
                begin=tl.load(OFFSETS+group*(REGIONS+1)+region)
                end=tl.load(OFFSETS+group*(REGIONS+1)+region+1)
            region_den=tl.zeros((32,),tl.float32);unit_den=tl.zeros((32,),tl.float32)
            rawmax=tl.full((4,),-float('inf'),tl.float32);rawden=tl.zeros((4,),tl.float32)
            for tile in range(tl.cdiv(end-begin,BK)):
                col=begin+tile*BK+cc
                if bank==0:
                    ids=tl.load(SPECIAL+col,col<end,0);valid=col<end;bias=tl.full((BK,),0.,tl.float32)
                else:
                    ids=tl.load(IDS+group*A+col,col<end,-1);valid=(col<end)&(ids>=0)
                    bias=tl.load(BIAS+group*A+col,col<end,-float('inf'))
                k=tl.load(K+head*KH+tl.maximum(ids,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0)
                qk=tl.dot(q,k,allow_tf32=False)
                z=tl.where(valid[None,:],qk*SCALE+bias[None,:],-float('inf'))
                if EVIDENCE and bank==1:
                    probe=tl.arange(0,4)
                    if STATIC_PROBES and R%32==0:pr=probe*8+4
                    else:pr=tl.minimum(probe*8+4,R-group*32-1)
                    raw=tl.where(valid[None,:],qk*SCALE,-float('inf'))
                    values=tl.gather(raw,tl.broadcast_to(pr[:,None],(4,BK)),0)
                    newmax=tl.maximum(rawmax,tl.max(values,1));safemax=tl.where(newmax==-float('inf'),0.,newmax)
                    rawden=rawden*tl.exp(rawmax-safemax)+tl.sum(tl.exp(values-safemax[:,None]),1);rawmax=newmax
                    logical=tl.load(ORDER+group*A+col,col<end,0)
                    tl.store(PROBE+((group*H+head)*4+probe[:,None])*A+logical[None,:],values,col[None,:]<end)
                mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
                correction=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
                v=tl.load(V+head*KH+tl.maximum(ids,0)[:,None]*D+dd[None,:],valid[:,None]&(dd[None,:]<D),0)
                acc=acc*correction[:,None]+tl.dot(p.to(v.dtype),v,allow_tf32=False)
                den=den*correction+tl.sum(p,1)
                region_den=region_den*correction+tl.sum(p,1)
                unit_den=unit_den*correction+tl.sum(p*tl.where(valid,tl.exp(-bias),0.)[None,:],1)
                maximum=mx
            if bank==1:
                if EVIDENCE:
                    probes=tl.arange(0,4)
                    tl.store(RAWGRID+((group*H+head)*4+probes)*REGIONS+region,rawmax+tl.log(rawden))
                base=((group*H+head)*2)*REGIONS*32+region*32+row
                tl.store(LOGGRID+base,maximum+tl.log(region_den))
                tl.store(LOGGRID+base+REGIONS*32,maximum+tl.log(unit_den))
    tl.store(O+(((group*H+head)*PARTS+part)*32+row[:,None])*BD+dd[None,:],tl.where(den[:,None]>0,acc/tl.maximum(den[:,None],1.e-30),0.),dd[None,:]<D)
    tl.store(LSE+((group*H+head)*PARTS+part)*32+row,maximum+tl.log(den))

@triton.jit
def _merge(PART,LSEPART,OUT,LSE,H:tl.constexpr,R:tl.constexpr,D:tl.constexpr,BD:tl.constexpr,S:tl.constexpr,BS:tl.constexpr):
    group=tl.program_id(0);head=tl.program_id(1);row=tl.program_id(2)
    ss=tl.arange(0,BS);dd=tl.arange(0,BD)
    z=tl.load(LSEPART+((group*H+head)*S+ss)*32+row,ss<S,-float('inf'))
    maximum=tl.max(z,0);weight=tl.exp(z-maximum);den=tl.sum(weight,0)
    value=tl.load(PART+(((group*H+head)*S+ss[:,None])*32+row)*BD+dd[None,:],(ss[:,None]<S)&(dd[None,:]<D),0.)
    result=tl.sum(value*weight[:,None],0)/den
    rr=group*32+row
    tl.store(OUT+head*R*D+rr*D+dd,result,(rr<R)&(dd<D))
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


def weighted_attention(q,k,v,indices,log_weights,specials,scale,key_grid,regions,emit_evidence=True,evidence_layout=None,parts=8,warps=4,stages=3,block_keys=64,static_probes=False,maxnreg=None,kernel_meta=None):
    h,r,d=q.shape;g,a=indices.shape
    if g!=triton.cdiv(r,32):raise ValueError('Invalid query group count')
    # Stable grouping changes physical scheduling, never logical history IDs.
    bank=key_grid.masked_fill(indices<0,regions)
    order=bank.argsort(dim=1,stable=True)
    sorted_ids=indices.gather(1,order).contiguous()
    sorted_bias=log_weights.gather(1,order).contiguous()
    counts=torch.zeros((g,regions+1),device=q.device,dtype=torch.int32)
    counts.scatter_add_(1,bank.long(),torch.ones_like(indices,dtype=torch.int32))
    offsets=torch.nn.functional.pad(counts[:,:regions].cumsum(1,dtype=torch.int32),(1,0)).contiguous()
    order=order.int().contiguous()
    if emit_evidence and evidence_layout is not None:
        evidence_layout['selected_grid_order']=order
        evidence_layout['selected_grid_offsets']=offsets
    output=torch.empty_like(q);lse=torch.empty((h,r),device=q.device,dtype=torch.float32)
    probes=torch.full((g,h,4,a),-torch.inf,device=q.device) if emit_evidence else lse
    logs=torch.empty((g,h,2,regions,32),device=q.device,dtype=torch.float32)
    summary=torch.empty((g,2,regions),device=q.device,dtype=torch.float32)
    bd=max(16,triton.next_power_of_2(d))
    partial=torch.empty((g,h,parts,32,bd),device=q.device,dtype=torch.float32)
    plse=torch.empty((g,h,parts,32),device=q.device,dtype=torch.float32)
    rawgrid=torch.empty((g,h,4,regions),device=q.device,dtype=torch.float32) if emit_evidence else lse
    if emit_evidence and evidence_layout is not None:evidence_layout['grid_raw_logsum']=rawgrid
    compiled=_attention[(g,h,parts)](q,k,v,sorted_ids,sorted_bias,order,offsets,specials,partial,plse,probes,logs,rawgrid,h,r,a,len(specials),d,regions,parts,triton.cdiv(regions,parts),q.stride(0),k.stride(0),scale,max(16,triton.next_power_of_2(d)),emit_evidence,block_keys,static_probes,num_warps=warps,num_stages=stages,maxnreg=maxnreg)
    if kernel_meta is not None:kernel_meta.update(registers=compiled.n_regs,spills=compiled.n_spills,shared_bytes=compiled.metadata.shared)
    _merge[(g,h,32)](partial,plse,output,lse,h,r,d,bd,parts,triton.next_power_of_2(parts),num_warps=4)
    _reduce[(g,2,triton.cdiv(regions,8))](logs,lse,summary,h,r,regions,8,triton.next_power_of_2(h),num_warps=4)
    return output,summary,probes if emit_evidence else None
