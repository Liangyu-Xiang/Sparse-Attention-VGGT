"""Cluster counts -> prefix -> direct output; preserve ascending original Key order."""
import torch
import triton
import triton.language as tl

@triton.jit
def _counts(SEL,AMOUNT,POP,REMOVE,OUT,BAD,F:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);c=tl.program_id(1)*B+tl.arange(0,B);valid=c<F
    s=tl.load(SEL+g*F+c,valid,0);a=tl.load(AMOUNT+g*F+c,valid,0);p=tl.load(POP+c,valid,0).to(tl.int32)
    remove=tl.load(REMOVE+g*F+c,valid,False)
    count=tl.where(s==1,1,tl.where(s==5,a,tl.where(s==10,p,0)))
    invalid=((s==1)&remove)|~((s==0)|(s==1)|(s==5)|(s==10))
    tl.store(OUT+g*F+c,count,valid);tl.store(BAD+g*F+c,invalid,valid)

@triton.jit
def _write(SEL,REP,MEM,QUAL,PREFIX,IDS,ACTUAL,N:tl.constexpr,F:tl.constexpr,W:tl.constexpr,A:tl.constexpr,C:tl.constexpr):
    g=tl.program_id(0);c=tl.program_id(1)*C+tl.arange(0,C);j=tl.arange(0,W)
    s=tl.load(SEL+g*F+c,c<F,0);rep=tl.load(REP+c,c<F,-1)
    # Drop/Rep clusters do not read all member slots.
    expanded=(c[:,None]<F)&((s[:,None]==5)|(s[:,None]==10))
    member=tl.load(MEM+c[:,None]*W+j[None,:],expanded,-1)
    qual=tl.load(QUAL+g*N+tl.maximum(member,0),expanded&(member>=0)&(s[:,None]==5),False)
    keep=(member>=0)&((s[:,None]==10)|((s[:,None]==5)&qual))
    rank=tl.cumsum(keep.to(tl.int32),1)-1
    end=tl.load(PREFIX+g*F+c,c<F,0);before=tl.load(PREFIX+g*F+c-1,(c<F)&(c>0),0)
    pos=before[:,None]+rank
    tl.store(IDS+g*A+pos,member,keep&(pos<A))
    tl.store(IDS+g*A+before,rep,(c<F)&(s==1)&(before<A))
    emitted=tl.sum(keep.to(tl.int32),1)+(s==1).to(tl.int32)
    tl.store(ACTUAL+g*F+c,emitted,c<F)

def pack(selected,reps,labels,population,budget,geometry,context):
    g,f=selected.shape;n=len(labels)
    counts=torch.empty((g,f),device=selected.device,dtype=torch.int32);bad=torch.empty_like(counts,dtype=torch.bool)
    _counts[(g,triton.cdiv(f,256))](selected,context['eligible_count'],population,context['remove_rep'],counts,bad,f,256,num_warps=4)
    prefix=counts.cumsum(1,dtype=torch.int32);total=prefix[:,-1];actual=torch.empty_like(counts)
    ids=torch.full((g,budget),n,device=selected.device,dtype=torch.int32)
    _write[(g,triton.cdiv(f,8))](selected,reps,geometry['members'],context['qualified'],prefix,ids,actual,n,f,geometry['width'],budget,8,num_warps=4)
    # Original pack emits ascending token IDs. Preserve that order for the
    # executor's stable Grid sort and all downstream floating-point reductions.
    ids=ids.sort(dim=1).values;ids.masked_fill_(ids==n,-1)
    emitted=actual.sum(1)
    context['pack_error']=(total>budget)|(emitted!=total)|bad.any(1)
    return ids,ids>=0,dict(reuse_edges=(context['eligible_count']*(selected==5)).sum(1),representatives=(selected==1).sum(1),exact_edges=(population[None]*(selected==10)).sum(1),dropped_clusters=(selected==0).sum(1),counts=total,native_cost_count_error=(emitted-total).abs().float())
