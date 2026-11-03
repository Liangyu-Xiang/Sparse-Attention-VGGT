"""Four contribution losses from uniform background + sparse history residuals.

No pairwise Value distances, nearest-history ownership, or mass transfer.
"""
import torch,triton
import triton.language as tl
from .reuse_drop_moment import prepare as prepare_bins


def prepare(z,labels,population):
 n,d=z.shape;f=len(population);width=triton.next_power_of_2(int(population.max()))
 order=labels.argsort(stable=True);starts=population.long().cumsum(0)-population.long()
 rank=torch.empty(n,device=z.device,dtype=torch.int32)
 rank[order]=(torch.arange(n,device=z.device)-starts[labels[order]]).int()
 members=torch.full((f,width),-1,device=z.device,dtype=torch.int32)
 members[labels,rank.long()]=torch.arange(n,device=z.device,dtype=torch.int32)
 sums=torch.zeros((f,d),device=z.device).index_add_(0,labels,z)
 return dict(members=members,sums=sums,width=width,bins=prepare_bins(population))

@triton.jit
def _moment(IDS,PROB,BACK,Z,PART,N:tl.constexpr,A:tl.constexpr,D:tl.constexpr,T:tl.constexpr,B:tl.constexpr,BD:tl.constexpr):
 g=tl.program_id(0);tile=tl.program_id(1);j=tile*B+tl.arange(0,B);d=tl.arange(0,BD)
 key=tl.load(IDS+g*A+j,j<A,-1);valid=(j<A)&(key>=0)
 p=tl.load(PROB+g*A+j,j<A,0.);b=tl.load(BACK+g);delta=tl.where(valid,p-b,0.)
 z=tl.load(Z+tl.maximum(key,0)[:,None]*D+d[None,:],valid[:,None]&(d[None,:]<D),0.)
 value=tl.sum(delta[:,None]*z,0)
 tl.store(PART+(g*T+tile)*D+d,value,d<D)

def moment(ids,probability,background,z):
 g,a=ids.shape;d=z.shape[1];tiles=triton.cdiv(a,128);partial=torch.empty((g,tiles,d),device=z.device)
 _moment[(g,tiles)](ids,probability,background,z,partial,len(z),a,d,tiles,128,triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
 return partial.sum(1)+background[:,None]*z.sum(0)

@triton.jit(do_not_specialize=['COUNT'])
def _states(POS,PROB,BACK,Z,SUM,MEM,REP,POP,ALLOW,O0,CLUSTERS,ERROR,COST,QUAL,
 N:tl.constexpr,A:tl.constexpr,F:tl.constexpr,D:tl.constexpr,W:tl.constexpr,COUNT,C:tl.constexpr,M:tl.constexpr,BD:tl.constexpr,PTR:tl.constexpr):
 if PTR:COUNT=tl.load(COUNT)
 g=tl.program_id(0);ci=tl.program_id(1)*C+tl.arange(0,C);m=tl.arange(0,M);d=tl.arange(0,BD)
 c=tl.load(CLUSTERS+ci,ci<COUNT,0);pop=tl.load(POP+c,ci<COUNT,0.)
 member=tl.load(MEM+c[:,None]*W+m[None,:],ci[:,None]<COUNT,-1);valid=(ci[:,None]<COUNT)&(member>=0)
 pos=tl.load(POS+g*(N+1)+tl.maximum(member,0),valid,-1);seen=valid&(pos>=0)
 b=tl.load(BACK+g);p=tl.load(PROB+g*A+tl.maximum(pos,0),seen,b);delta=tl.where(seen,p-b,0.)
 pi=pop*b+tl.sum(delta,1)
 rep=tl.load(REP+c,ci<COUNT,0);rep_pos=tl.load(POS+g*(N+1)+rep,ci<COUNT,-1);rep_seen=rep_pos>=0
 rep_p=tl.load(PROB+g*A+tl.maximum(rep_pos,0),(ci<COUNT)&rep_seen,b)
 qualified=seen&((p>tl.div_rn(pi,tl.maximum(pop,1.))[:,None])|(member==rep[:,None]))
 tl.store(QUAL+g*N+member,qualified,valid)
 values=tl.load(Z+tl.maximum(member,0)[:,:,None]*D+d[None,None,:],seen[:,:,None]&(d[None,None,:]<D),0.)
 sums=tl.load(SUM+c[:,None]*D+d[None,:],(ci[:,None]<COUNT)&(d[None,:]<D),0.)
 target=b*sums+tl.sum(delta[:,:,None]*values,1)
 hm=tl.sum(tl.where(qualified,p,0.),1);hv=tl.sum(tl.where(qualified,p,0.)[:,:,None]*values,1)
 rv=tl.load(Z+rep[:,None]*D+d[None,:],(ci[:,None]<COUNT)&(d[None,:]<D),0.)
 o=tl.load(O0+g*D+d,d<D,0.)
 drop=-target+pi[:,None]*o[None,:]
 rep_res=pop[:,None]*rep_p[:,None]*rv-target-(pop*rep_p-pi)[:,None]*o[None,:]
 hist=hv-target-(hm-pi)[:,None]*o[None,:]
 de=tl.sum(drop*drop,1);re=tl.sum(rep_res*rep_res,1);he=tl.sum(hist*hist,1)
 amount=tl.sum(qualified.to(tl.int32),1);allowed=tl.load(ALLOW+g*F+c,ci<COUNT,False)
 base=(g*F+c)*4
 tl.store(ERROR+base,de,ci<COUNT);tl.store(ERROR+base+1,re,ci<COUNT)
 tl.store(ERROR+base+2,tl.where(amount>0,he,float('inf')),ci<COUNT);tl.store(ERROR+base+3,tl.where(allowed,0.,float('inf')),ci<COUNT)
 tl.store(COST+base,0.,ci<COUNT);tl.store(COST+base+1,1.,ci<COUNT);tl.store(COST+base+2,amount.to(tl.float32),ci<COUNT);tl.store(COST+base+3,pop,ci<COUNT)

def states(ids,probability,background,z,reps,labels,population,allowed,o0,geometry):
 g,a=ids.shape;n,d=z.shape;f=len(reps)
 positions=torch.full((g,n+1),-1,device=z.device,dtype=torch.int32)
 positions.scatter_(1,ids.long().masked_fill(ids<0,n),torch.arange(a,device=z.device,dtype=torch.int32)[None].expand(g,-1).masked_fill(ids<0,-1))
 errors=torch.empty((g,f,4),device=z.device);costs=torch.empty_like(errors);qualified=torch.empty((g,n),device=z.device,dtype=torch.bool)
 allowed=allowed.contiguous()
 for bin_index,(size,clusters) in enumerate(geometry['bins']):
  c=min(16,max(1,128//size))
  _states[(g,triton.cdiv(len(clusters),c))](positions,probability,background,z,geometry['sums'],geometry['members'],reps,population,allowed,o0,clusters,errors,costs,qualified,n,a,f,d,geometry['width'],geometry['bin_counts'][bin_index] if 'bin_counts' in geometry else len(clusters),c,size,triton.next_power_of_2(d),'bin_counts' in geometry,num_warps=8 if size>=16 else 4,enable_fp_fusion=False)
 return errors,costs,qualified

@triton.jit
def _pack(SEL,OFFSET,COUNT,MEM,REP,POP,QUAL,IDS,WEIGHT,F:tl.constexpr,N:tl.constexpr,A:tl.constexpr,W:tl.constexpr,M:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);c=tl.program_id(1)*B+tl.arange(0,B);m=tl.arange(0,M)
 state=tl.load(SEL+g*F+c,c<F,0);offset=tl.load(OFFSET+g*F+c,c<F,0);count=tl.load(COUNT+g*F+c,c<F,0)
 member=tl.load(MEM+c[:,None]*W+m[None,:],c[:,None]<F,-1);valid=(c[:,None]<F)&(member>=0)
 qual=tl.load(QUAL+g*N+tl.maximum(member,0),valid,False)
 history=valid&qual;rank=tl.cumsum(history.to(tl.int32),1)-1
 emit=valid&((state[:,None]==3)|((state[:,None]==2)&history))
 rep=tl.load(REP+c,c<F,0);key=tl.where(state[:,None]==1,rep[:,None],member)
 emit=emit|((c[:,None]<F)&(state[:,None]==1)&(m[None,:]==0))
 pos=offset[:,None]+tl.where(state[:,None]==2,rank,m[None,:])
 population=tl.load(POP+c,c<F,1.)
 weight=tl.where(state[:,None]==1,population[:,None],1.)
 tl.store(IDS+g*A+pos,key,emit&(pos<A));tl.store(WEIGHT+g*A+pos,weight,emit&(pos<A))

@triton.jit(do_not_specialize=['NCL'])
def _pack_binned(SEL,OFFSET,COUNT,MEM,REP,POP,QUAL,IDS,WEIGHT,CLUSTERS,NCL,F:tl.constexpr,N:tl.constexpr,A:tl.constexpr,W:tl.constexpr,M:tl.constexpr,B:tl.constexpr,PTR:tl.constexpr):
 if PTR:NCL=tl.load(NCL)
 g=tl.program_id(0);ci=tl.program_id(1)*B+tl.arange(0,B);c=tl.load(CLUSTERS+ci,ci<NCL,0);m=tl.arange(0,M)
 state=tl.load(SEL+g*F+c,ci<NCL,0);offset=tl.load(OFFSET+g*F+c,ci<NCL,0);count=tl.load(COUNT+g*F+c,ci<NCL,0)
 member=tl.load(MEM+c[:,None]*W+m[None,:],c[:,None]<F,-1);valid=(c[:,None]<F)&(member>=0)
 qual=tl.load(QUAL+g*N+tl.maximum(member,0),valid,False)
 history=valid&qual;rank=tl.cumsum(history.to(tl.int32),1)-1
 emit=valid&((state[:,None]==3)|((state[:,None]==2)&history))
 rep=tl.load(REP+c,ci<NCL,0);key=tl.where(state[:,None]==1,rep[:,None],member)
 emit=emit|((c[:,None]<F)&(state[:,None]==1)&(m[None,:]==0))
 pos=offset[:,None]+tl.where(state[:,None]==2,rank,m[None,:])
 population=tl.load(POP+c,ci<NCL,1.)
 weight=tl.where(state[:,None]==1,population[:,None],1.)
 tl.store(IDS+g*A+pos,key,emit&(pos<A));tl.store(WEIGHT+g*A+pos,weight,emit&(pos<A))

def pack_reference(selected,costs,reps,population,qualified,geometry,budget):
 g,f=selected.shape;n=qualified.shape[1];count=costs.gather(2,selected[:,:,None]).squeeze(-1).int()
 prefix=count.cumsum(1,dtype=torch.int32);offset=prefix-count
 ids=torch.full((g,budget),-1,device=selected.device,dtype=torch.int32);weights=torch.zeros((g,budget),device=selected.device)
 w=geometry['width'];b=min(8,max(1,128//w))
 _pack[(g,triton.cdiv(f,b))](selected,offset,count,geometry['members'],reps,population,qualified,ids,weights,f,n,budget,w,w,b,num_warps=4)
 return ids,weights,prefix[:,-1]

def pack(selected,costs,reps,population,qualified,geometry,budget):
 g,f=selected.shape;n=qualified.shape[1];count=costs.gather(2,selected[:,:,None]).squeeze(-1).int()
 prefix=count.cumsum(1,dtype=torch.int32);offset=prefix-count
 ids=torch.full((g,budget),-1,device=selected.device,dtype=torch.int32);weights=torch.zeros((g,budget),device=selected.device)
 for bin_index,(size,clusters) in enumerate(geometry['bins']):
  b=min(16,max(1,128//size))
  _pack_binned[(g,triton.cdiv(len(clusters),b))](selected,offset,count,geometry['members'],reps,population,qualified,ids,weights,clusters,geometry['bin_counts'][bin_index] if 'bin_counts' in geometry else len(clusters),f,n,budget,geometry['width'],size,b,'bin_counts' in geometry,num_warps=4)
 return ids,weights,prefix[:,-1]
