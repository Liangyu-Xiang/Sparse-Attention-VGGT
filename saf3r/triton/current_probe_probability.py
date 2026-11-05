"""Exact sampled-query head/probe-mean Dense probabilities, no HxQxK bank.

Pass1 online softmax LSE;pass2 streams heads then reduces probes. The extra QK
is2*P/group_size of full Dense QK;the normal sparse AV is not repeated.
"""
import torch,triton
import triton.language as tl

@triton.jit
def _lse(Q,K,Z,M:tl.constexpr,N:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
 block=tl.program_id(0);head=tl.program_id(1);row=block*BM+tl.arange(0,BM);d=tl.arange(0,BD);col=tl.arange(0,BK)
 q=tl.load(Q+head*QH+row[:,None]*D+d[None,:],(row[:,None]<M)&(d[None,:]<D),0.)
 maximum=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32)
 for start in range(tl.cdiv(N,BK)):
  key=start*BK+col;k=tl.load(K+head*KH+key[None,:]*D+d[:,None],(key[None,:]<N)&(d[:,None]<D),0.)
  scores=tl.where(key[None,:]<N,tl.dot(q,k,allow_tf32=False)*SCALE,-float('inf'))
  nextmax=tl.maximum(maximum,tl.max(scores,1));den=den*tl.exp(maximum-nextmax)+tl.sum(tl.exp(scores-nextmax[:,None]),1);maximum=nextmax
 tl.store(Z+head*M+row,maximum+tl.log(den),row<M)

@triton.jit
def _mean(Q,K,Z,PATCH,OUT,G:tl.constexpr,H:tl.constexpr,N:tl.constexpr,NP:tl.constexpr,D:tl.constexpr,P:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,SCALE:tl.constexpr,BG:tl.constexpr,BK:tl.constexpr,BD:tl.constexpr):
 # Fast group-block axis shares the same Key tile across nearby CTAs/L2.
 groups=tl.cdiv(G,BG);linear=tl.program_id(0);gb=linear%groups;tile=linear//groups
 r=tl.arange(0,BG*P);rows=gb*BG*P+r;d=tl.arange(0,BD);col=tile*BK+tl.arange(0,BK)
 key=tl.load(PATCH+col,col<NP,0);total=tl.zeros((BG*P,BK),tl.float32)
 for head in range(H):
  q=tl.load(Q+head*QH+rows[:,None]*D+d[None,:],(rows[:,None]<G*P)&(d[None,:]<D),0.)
  k=tl.load(K+head*KH+key[None,:]*D+d[:,None],(col[None,:]<NP)&(d[:,None]<D),0.)
  lse=tl.load(Z+head*G*P+rows,rows<G*P,0.)
  logits=tl.dot(q,k,allow_tf32=False)*SCALE
  total+=tl.exp(logits-lse[:,None])
 average=tl.sum(total.reshape((BG,P,BK)),1)/(H*P)
 group=gb*BG+tl.arange(0,BG)
 tl.store(OUT+group[:,None]*NP+col[None,:],average,(group[:,None]<G)&(col[None,:]<NP))

def probability(q,k,patches,group_size,probes,scale):
 h,n,d=q.shape;g=triton.cdiv(n,group_size)
 if probes not in (1,4):raise ValueError('Use1/4 observed queries per group')
 offsets=(torch.tensor([1] if probes==1 else [1,3,5,7],device=q.device)* (group_size//8))
 positions=(torch.arange(g,device=q.device)[:,None]*group_size+offsets).clamp_max(n-1)
 qp=q[:,positions.flatten()].contiguous();m=g*probes
 z=torch.empty((h,m),device=q.device,dtype=torch.float32);out=torch.empty((g,len(patches)),device=q.device,dtype=torch.float32)
 _lse[(triton.cdiv(m,32),h)](qp,k,z,m,n,d,qp.stride(0),k.stride(0),scale,32,128,triton.next_power_of_2(d),num_warps=4,num_stages=2)
 bg=16//probes
 _mean[(triton.cdiv(g,bg)*triton.cdiv(len(patches),128),)](qp,k,z,patches,out,g,h,n,len(patches),d,probes,qp.stride(0),k.stride(0),scale,bg,128,triton.next_power_of_2(d),num_warps=4,num_stages=2)
 return out,positions
