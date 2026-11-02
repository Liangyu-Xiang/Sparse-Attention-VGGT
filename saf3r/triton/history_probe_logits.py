"""Only four current probes x executed keys; no Dense QK materialization."""
import torch
import triton
import triton.language as tl

@triton.jit
def _logits(Q,K,IDS,POS,OUT,N:tl.constexpr,D:tl.constexpr,H:tl.constexpr,B:tl.constexpr,G:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr,T:tl.constexpr):
 tile=tl.program_id(0);group=tl.program_id(1);head=tl.program_id(2)
 rows=tl.arange(0,16);cols=tile*T+tl.arange(0,T);dd=tl.arange(0,BD)
 qi=tl.load(POS+group*4+rows,rows<4,0)
 ki=tl.load(IDS+group*B+cols,cols<B,-1)
 q=tl.load(Q+head*N*D+qi[:,None]*D+dd[None,:],(rows[:,None]<4)&(dd[None,:]<D),0.)
 k=tl.load(K+head*N*D+tl.maximum(ki,0)[None,:]*D+dd[:,None],(ki[None,:]>=0)&(cols[None,:]<B)&(dd[:,None]<D),0.)
 raw=tl.dot(q,k,allow_tf32=False)*SCALE
 raw=tl.where(ki[None,:]>=0,raw,-float('inf'))
 tl.store(OUT+((group*H+head)*4+rows[:,None])*B+cols[None,:],raw,(rows[:,None]<4)&(cols[None,:]<B))

def selected_probe_logits(q,k,actual_ids,probe_positions,scale):
 g,b=actual_ids.shape;h,n,d=q.shape
 if q.is_cuda:
  out=torch.empty((g,h,4,b),device=q.device,dtype=torch.float32)
  _logits[(triton.cdiv(b,64),g,h)](q,k,actual_ids,probe_positions,out,n,d,h,b,g,scale,triton.next_power_of_2(d),64,num_warps=4)
  return out
 rows=[]
 for group in range(g):
  raw=q[:,probe_positions[group]].float()@k[:,actual_ids[group].clamp_min(0).long()].float().transpose(-1,-2)*scale
  rows.append(raw.masked_fill(actual_ids[group][None,None]<0,-torch.inf))
 return torch.stack(rows)
