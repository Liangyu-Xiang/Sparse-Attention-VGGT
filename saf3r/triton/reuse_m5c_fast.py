"""Fuse historical selected-QK reconstruction, M5c correction and probe averaging."""
import torch,triton
import triton.language as tl

@triton.jit
def _predict(Q,K,IDS,ACTUAL,POS,GRID,SIZES,CURRENT,PREVIOUS,CORRECTED,LOGZ,WEIGHTS,EDGE,
 N:tl.constexpr,P:tl.constexpr,H:tl.constexpr,D:tl.constexpr,A:tl.constexpr,R:tl.constexpr,SCALE:tl.constexpr,EPS:tl.constexpr,BD:tl.constexpr,T:tl.constexpr):
 tile=tl.program_id(0);group=tl.program_id(1);head=tl.program_id(2)
 rows=tl.arange(0,16);cols=tile*T+tl.arange(0,T);dd=tl.arange(0,BD)
 actual=tl.load(ACTUAL+group*A+cols,cols<A,-1);patch=tl.load(IDS+group*A+cols,cols<A,-1);valid=(cols<A)&(patch>=0)
 qi=tl.load(POS+group*4+rows,rows<4,0)
 q=tl.load(Q+head*N*D+qi[:,None]*D+dd[None,:],(rows[:,None]<4)&(dd[None,:]<D),0.)
 k=tl.load(K+head*N*D+tl.maximum(actual,0)[None,:]*D+dd[:,None],valid[None,:]&(dd[:,None]<D),0.)
 raw=tl.dot(q,k,allow_tf32=False)*SCALE
 region=tl.load(GRID+tl.maximum(patch,0),valid,0);size=tl.load(SIZES+region,valid,1.)
 index=((group*H+head)*4+rows[:,None])*R+region[None,:];mask=(rows[:,None]<4)&valid[None,:]
 current=tl.load(CURRENT+index,mask,0.);previous=tl.load(PREVIOUS+index,mask,0.);corrected=tl.load(CORRECTED+index,mask,1.)
 ratio=(current+EPS)/(previous+EPS);confidence=tl.minimum(1.,tl.where(ratio<1.,ratio,1./(ratio*ratio)))
 logz=tl.load(LOGZ+(group*H+head)*4+rows,rows<4,0.)
 old_probability=tl.exp(raw-logz[:,None])
 predicted=current*(confidence*old_probability/tl.maximum(corrected,EPS)+(1.-confidence)/size[None,:])
 weight=tl.load(WEIGHTS+group*4+rows,rows<4,0.)
 edge=tl.sum(tl.where(mask,predicted*weight[:,None],0.),0)/H
 tl.atomic_add(EDGE+group*A+cols,edge,valid,sem='relaxed')

@triton.jit
def _scatter(EDGE,IDS,PRIOR,A:tl.constexpr,P:tl.constexpr,B:tl.constexpr):
 group=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B);key=tl.load(IDS+group*A+i,i<A,-1);value=tl.load(EDGE+group*A+i,i<A,0.)
 tl.store(PRIOR+group*P+key,value,(i<A)&(key>=0))

def predict(q,k,ids,actual,obs,old,grid,sizes,prior,scale,eps):
 g,a=ids.shape;h,n,d=q.shape;r=len(sizes);edge=torch.zeros((g,a),device=q.device)
 _predict[(triton.cdiv(a,64),g,h)](q,k,ids,actual,obs['positions'],grid,sizes,obs['coarse_grid_probability'].contiguous(),old['previous_coarse_grid_probability'].contiguous(),old['previous_corrected_grid_probability'].contiguous(),old['history_dense_log_normalizer'].contiguous(),obs['weights'].contiguous(),edge,n,len(grid),h,d,a,r,scale,eps,triton.next_power_of_2(d),64,num_warps=4,enable_fp_fusion=False)
 _scatter[(g,triton.cdiv(a,256))](edge,ids,prior,a,len(grid),256,num_warps=4)
 return edge
