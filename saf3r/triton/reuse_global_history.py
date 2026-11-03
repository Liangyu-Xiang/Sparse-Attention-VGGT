"""Current-layer normalization then head/probe mean. Cross-layer cache is FP32."""
import torch,triton
import triton.language as tl

@triton.jit
def _store(RAW,Z,WEIGHT,OUT,IDS,H:tl.constexpr,P:tl.constexpr,A:tl.constexpr,BH:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B);hp=tl.arange(0,BH*P);h=hp//P;p=hp%P
 raw=tl.load(RAW+((g*H+h[:,None])*P+p[:,None])*A+col[None,:],(h[:,None]<H)&(col[None,:]<A),-float('inf'))
 z=tl.load(Z+(g*H+h)*P+p,h<H,0.);w=tl.load(WEIGHT+g*P+p)
 prob=tl.where(h[:,None]<H,tl.exp(raw-z[:,None])*w[:,None]/H,0.)
 valid=tl.load(IDS+g*A+col,col<A,-1)>=0
 tl.store(OUT+g*A+col,tl.where(valid,tl.sum(prob,0),0.),col<A)

def store(raw,logz,weights,ids):
 g,h,p,a=raw.shape;out=torch.empty((g,a),device=raw.device)
 _store[(g,triton.cdiv(a,128))](raw,logz.contiguous(),weights.contiguous(),out,ids,h,p,a,triton.next_power_of_2(h),128,num_warps=8)
 return out
