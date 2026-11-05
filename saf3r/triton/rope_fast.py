"""2-D RoPE retaining the reference's intermediate dtype roundings."""
import torch,triton
import triton.language as tl


@triton.jit
def _rope(X,POS,COS,SIN,Y,N:tl.constexpr,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr,
          S0:tl.constexpr,S1:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,BT:tl.constexpr,BD:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT);h=tl.program_id(1);d=tl.arange(0,BD)
    axis=d//(D//2);local=d%(D//2);partner=axis*(D//2)+(local+D//4)%(D//2)
    offset=(t//N)[:,None]*S0+h*S1+(t%N)[:,None]*S2
    val=tl.load(X+offset+d[None,:]*S3,(t[:,None]<T)&(d[None,:]<D),0.)
    rot=tl.load(X+offset+partner[None,:]*S3,(t[:,None]<T)&(d[None,:]<D),0.).to(tl.float32)
    rot=tl.where(local[None,:]<D//4,-rot,rot)
    position=tl.load(POS+t[:,None]*2+axis[None,:],(t[:,None]<T)&(d[None,:]<D),0)
    c=tl.load(COS+position*(D//2)+local[None,:],(t[:,None]<T)&(d[None,:]<D),0.).to(tl.float32)
    s=tl.load(SIN+position*(D//2)+local[None,:],(t[:,None]<T)&(d[None,:]<D),0.).to(tl.float32)
    a=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[val.to(tl.float32),c],dtype=tl.float32,is_pure=True,pack=1).to(val.dtype).to(tl.float32)
    b=tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[rot,s],dtype=tl.float32,is_pure=True,pack=1).to(val.dtype).to(tl.float32)
    result=tl.inline_asm_elementwise('add.rn.f32 $0, $1, $2;',constraints='=f,f,f',args=[a,b],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(Y+((t//N)[:,None]*H+h)*N*D+(t%N)[:,None]*D+d[None,:],result,(t[:,None]<T)&(d[None,:]<D))


def apply(tokens,positions,cos,sin):
    b,h,n,d=tokens.shape
    if d%4:raise ValueError('2-D RoPE head dimension must divide4')
    out=torch.empty((b,h,n,d),device=tokens.device,dtype=tokens.dtype)
    _rope[(triton.cdiv(b*n,32),h)](tokens,positions.contiguous(),cos,sin,out,n,b*n,h,d,*tokens.stride(),32,triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
    return out


def forward(self,tokens,positions):
    if not hasattr(self,'_fixed_max_position'):raise ValueError('Install fixed positions before fused RoPE')
    cos,sin=self._compute_frequency_components(tokens.shape[-1]//2,self._fixed_max_position,tokens.device,tokens.dtype)
    return apply(tokens,positions,cos,sin)
