"""Fused normalizations of the two ORIGINAL coarse observations."""
import torch,triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def _m5c(GRID,SPECIAL,SIZE,LOGSIZE,COARSE,LZ,H:tl.constexpr,R:tl.constexpr,S:tl.constexpr,GG:tl.constexpr,GH:tl.constexpr,GP:tl.constexpr,SG:tl.constexpr,SH:tl.constexpr,SP:tl.constexpr,BR:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0);p=row%4;h=(row//4)%H;g=row//(H*4);r=tl.arange(0,BR);s=tl.arange(0,BS)
    base=tl.load(GRID+g*GG+h*GH+p*GP+r,r<R,-float('inf'));n=tl.load(SIZE+r,r<R,1.);ln=tl.load(LOGSIZE+r,r<R,0.)
    patch=base+ln;special=tl.load(SPECIAL+g*SG+h*SH+p*SP+s,s<S,-float('inf'))
    pm=tl.max(patch,0);sm=tl.max(special,0);safe=tl.where(sm==-float('inf'),0.,sm)
    pl=pm+tl.log(tl.sum(tl.exp(patch-pm),0));sl=sm+tl.log(tl.sum(tl.exp(special-safe),0))
    z=tl.maximum(pl,sl)+libdevice.log1p(tl.exp(-tl.abs(pl-sl)))
    tl.store(LZ+row,z);tl.store(COARSE+row*R+r,tl.exp(base-z)*n,r<R)

@triton.jit
def _patch_stats(DOT,LOGSIZE,OUT,H:tl.constexpr,G:tl.constexpr,R:tl.constexpr,SCALE:tl.constexpr,BR:tl.constexpr):
    row=tl.program_id(0);p=row%4;h=(row//4)%H;g=row//(H*4);r=tl.arange(0,BR)
    value=tl.load(DOT+((h*G+g)*4+p)*R+r,r<R,0.).to(tl.float32)*SCALE+tl.load(LOGSIZE+r,r<R,0.)
    value=tl.where(r<R,value,-float('inf'));m=tl.max(value,0);total=tl.sum(tl.exp(value-m),0)
    tl.store(OUT+row*2,m);tl.store(OUT+row*2+1,total)

@triton.jit
def _patch_mean(DOT,LOGSIZE,STATS,PREV,REAL,PROB,DEMAND,H:tl.constexpr,G:tl.constexpr,R:tl.constexpr,SCALE:tl.constexpr,BH:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);r=tl.program_id(1)*B+tl.arange(0,B);hp=tl.arange(0,BH*4);h=hp//4;p=hp%4
    z=tl.load(DOT+((h[:,None]*G+g)*4+p[:,None])*R+r[None,:],(h[:,None]<H)&(r[None,:]<R),0.).to(tl.float32)*SCALE+tl.load(LOGSIZE+r,r<R,0.)[None,:]
    row=(g*H+h)*4+p;m=tl.load(STATS+row*2,h<H,0.);s=tl.load(STATS+row*2+1,h<H,1.)
    value=tl.exp(z-m[:,None])/s[:,None];prob=tl.sum(tl.where(h[:,None]<H,value,0.),0)/(H*4)
    previous=tl.load(PREV+g*R+r,r<R,0.).to(tl.float32);real=tl.load(REAL+g*R+r,r<R,0.).to(tl.float32)
    beta=tl.where(previous>1.e-12,tl.minimum(tl.div_rn(prob,tl.maximum(previous,1.e-12)),1.),0.)
    demand=tl.maximum(prob-beta*real,0.)
    tl.store(PROB+g*R+r,prob,r<R);tl.store(DEMAND+g*R+r,demand,r<R)

def m5c(grid,special,sizes,log_sizes):
    g,h,_,r=grid.shape;s=special.shape[-1]
    coarse=torch.empty((g,h,4,r),device=grid.device);lz=torch.empty((g,h,4,1),device=grid.device)
    _m5c[(g*h*4,)](grid,special,sizes,log_sizes,coarse,lz,h,r,s,*grid.stride()[:3],*special.stride()[:3],triton.next_power_of_2(r),triton.next_power_of_2(s),num_warps=4,enable_fp_fusion=False)
    return coarse,lz

def patch(dot,log_sizes,previous,real,scale):
    h,g,_,r=dot.shape;dot=dot.contiguous()
    stats=torch.empty((g,h,4,2),device=dot.device);prob=torch.empty((g,r),device=dot.device);demand=torch.empty_like(prob)
    _patch_stats[(g*h*4,)](dot,log_sizes,stats,h,g,r,scale,triton.next_power_of_2(r),num_warps=4,enable_fp_fusion=False)
    _patch_mean[(g,triton.cdiv(r,8))](dot,log_sizes,stats,previous,real,prob,demand,h,g,r,scale,triton.next_power_of_2(h),8,num_warps=4,enable_fp_fusion=False)
    return prob,demand
