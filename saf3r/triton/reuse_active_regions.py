"""Prune empty-grid loop iterations; keep exactly the same nonempty grids."""
import torch,triton
import triton.language as tl

@triton.jit
def _compact(OFF,PREFIX,ACTIVE,COUNTS,R:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);r=tl.arange(0,B)
    begin=tl.load(OFF+g*(R+1)+r,r<R,0);end=tl.load(OFF+g*(R+1)+r+1,r<R,0)
    pos=tl.load(PREFIX+g*R+r,r<R,0)-1;valid=(r<R)&(end>begin)
    tl.store(ACTIVE+g*R+pos,r,valid)
    count=tl.load(PREFIX+g*R+R-1);tl.store(COUNTS+g,count)

@triton.jit
def _bounds(ACTIVE,OFF,COUNTS,BOUNDS,R:tl.constexpr,S:tl.constexpr,PARTS:tl.constexpr,BP:tl.constexpr,BALANCED:tl.constexpr):
    g=tl.program_id(0);p=tl.arange(0,BP);count=tl.load(COUNTS+g);total=tl.load(OFF+g*(R+1)+R)
    lo=tl.full((BP,),0,tl.int32);hi=tl.full((BP,),count,tl.int32)
    target=tl.maximum((p*(total+S))//PARTS-S,0) if BALANCED else p*tl.cdiv(R,PARTS)
    for i in range(12):
        mid=(lo+hi)//2;region=tl.load(ACTIVE+g*R+mid,mid<count,R)
        value=tl.load(OFF+g*(R+1)+tl.minimum(region,R)) if BALANCED else region
        use=(mid<count)&(value<target)
        lo=tl.where(use,mid+1,lo);hi=tl.where(~use,mid,hi)
    result=tl.where(p==0,0,tl.where(p==PARTS,count,lo));tl.store(BOUNDS+g*(PARTS+1)+p,result,p<=PARTS)

@triton.jit
def _zero(OFF,LOGS,RAW,H:tl.constexpr,R:tl.constexpr,EVIDENCE:tl.constexpr,B:tl.constexpr):
    g=tl.program_id(0);h=tl.program_id(1);r=tl.program_id(2)*B+tl.arange(0,B);row=tl.arange(0,32)
    begin=tl.load(OFF+g*(R+1)+r,r<R,0);end=tl.load(OFF+g*(R+1)+r+1,r<R,0);empty=(r<R)&(begin==end)
    base=(((g*H+h)*2)*R+r[:,None])*32+row[None,:]
    tl.store(LOGS+base,-float('inf'),empty[:,None]);tl.store(LOGS+base+R*32,-float('inf'),empty[:,None])
    if EVIDENCE:
        p=tl.arange(0,4)
        tl.store(RAW+((g*H+h)*4+p[:,None])*R+r[None,:],-float('inf'),empty[None,:])

def prepare(offsets,heads,parts,specials,logs,raw,evidence,balanced=False):
    g,r1=offsets.shape;r=r1-1
    positive=(offsets[:,1:]-offsets[:,:-1])>0;prefix=positive.cumsum(1,dtype=torch.int32)
    active=torch.empty((g,r),device=offsets.device,dtype=torch.int32);counts=torch.empty(g,device=offsets.device,dtype=torch.int32);bounds=torch.empty((g,parts+1),device=offsets.device,dtype=torch.int32)
    _compact[(g,)](offsets,prefix,active,counts,r,triton.next_power_of_2(r),num_warps=4)
    _bounds[(g,)](active,offsets,counts,bounds,r,specials,parts,triton.next_power_of_2(parts+1),balanced,num_warps=4)
    _zero[(g,heads,triton.cdiv(r,8))](offsets,logs,raw,heads,r,evidence,8,num_warps=4)
    return active,bounds
