"""Masked stores replace dynamic boolean-index compaction; no arithmetic change."""
import triton
import triton.language as tl

@triton.jit
def _assignment(TABLE, ASSIGNED, COST, NEW, COSTS, N:tl.constexpr, COUNT:tl.constexpr, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    token=tl.load(TABLE+i,i<COUNT,N)
    valid=(i<COUNT)&(token<N)
    label=tl.load(ASSIGNED+i,valid,0)
    cost=tl.load(COST+i,valid,0.)
    tl.store(NEW+token,label,valid)
    tl.store(COSTS+token,cost,valid)

@triton.jit
def _repair(TARGET, CANDIDATE, MISSING, NEW, COUNT:tl.constexpr, B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    live=(i<COUNT)&tl.load(MISSING+i,i<COUNT,False)
    target=tl.load(TARGET+i,live,0)
    cluster=tl.load(CANDIDATE+i,live,0)
    tl.store(NEW+target,cluster,live)

def assignment(table,assigned,cost,new,costs,n):
    _assignment[(triton.cdiv(table.numel(),256),)](table,assigned,cost,new,costs,n,table.numel(),256)

def repair(target,candidate,missing,new):
    _repair[(triton.cdiv(target.numel(),256),)](target,candidate,missing,new,target.numel(),256)
