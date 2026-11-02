"""Previous layer prepares conditionals; current layer performs no old QK.

Equations and whole-grid pooling are preserved. Edge storage defaults to FP16
and is an explicit quantization change; computation/grid statistics are FP32.
"""
import torch
from .reuse_m5c import observe,mean_probes
from .history_dense_probability import EPS
from saf3r.triton.reuse_ready_history import predict,calibrate,store,logsumexp


def ensure_cache(s,q):
    h=q.shape[0];dtype=getattr(s,'ready_cache_dtype',torch.float16)
    packed=dtype==torch.float16 and not getattr(s,'ready_cache_plain',False);stride=(s.budget*14+7)//8 if packed else s.budget
    dtype=torch.uint8 if packed else dtype
    row_bytes=h*4*stride*torch.empty((),dtype=dtype).element_size()
    limit=getattr(s,'ready_gpu_limit_bytes',8*2**30)
    window=getattr(getattr(s,'config',None),'window_groups',64)
    resident=s.groups if s.groups*row_bytes<=limit else (limit//row_bytes//window)*window
    resident=min(resident,s.groups);s.ready_resident_groups=resident;s.ready_window=window
    s.ready_edges=torch.empty((resident,h,4,stride),device=q.device,dtype=dtype)
    s.ready_coarse=torch.empty((s.groups,h,4,s.regions),device=q.device,dtype=torch.float32)
    s.ready_uniform=torch.empty_like(s.ready_coarse)
    spill=s.groups-resident;host_shape=(spill,h,4,stride)
    host=getattr(s,'ready_host',None)
    if spill:
        if host is None or host.shape!=host_shape or host.dtype!=dtype:
            s.ready_host=torch.empty(host_shape,dtype=dtype,pin_memory=True)
        if getattr(s,'ready_copy_stream',None) is None:s.ready_copy_stream=torch.cuda.Stream()
        s.ready_stages=[torch.empty((window,h,4,stride),device=q.device,dtype=dtype) for _ in range(2)]
        for stage in s.ready_stages:stage.record_stream(s.ready_copy_stream)
    else:
        s.ready_host=None;s.ready_stages=[]
    s.ready_host_events={};s.ready_stage_free=[None,None];s.ready_uploaded={}


def _slot(s,start):return (start//s.ready_window)%2


def prefetch(s,start,g):
    if start<s.ready_resident_groups or start>=s.groups:return
    token=(s.previous_layer,start)
    if s.ready_uploaded.get(start,(None,None))[0]==token:return
    j=_slot(s,start);g=min(g,s.groups-start)
    with torch.cuda.stream(s.ready_copy_stream):
        if s.ready_stage_free[j] is not None:s.ready_copy_stream.wait_event(s.ready_stage_free[j])
        if start in s.ready_host_events:s.ready_copy_stream.wait_event(s.ready_host_events[start])
        local=start-s.ready_resident_groups
        s.ready_stages[j][:g].copy_(s.ready_host[local:local+g],non_blocking=True)
        event=torch.cuda.Event();event.record(s.ready_copy_stream)
    s.ready_uploaded[start]=(token,event)


def cache_read(s,start,g):
    if start<s.ready_resident_groups:return s.ready_edges[start:start+g]
    prefetch(s,start,g)
    torch.cuda.current_stream().wait_event(s.ready_uploaded[start][1])
    return s.ready_stages[_slot(s,start)][:g]


def cache_write(s,start,g):
    if start<s.ready_resident_groups:return s.ready_edges[start:start+g]
    j=_slot(s,start)
    if s.ready_stage_free[j] is not None:torch.cuda.current_stream().wait_event(s.ready_stage_free[j])
    return s.ready_stages[j][:g]


def cache_written(s,start,g):
    if start<s.ready_resident_groups:return
    j=_slot(s,start);produced=torch.cuda.Event();produced.record()
    with torch.cuda.stream(s.ready_copy_stream):
        s.ready_copy_stream.wait_event(produced)
        local=start-s.ready_resident_groups
        s.ready_host[local:local+g].copy_(s.ready_stages[j][:g],non_blocking=True)
        event=torch.cuda.Event();event.record(s.ready_copy_stream)
    s.ready_host_events[start]=event;s.ready_stage_free[j]=event
    s.ready_uploaded.pop(start,None)


def peek_cache(s,start,g):
    # Diagnostic read of the just-written values, on the producer stream.
    return s.ready_edges[start:start+g] if start<s.ready_resident_groups else s.ready_stages[_slot(s,start)][:g]


def build_prior(s,start,old_ids,obs):
    if s.ready_written.get(start)!=s.previous_layer:raise RuntimeError('Missing or stale preceding-layer cache')
    g=len(old_ids);current=obs['coarse_grid_probability'];previous=s.ready_coarse[start:start+g]
    ratio=(current+EPS)/(previous+EPS)
    confidence=torch.where(ratio<1,ratio,ratio.pow(-2)).clamp(0,1)
    base=current*(confidence*s.ready_uniform[start:start+g]+(1-confidence)/s.grid_sizes)
    prior=mean_probes(base,obs['weights'])[:,s.grid].contiguous()
    predict(cache_read(s,start,g),old_ids,s.grid,s.grid_sizes,current,previous,obs['weights'],prior,EPS)
    pi=torch.zeros((g,s.fine.base_count),device=prior.device).scatter_add_(1,s.fine.labels[None].expand(g,-1),prior)
    prefetch(s,start+g,min(s.ready_window,s.groups-start-g))
    expected=mean_probes(current,obs['weights']).sum(-1)
    s.m5c_prior_errors.append((prior.sum(-1)-expected).abs().max())
    return prior,pi


def save_evidence(s,q,k,start,ids,actual,obs,scale):
    # q/k/actual/scale kept for the common diagnostic hook signature, not used
    # for any additional matrix multiplication.
    valid=ids>=0;regions=s.grid[ids.long().clamp_min(0)]
    count=torch.zeros((len(ids),s.regions),device=ids.device).scatter_add_(1,regions,valid.float())
    remaining=s.grid_sizes[None]-count
    s.m5c_count_errors.append((remaining<0).any())
    raw=obs['ready_logits']
    missing=(obs['grid_logits']+remaining.clamp_min(1).log()[:,None,None,:]).masked_fill(remaining[:,None,None,:]==0,-torch.inf)
    logz=torch.logaddexp(torch.logaddexp(missing.logsumexp(-1),logsumexp(raw)),obs['special_logits'].logsumexp(-1))[:,:,:,None]
    unobserved=(obs['grid_logits']-logz).exp()
    corrected=calibrate(raw,logz,ids,s.grid,unobserved*remaining[:,None,None,:],regions)
    special=(obs['special_logits']-logz).exp()
    s.m5c_seed_errors.append((corrected.sum(-1)+special.sum(-1)-1).abs().max())
    end=start+len(ids)
    store(raw,logz,ids,s.grid,corrected,cache_write(s,start,len(ids)),EPS)
    cache_written(s,start,len(ids))
    s.ready_uniform[start:end].copy_(unobserved/corrected.clamp_min(EPS))
    s.ready_coarse[start:end].copy_(obs['coarse_grid_probability'])
    s.ready_written[start]=s.previous_layer+1


def release(s):
    s.ready_edges=None;s.ready_coarse=None;s.ready_uniform=None;s.ready_stages=[];s.ready_written={}


def trim(s):
    # The large persistent evidence bank shares the allocator with >1 GiB
    # transformer workspaces. Return fragmented inactive pages only under
    # pressure, rather than clearing the allocator every layer/window.
    free,_=torch.cuda.mem_get_info()
    inactive=torch.cuda.memory_reserved()-torch.cuda.memory_allocated()
    if free<3*2**30 and inactive>2**29:
        torch.cuda.empty_cache()
        s.ready_allocator_trims=getattr(s,'ready_allocator_trims',0)+1


def host_bytes(s):
    t=getattr(s,'ready_host',None)
    return 0 if t is None else t.numel()*t.element_size()


def device_bytes(s):
    return sum(t.numel()*t.element_size() for t in [s.ready_edges,s.ready_coarse,s.ready_uniform]+getattr(s,'ready_stages',[]) if t is not None)
