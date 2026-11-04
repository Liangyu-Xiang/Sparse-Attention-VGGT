"""Original AV arithmetic; expose final sparse normalizer and logical bias for history."""
import torch,triton
from .reuse_single_qk_grid import _attention,_reduce

def weighted_attention(q,k,v,indices,log_weights,specials,scale,key_grid,regions,emit_evidence=True,evidence_layout=None,stages=1):
    h,r,d=q.shape;g,a=indices.shape
    if g!=triton.cdiv(r,32):raise ValueError('Invalid query group count')
    # Stable grouping changes physical scheduling, never logical history IDs.
    bank=key_grid.masked_fill(indices<0,regions)
    order=bank.argsort(dim=1,stable=True)
    sorted_ids=indices.gather(1,order).contiguous()
    sorted_bias=log_weights.gather(1,order).contiguous()
    counts=torch.zeros((g,regions+1),device=q.device,dtype=torch.int32)
    counts.scatter_add_(1,bank.long(),torch.ones_like(indices,dtype=torch.int32))
    offsets=torch.nn.functional.pad(counts[:,:regions].cumsum(1,dtype=torch.int32),(1,0)).contiguous()
    order=order.int().contiguous()
    if emit_evidence and evidence_layout is not None:
        evidence_layout['selected_grid_order']=order
        evidence_layout['selected_grid_offsets']=offsets
    output=torch.empty_like(q);lse=torch.empty((h,r),device=q.device,dtype=torch.float32)
    probes=torch.full((g,h,4,a),-torch.inf,device=q.device) if emit_evidence else lse
    logs=torch.empty((g,h,2,regions,32),device=q.device,dtype=torch.float32)
    summary=torch.empty((g,2,regions),device=q.device,dtype=torch.float32)
    _attention[(g,h)](q,k,v,sorted_ids,sorted_bias,order,offsets,specials,output,lse,probes,logs,h,r,a,len(specials),d,regions,q.stride(0),k.stride(0),scale,max(16,triton.next_power_of_2(d)),emit_evidence,num_warps=4,num_stages=stages)
    _reduce[(g,2,triton.cdiv(regions,8))](logs,lse,summary,h,r,regions,8,triton.next_power_of_2(h),num_warps=4)
    if emit_evidence and evidence_layout is not None:
        evidence_layout['sparse_lse']=lse
        evidence_layout['sparse_log_weights']=log_weights
    return output,summary,probes if emit_evidence else None
