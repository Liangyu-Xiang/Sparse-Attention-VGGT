"""Uniform-CTA branch: sampled and ordinary AV groups share one launch.

Fixed interval32*probe_group_stride for one Probe,all selected Keys,Head mean
AFTER per-Head final normalization. No extra QK; only sampled groups export.
"""
import torch,triton
import triton.language as tl
from contextlib import nullcontext
from .reuse_direct_sparse_attention import _attention as _body,_inline_mean,_inline_log_mean

@triton.jit
def _branch(Q,K,V,IDS,SPECIAL,BIAS,POS,OUT,LSE,TILE_P,TILE_LSE,
 H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,
 QH:tl.constexpr,KH:tl.constexpr,VH:tl.constexpr,SCALE:tl.constexpr,BD:tl.constexpr,BK:tl.constexpr,
 HAS_BIAS:tl.constexpr,P:tl.constexpr,T:tl.constexpr,FAST:tl.constexpr,STRIDE:tl.constexpr,GSIZE:tl.constexpr,BM:tl.constexpr):
 support_g=tl.program_id(0)//(GSIZE//BM)
 first_probe=tl.load(POS+support_g*P)
 if support_g%STRIDE==0 and first_probe//BM==tl.program_id(0):
  _body(Q,K,V,IDS,SPECIAL,BIAS,POS,OUT,LSE,TILE_P,TILE_LSE,H,R,N,A,S,D,QH,KH,VH,SCALE,BD,BK,HAS_BIAS,True,P,T,FAST,STRIDE,0,GSIZE,BM)
 else:
  _body(Q,K,V,IDS,SPECIAL,BIAS,POS,OUT,LSE,TILE_P,TILE_LSE,H,R,N,A,S,D,QH,KH,VH,SCALE,BD,BK,HAS_BIAS,False,P,T,0,STRIDE,0,GSIZE,BM)


def validate_support_indices(ids, specials, keys):
 """Explicit diagnostic guard. Production selector already emits unique supports.

 This synchronizes to validate contents; callers must keep it inside diagnostic
 timing if enabled. It is never silently added to performance measurements.
 """
 if ids.ndim!=2 or specials.ndim!=1:raise ValueError('Invalid support dimensions')
 if bool(((ids < -1) | (ids >= keys)).any()):raise ValueError('Out-of-range support index')
 if bool(((specials < 0) | (specials >= keys)).any()):raise ValueError('Out-of-range special index')
 bank=torch.cat((specials.expand(len(ids),-1),ids),1).sort(1).values
 if bool(((bank[:,1:]==bank[:,:-1]) & (bank[:,1:]>=0)).any()):raise ValueError('Duplicate support or special Key')


def direct_sparse_attention_sampled(q,k,v,sparse_key_ids,special_key_ids,scale,logit_bias=None,probe_positions=None,*,probe_count=1,probe_group_stride=8,query_group_size=32,debiased_log=False,block_keys=128,num_warps=4,num_stages=2,scope=None,block_queries=32,split_export=False):
 if split_export and not debiased_log:raise ValueError('Split export requires log history')
 scope=scope or (lambda name:nullcontext())
 h,r,d=q.shape;g,a=sparse_key_ids.shape;s=len(special_key_ids)
 if query_group_size not in (32,128,256) or g!=triton.cdiv(r,query_group_size):raise ValueError('Invalid shared query group size')
 if query_group_size!=32 and probe_count!=1:raise ValueError('Larger groups use one actual history probe')
 if probe_count not in (1,4) or not isinstance(probe_group_stride,int) or probe_group_stride<1:raise ValueError('One/four probes and positive integer stride required')
 if probe_positions is None:probe_positions=(torch.arange(g,device=q.device)[:,None]*query_group_size+((torch.arange(probe_count,device=q.device)*2+1)*(query_group_size//8))[None]).clamp_max(r-1).int().contiguous()
 p=probe_positions.shape[1]
 if p not in (1,4):raise ValueError('One/four probes required')
 for t in (q,k,v):
  if t.stride(-1)!=1 or t.stride(-2)!=d:raise ValueError('Contiguous head rows required')
 if block_queries not in (32,64,128) or query_group_size%block_queries:raise ValueError('Query tile must divide the support group')
 if p!=1 and block_queries!=32:raise ValueError('Larger tiles require one actual history probe')
 sampled=triton.cdiv(g,probe_group_stride);tiles=triton.cdiv(s+a,block_keys)
 out=torch.empty((h,r,d),device=q.device,dtype=q.dtype);lse=torch.empty((h,r),device=q.device,dtype=torch.float32)
 local=torch.empty((sampled,h,p,s+a),device=q.device,dtype=torch.float16);shift=torch.empty((sampled,h,p,tiles),device=q.device,dtype=torch.float32)
 with scope('sampled_direct_av'):
  compiled=_branch[(triton.cdiv(r,block_queries),h)](q,k,v,sparse_key_ids,special_key_ids,logit_bias if logit_bias is not None else lse,probe_positions,out,lse,local,shift,h,r,k.shape[1],a,s,d,q.stride(0),k.stride(0),v.stride(0),scale,max(16,triton.next_power_of_2(d)),block_keys,logit_bias is not None,p,tiles,4 if debiased_log else 3,probe_group_stride,query_group_size,block_queries,num_warps=num_warps,num_stages=num_stages)
 probability=torch.empty((sampled,p,s+a),device=q.device,dtype=torch.float32)
 with scope('sampled_probability_export'):
  if debiased_log:_inline_log_mean[(tiles,sampled,p)](local,shift,lse,probe_positions,probability,h,r,a,s,p,tiles,block_keys,triton.next_power_of_2(h),True,probe_group_stride,sampled if split_export else 0,num_warps=4)
  else:_inline_mean[(tiles,sampled,p)](local,shift,lse,probe_positions,probability,h,r,a,s,p,tiles,block_keys,triton.next_power_of_2(h),True,probe_group_stride,num_warps=4)
 if split_export:
  flat=probability.view(-1);cut=sampled*p*s
  special_probability=flat[:cut].view(sampled,p,s);patch_probability=flat[cut:].view(sampled,p,a)
 else:special_probability=probability[:,:,:s];patch_probability=probability[:,:,s:]
 prefix='debiased_log_' if debiased_log else ''
 evidence={prefix+'special_probability':special_probability,prefix+'patch_probability':patch_probability,'lse':lse,'probe_positions':probe_positions[::probe_group_stride],'scratch_bytes':local.numel()*local.element_size()+shift.numel()*shift.element_size(),'compiler_resources':dict(registers=getattr(compiled,'n_regs',None),spills=getattr(compiled,'n_spills',None),shared_bytes=getattr(compiled.metadata,'shared',None))}
 return out,evidence
