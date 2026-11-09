import math
"""Export group-mean raw logits from formal sparse AV;no extra QK."""
import torch,triton
import triton.language as tl

@triton.jit
def _pool(Q,MASK,COUNT,OUT,N:tl.constexpr,G:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,BD:tl.constexpr):
 p=tl.program_id(0);h=tl.program_id(1);r=p*64+tl.arange(0,64);d=tl.arange(0,BD)
 valid=tl.load(MASK+r,r<N,False);x=tl.load(Q+h*QH+r[:,None]*D+d[None],(r[:,None]<N)&(d[None]<D),0.).to(tl.float32)
 count=tl.maximum(tl.load(COUNT+p),1);value=tl.sum(tl.where(valid[:,None],x,0.),0)/count
 tl.store(OUT+(h*G*4+p)*D+d,value,d<D)

def pool_queries(q,mask,counts):
 h,n,d=q.shape;g=len(counts);out=torch.empty(h,g,4,d,device=q.device,dtype=torch.float32)
 _pool[(g*4,h)](q,mask,counts,out,n,g,d,q.stride(0),triton.next_power_of_2(d),num_warps=4)
 return out

@triton.jit
def _biasmax(BIAS,REF,A:tl.constexpr,B:tl.constexpr):
 g=tl.program_id(0);a=tl.arange(0,B);x=tl.load(BIAS+g*A+a,a<A,-float('inf'));tl.store(REF+g,tl.maximum(tl.max(x,0),0.))

@triton.jit
def _attention(Q,K,V,IDS,SPECIAL,BIAS,MASK,COUNT,SHIFT,REFERENCE,OUT,MEAN,STATS,HISTORY_SUM,VALID_COUNTS,H:tl.constexpr,R:tl.constexpr,N:tl.constexpr,G:tl.constexpr,A:tl.constexpr,S:tl.constexpr,D:tl.constexpr,QH:tl.constexpr,KH:tl.constexpr,VH:tl.constexpr,OH:tl.constexpr,OR:tl.constexpr,SCALE:tl.constexpr,CAPTURE:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,DIRECT_EXPORT:tl.constexpr,SPLITS:tl.constexpr,FUSED_ACC:tl.constexpr,CACHE:tl.constexpr,EVICT:tl.constexpr,TRANSPOSE_QK:tl.constexpr,SKIP_PREFIX:tl.constexpr,UNROLL:tl.constexpr,WS:tl.constexpr,EXPORT_SUM:tl.constexpr,BASE2:tl.constexpr,PRE_SCALED_BIAS:tl.constexpr,ALIGNED_SUPPORT:tl.constexpr,FIXED_GAUGE:tl.constexpr,SEPARATE_PREFIX:tl.constexpr,SPLIT_TILES:tl.constexpr,TRIM_TAIL:tl.constexpr,ZERO_COPY:tl.constexpr,ORIGINAL_A:tl.constexpr,ORIGINAL_S:tl.constexpr,GRID_ORDER:tl.constexpr,ACTIVE_SPLITS:tl.constexpr,WARP_HISTORY:tl.constexpr):
 if GRID_ORDER==1:
  t=tl.program_id(0);h=tl.program_id(2);split=tl.program_id(1)
 elif GRID_ORDER==2:
  t=tl.program_id(1);h=tl.program_id(2);split=tl.program_id(0)
 elif GRID_ORDER==3:
  t=tl.program_id(1);h=tl.program_id(0);split=tl.program_id(2)
 else:
  t=tl.program_id(0);h=tl.program_id(1);split=tl.program_id(2)
 g=t//(256//BM);row=tl.arange(0,BM);rr=t*BM+row;dd=tl.arange(0,BD);kk=tl.arange(0,BK)
 q=tl.load(Q+h*QH+rr[:,None]*D+dd[None],(rr[:,None]<R)&(dd[None]<D),0.)
 maximum=tl.full((BM,),-float('inf'),tl.float32);den=tl.zeros((BM,),tl.float32);acc=tl.zeros((BM,BD),tl.float32)
 if CAPTURE or FIXED_GAUGE:
  qm=tl.load(MASK+rr,rr<R,False)
  parts=t*(BM//64)+tl.arange(0,BM//64)
  count=tl.maximum(tl.load(COUNT+parts),1)
  shift=tl.load(SHIFT+h*G*4+parts)
  if BASE2:shift*=1.4426950408889634
  if not DIRECT_EXPORT:reference=tl.load(REFERENCE+g)
  if EXPORT_SUM:history_sum=tl.zeros((BM//64,),tl.float32)
 tiles:tl.constexpr=(S+A+BK-1)//BK
 span:tl.constexpr=SPLIT_TILES if SPLIT_TILES else (tiles+SPLITS-1)//SPLITS
 for segment in tl.static_range(2 if SEPARATE_PREFIX else 1):
  if SEPARATE_PREFIX:
   section_begin=0 if segment==0 else S//BK
   section_end=S//BK if segment==0 else tiles
   begin=tl.maximum(split*span,section_begin)
   end=tl.minimum((split+1)*span,section_end)
  else:
   begin=split*span
   end=tl.minimum((split+1)*span,tiles)
  if TRIM_TAIL:
   full_end=end
   valid_end=(S+tl.load(VALID_COUNTS+g)+BK-1)//BK
   end=tl.minimum(end,valid_end)
  for tile in tl.range(begin,end,loop_unroll_factor=UNROLL,warp_specialize=WS):
   at=tile*BK+kk
   if ZERO_COPY:
    if segment==0:
     key=tl.load(SPECIAL+at,at<ORIGINAL_S,-1)
     bias=tl.full((BK,),1.,tl.float32)
    else:
     pos=at-S
     key=tl.load(IDS+g*ORIGINAL_A+pos,pos<ORIGINAL_A,-1)
     bias=tl.load(BIAS+g*ORIGINAL_A+pos,pos<ORIGINAL_A,1.)
    valid=key>=0
   elif ALIGNED_SUPPORT:
    key=tl.load(IDS+g*(S+A)+at,at<S+A,-1)
    valid=key>=0
    bias=tl.full((BK,),1. if FIXED_GAUGE else 0.,tl.float32) if SEPARATE_PREFIX and segment==0 else tl.load(BIAS+g*(S+A)+at,at<S+A,0.)
   else:
    key=tl.load(IDS+g*A+at-S,(at>=S)&(at<S+A),-1)
    special=tl.load(SPECIAL+at,at<S,0)
    key=tl.where(at<S,special,key)
    valid=(at<S+A)&(key>=0)
    bias=tl.load(BIAS+g*A+at-S,(at>=S)&valid,1. if FIXED_GAUGE else 0.)
   k=tl.load(K+h*KH+tl.maximum(key,0)[None,:]*D+dd[:,None],valid[None]&(dd[:,None]<D),0.,cache_modifier=CACHE,eviction_policy=EVICT)
   v=tl.load(V+h*VH+tl.maximum(key,0)[:,None]*D+dd[None],valid[:,None]&(dd[None]<D),0.,cache_modifier=CACHE,eviction_policy=EVICT)
   if TRANSPOSE_QK:
    raw=tl.trans(tl.dot(tl.trans(k),tl.trans(q),allow_tf32=False))*SCALE
   else:
    raw=tl.dot(q,k,allow_tf32=False)*SCALE
   if FIXED_GAUGE:
    row_shift=tl.reshape(tl.broadcast_to(shift[:,None],(BM//64,64)),(BM,))
    observed=tl.where(valid[None,:],tl.exp2(raw-row_shift[:,None]),0.)
    p=observed*bias[None,:]
   else:
    if BASE2 and not PRE_SCALED_BIAS:bias*=1.4426950408889634
    z=tl.where(valid[None],raw+bias[None],-float('inf'))
    mx=tl.maximum(maximum,tl.max(z,1));safe=tl.where(mx==-float('inf'),0.,mx)
    if BASE2:
     alpha=tl.exp2(maximum-safe);p=tl.exp2(z-safe[:,None])
    else:
     alpha=tl.exp(maximum-safe);p=tl.exp(z-safe[:,None])
   pden=tl.sum(p,1);pcast=p.to(v.dtype)
   if CAPTURE and (not SEPARATE_PREFIX or segment==1):
    if not SKIP_PREFIX or (tile+1)*BK>S:
     row_shift=tl.reshape(tl.broadcast_to(shift[:,None],(BM//64,64)),(BM,))
     if FIXED_GAUGE:
      contribution=tl.where(qm[:,None],observed,0.)
     elif DIRECT_EXPORT:
      observed=tl.exp2(raw-row_shift[:,None]) if BASE2 else tl.exp(raw-row_shift[:,None])
      contribution=tl.where(qm[:,None]&valid[None,:],observed,0.)
     else:
      row_factor=tl.exp(safe-row_shift-reference)
      column_factor=tl.where(valid,tl.exp(reference-bias),0.)
      contribution=tl.where(qm[:,None],p*row_factor[:,None]*column_factor[None,:],0.)
     if WARP_HISTORY:
      value=tl.sum(tl.reshape(contribution,(4,16,BK)),1)
      ab,cd=tl.split(tl.permute(tl.reshape(value,(2,2,BK)),(1,2,0)))
      left,right=tl.split(tl.trans(ab+cd))
      mean_value=(left+right)/tl.sum(count,0)
      tl.store(MEAN+(h*G*4+t)*A+at-S,mean_value,(at>=S)&(at<S+A))
     else:
      value=tl.sum(tl.reshape(contribution,(BM//64,64,BK)),1)/count[:,None]
      tl.store(MEAN+((h*G*4+parts[:,None])*A+at[None,:]-S),value,(at[None,:]>=S)&(at[None,:]<S+A))
      if EXPORT_SUM:history_sum+=tl.sum(tl.where(at[None,:]>=S,value,0.),1)
   if FIXED_GAUGE:
    acc=tl.dot(pcast,v,acc,allow_tf32=False)
    den+=pden
   elif FUSED_ACC:
    acc=tl.dot(pcast,v,acc*alpha[:,None],allow_tf32=False)
   else:
    acc=acc*alpha[:,None]+tl.dot(pcast,v,allow_tf32=False)
   if not FIXED_GAUGE:den=den*alpha+pden;maximum=mx
  if TRIM_TAIL and CAPTURE:
   for tile in range(tl.maximum(begin,end),full_end):
    at=tile*BK+kk
    tl.store(MEAN+((h*G*4+parts[:,None])*A+at[None,:]-S),0.,(at[None,:]>=S)&(at[None,:]<S+A))
 if CAPTURE and EXPORT_SUM:tl.store(HISTORY_SUM+(split*H+h)*G*4+parts,history_sum)
 if SPLITS==1:
  out=acc/den[:,None]
  if FIXED_GAUGE:
   safe=(den>=5.421010862427522e-20)&(den<=1.8446744073709552e19)
   out=tl.where(safe[:,None],out,float('nan'))
  tl.store(OUT+h*OH+rr[:,None]*OR+dd[None],out,(rr[:,None]<R)&(dd[None]<D))
 else:
  base=((split*H+h)*R+rr)
  write=rr<R
  if ACTIVE_SPLITS:
   write=write&(split*span*BK<S+tl.load(VALID_COUNTS+g))
  tl.store(OUT+base[:,None]*D+dd[None],acc,write[:,None]&(dd[None]<D))
  tl.store(STATS+base*2,maximum,write)
  tl.store(STATS+base*2+1,den,write)

@triton.jit
def _merge(PARTIAL,STATS,OUT,VALID_COUNTS,H:tl.constexpr,R:tl.constexpr,D:tl.constexpr,OH:tl.constexpr,OR:tl.constexpr,
           SPLITS:tl.constexpr,BD:tl.constexpr,BM:tl.constexpr,BASE2:tl.constexpr,FIXED_GAUGE:tl.constexpr,ACTIVE_SPLITS:tl.constexpr,SP:tl.constexpr,SK:tl.constexpr):
 h=tl.program_id(1)
 rows=tl.program_id(0)*BM+tl.arange(0,BM)
 cols=tl.arange(0,BD)
 acc=tl.zeros((BM,BD),tl.float32)
 denominator=tl.zeros((BM,),tl.float32)
 maximum=tl.full((BM,),-float('inf'),tl.float32)
 if ACTIVE_SPLITS:
  key_count=SP+tl.load(VALID_COUNTS+tl.program_id(0)//(256//BM))
 for split in tl.static_range(SPLITS):
  base=(split*H+h)*R+rows
  read=rows<R
  if ACTIVE_SPLITS:
   read=read&(split*SK<key_count)
  m=tl.load(STATS+base*2,read,-float('inf'))
  z=tl.load(STATS+base*2+1,read,0.)
  v=tl.load(PARTIAL+base[:,None]*D+cols[None,:],read[:,None]&(cols[None,:]<D),0.)
  if FIXED_GAUGE:
   acc+=v
   denominator+=z
  else:
   new_max=tl.maximum(maximum,m)
   safe=tl.where(new_max==-float('inf'),0.,new_max)
   alpha=tl.exp2(maximum-safe) if BASE2 else tl.exp(maximum-safe)
   beta=tl.exp2(m-safe) if BASE2 else tl.exp(m-safe)
   acc=acc*alpha[:,None]+v*beta[:,None]
   denominator=denominator*alpha+z*beta
   maximum=new_max
 out=acc/denominator[:,None]
 if FIXED_GAUGE:
  safe=(denominator>=5.421010862427522e-20)&(denominator<=1.8446744073709552e19)
  out=tl.where(safe[:,None],out,float('nan'))
 tl.store(OUT+h*OH+rows[:,None]*OR+cols[None,:],out,(rows[:,None]<R)&(cols[None,:]<D))

@triton.jit
def _pack_aligned_support(IDS, SPECIAL, WEIGHTS, PATCHES, OUT_IDS, OUT_WEIGHTS,
                          A:tl.constexpr, S:tl.constexpr, AP:tl.constexpr,
                          SP:tl.constexpr, B:tl.constexpr, MAP_PATCHES:tl.constexpr):
 group=tl.program_id(0)
 slot=tl.program_id(1)*B+tl.arange(0,B)
 patch=slot-SP
 selected=(patch>=0)&(patch<A)
 key=tl.load(IDS+group*A+patch,selected,-1)
 if MAP_PATCHES:
  key=tl.load(PATCHES+key,key>=0,-1)
 special=tl.load(SPECIAL+slot,slot<S,-1)
 weight=tl.load(WEIGHTS+group*A+patch,selected,1.)
 tl.store(OUT_IDS+group*(SP+AP)+slot,tl.where(slot<SP,special,key),slot<SP+AP)
 tl.store(OUT_WEIGHTS+group*(SP+AP)+slot,weight,slot<SP+AP)


def _workspace(cache, name, shape, device):
 if cache is None:
  return torch.empty(shape,device=device,dtype=torch.float32)
 size=math.prod(shape)
 buffer=cache.get(name)
 if buffer is None or buffer.numel()<size or buffer.device!=device:
  buffer=torch.empty(size,device=device,dtype=torch.float32)
  cache[name]=buffer
 return buffer[:size].view(shape)


def attention(q,k,v,ids,specials,weights,scale,mask,counts,capture=True,warps=4,stages=1,shift=None, *, block_keys=128, block_queries=64, output=None, direct_export=False, splits=1, fused_acc=False, cache_modifier="", eviction_policy="", transpose_qk=False, skip_prefix=False, unroll=1, warp_specialize=False, export_sum=False, base2=False, pre_scaled_bias=False, aligned_support=False, fixed_gauge=False, scratch=None, separate_prefix=False, split_keys=None, valid_counts=None, fused_layout=False, zero_copy_layout=False, grid_order=0, merge_rows=16, merge_warps=4, patch_map=None, active_splits=False):
 h,r,d=q.shape;g,a=ids.shape;assert g==triton.cdiv(r,256)
 orig_a=a
 warp_history=capture and fixed_gauge and block_queries==64 and not export_sum and warps==4
 if active_splits:assert fixed_gauge and valid_counts is not None and split_keys is not None and 256%merge_rows==0
 if patch_map is not None:assert fixed_gauge and aligned_support and not zero_copy_layout
 if zero_copy_layout:assert fixed_gauge and aligned_support and separate_prefix
 if valid_counts is not None:assert fixed_gauge and aligned_support and separate_prefix
 if separate_prefix:assert aligned_support
 if fixed_gauge:assert base2 and direct_export
 if aligned_support:assert base2 and direct_export
 if base2:
  assert direct_export
  scale*=1.4426950408889634
 bias=(weights if fixed_gauge else (weights.log()*1.4426950408889634 if base2 and pre_scaled_bias else weights.log())).contiguous();reference=torch.empty(g,device=q.device) if capture else bias
 if capture and not direct_export:_biasmax[(g,)](bias,reference,a,triton.next_power_of_2(a),num_warps=8)
 special_count=len(specials)
 if aligned_support:
  pad_s=(-special_count)%block_keys
  pad_a=(-a)%block_keys
  a+=pad_a
  special_count+=pad_s
  if zero_copy_layout:
   pass
  elif (fused_layout or patch_map is not None) and fixed_gauge:
   aligned_ids=torch.empty((g,special_count+a),device=q.device,dtype=torch.int32)
   aligned_bias=torch.empty((g,special_count+a),device=q.device,dtype=bias.dtype)
   _pack_aligned_support[(g,triton.cdiv(special_count+a,256))](ids,specials,bias,ids if patch_map is None else patch_map,aligned_ids,aligned_bias,orig_a,special_count-pad_s,a,special_count,256,patch_map is not None,num_warps=4)
   ids,bias=aligned_ids,aligned_bias
  else:
   ids=torch.cat((torch.nn.functional.pad(specials,(0,pad_s),value=-1).expand(g,-1),
                  torch.nn.functional.pad(ids,(0,pad_a),value=-1)),dim=1).int()
   bias=torch.nn.functional.pad(bias,(special_count,pad_a),value=1. if fixed_gauge else 0.)
 out=torch.empty_like(q) if output is None else output
 mean=out
 if capture:
  mean=_workspace(scratch,'mean',(h,g,4,a),q.device)
 if capture and r%256:mean[:,-1,triton.cdiv(r%256,64):].zero_()
 if split_keys is not None:
  if split_keys<block_keys or split_keys%block_keys:
   raise ValueError("Split size must be a positive multiple of the Key tile")
  splits=triton.cdiv(a+special_count,split_keys)
 partial=_workspace(scratch,'partial',(splits,h,r,d),q.device) if splits>1 else out
 stats=_workspace(scratch,'stats',(splits,h,r,2),q.device) if splits>1 else out
 history_sum=torch.zeros((splits,h,g,4),device=q.device,dtype=torch.float32) if capture and export_sum else out
 query_tiles=triton.cdiv(r,block_queries)
 grid=((query_tiles,h,splits),(query_tiles,splits,h),(splits,query_tiles,h),(h,query_tiles,splits))[grid_order]
 compiled=_attention[grid](q,k,v,ids,specials,bias,mask,counts,shift,reference,partial if splits>1 else out,mean,stats,history_sum,valid_counts if valid_counts is not None else counts,h,r,k.shape[1],g,a,special_count,d,q.stride(0),k.stride(0),v.stride(0),out.stride(0),out.stride(1),scale,capture,triton.next_power_of_2(d),block_queries,block_keys,direct_export,splits,fused_acc,cache_modifier,eviction_policy,transpose_qk,skip_prefix,unroll,warp_specialize,export_sum,base2,pre_scaled_bias,aligned_support,fixed_gauge,separate_prefix,split_keys//block_keys if split_keys is not None else 0,valid_counts is not None,zero_copy_layout,orig_a,len(specials),grid_order,active_splits,warp_history,num_warps=warps,num_stages=stages)
 if splits>1:_merge[(triton.cdiv(r,merge_rows),h)](partial,stats,out,valid_counts if valid_counts is not None else counts,h,r,d,out.stride(0),out.stride(1),splits,triton.next_power_of_2(d),merge_rows,base2,fixed_gauge,active_splits,special_count,split_keys or 0,num_warps=merge_warps)
 return out,dict(mean_exp=mean[..., :orig_a] if capture else mean,block_keys=block_keys,history_sum=history_sum if capture and export_sum else None, registers=compiled.n_regs, spills=compiled.n_spills, shared_bytes=compiled.metadata.shared,partial_bytes=(partial.numel()+stats.numel())*4 if splits>1 else 0)
