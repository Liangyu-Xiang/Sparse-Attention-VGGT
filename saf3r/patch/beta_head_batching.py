"""Batch independent DPT frames without dropping heads or changing precision."""
from types import MethodType
import torch

def install(model,chunk,validate=False):
 checks=[]
 for name in ['depth_head','point_head']:
  head=getattr(model,name,None)
  if head is None:continue
  original=head.forward
  def forward(module,*args,_original=original,_name=name,**kwargs):
   kwargs['frames_chunk_size']=chunk
   result=_original(*args,**kwargs)
   if validate and not getattr(module,'_batching_checked',False):
    baseline=_original(*args,**dict(kwargs,frames_chunk_size=8))
    for key,(got,ref) in enumerate(zip(result,baseline)):
     delta=(got-ref).float();relative=float(delta.norm()/ref.float().norm().clamp_min(1e-30));maximum=float(delta.abs().max())
     if relative>3e-5:raise RuntimeError(f'DPT batching mismatch {_name}/{key}: {relative}')
     checks.append(dict(head=_name,component=key,relative_L2=relative,max_absolute_difference=maximum,reference_chunk=8,new_chunk=chunk,precision='FP32 unchanged'))
    module._batching_checked=True
   return result
  head.forward=MethodType(forward,head)
 return checks
