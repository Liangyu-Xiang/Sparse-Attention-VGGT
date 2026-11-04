import json,torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.profiler import profile,ProfilerActivity
rows=[];torch.manual_seed(0)
for dtype in [torch.float16,torch.bfloat16]:
 q=torch.randn(2,3,17,64,device='cuda',dtype=dtype);k=torch.randn(2,3,53,64,device='cuda',dtype=dtype);v=torch.randn_like(k);bias=torch.randn(2,1,1,53,device='cuda')*3;bias[:,:,:,-3:]=-torch.inf
 ref=(q.float()@k.float().transpose(-1,-2)*.125+bias).softmax(-1)@v.float()
 try:
  with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):out=F.scaled_dot_product_attention(q,k,v,attn_mask=bias)
  rows.append(dict(dtype=str(dtype),mode='direct_bias',works=True))
 except Exception as e:rows.append(dict(dtype=str(dtype),mode='direct_bias',works=False,reason=str(e)))
 for dim in [72,128]:
  qa=torch.zeros((*q.shape[:-1],dim),device='cuda',dtype=dtype);ka=torch.zeros((*k.shape[:-1],dim),device='cuda',dtype=dtype);va=torch.zeros_like(ka);qa[...,:64]=q;ka[...,:64]=k;va[...,:64]=v;qa[...,64:67]=1
  residual=bias[:,0,0]/.125;valid=torch.isfinite(residual)
  for i in range(3):
   part=torch.where(valid,residual,0).to(dtype);ka[...,64+i]=part[:,None];residual=residual-part.float()
  ka[...,64]=ka[...,64].masked_fill(~valid[:,None],-torch.inf)
  try:
   with profile(activities=[ProfilerActivity.CPU]) as prof:
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):out=F.scaled_dot_product_attention(qa,ka,va,scale=.125)[...,:64]
   torch.cuda.synchronize();rel=float((out.float()-ref).norm()/ref.norm());ops=[x.key for x in prof.key_averages() if 'scaled_dot_product' in x.key];rows.append(dict(dtype=str(dtype),mode='augment',dim=dim,works=True,finite=bool(torch.isfinite(out).all()),relative_l2=rel,ops=ops))
  except Exception as e:rows.append(dict(dtype=str(dtype),mode='augment',dim=dim,works=False,reason=str(e)))
print(json.dumps(dict(checks=rows,quality={k:None for k in ['auc03','auc30','acc','comp','cd','f1']},quality_reason='Backend capability diagnostic only; no geometry evaluation',dense_speedup=None,dense_reason='Capability diagnostic,not timed full-forward comparison'),indent=2),flush=True)
