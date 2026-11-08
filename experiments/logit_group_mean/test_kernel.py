import sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));import torch,saf3r
p=types.ModuleType('saf3r.patch');p.__path__=[str(ROOT/'saf3r/patch')];sys.modules['saf3r.patch']=p;saf3r.patch=p
from saf3r.triton.group_mean_logit_attention import pool_queries,attention
from saf3r.triton.teacher_history_attention import attention as original

def main():
 torch.manual_seed(41);torch.backends.cuda.matmul.allow_tf32=False
 for n in [256,289,512,735]:
  h,d=4,64;g=(n+255)//256;q=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q);specials=torch.tensor([0,1,2,3,4,255],device='cuda',dtype=torch.int32);mask=torch.ones(n,device='cuda',dtype=torch.bool);mask[specials]=False;counts=torch.nn.functional.pad(mask.float(),(0,g*256-n)).reshape(g,256).sum(1);ids=torch.stack([torch.randperm(n,device='cuda')[:117] for _ in range(g)]).int();weights=torch.rand(g,117,device='cuda')*3+.2
  pooled=pool_queries(q,mask,counts)
  ref=torch.stack([q[:,i*256:min((i+1)*256,n)][:,mask[i*256:min((i+1)*256,n)]].float().mean(1) for i in range(g)],1)[:,:,None]
  assert torch.allclose(pooled,ref,atol=2e-7,rtol=2e-6)
  out,ev=attention(q,k,v,ids,specials,weights,d**-.5,mask,counts)
  old,_=original(q,k,v,ids,specials,weights,d**-.5,4,False)
  err=float((out.float()-old.float()).norm()/old.float().norm());assert err<.003,err
  truth=torch.stack([torch.einsum('hd,had->ha',ref[:,i,0],k[:,ids[i].long()].float())*d**-.5 for i in range(g)],1)[:,:,None]
  assert torch.allclose(ev['mean_logits'],truth,atol=3e-5,rtol=5e-4),float((ev['mean_logits']-truth).abs().max())
  print('GROUP_MEAN_PASS',n,'AV relL2',err,'rawmean maxdiff',float((ev['mean_logits']-truth).abs().max()),flush=True)
if __name__=='__main__':main()
