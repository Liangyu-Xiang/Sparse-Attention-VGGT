import importlib.util,json,torch
from pathlib import Path
from scripts.quality_metrics import quality_record,quality_markdown

from saf3r.triton import zdrop_history_states as hist
from saf3r.triton import reuse_drop_moment as drop
out=Path('outputs/cluster_wide_validation_20261104');out.mkdir(parents=True,exist_ok=True);checks=[]
torch.manual_seed(0)
for w in [32,64,128,256,512]:
 f=2;n=f*w;g=3;d=64;labels=torch.arange(n,device='cuda')//w;members=torch.arange(n,device='cuda',dtype=torch.int32).reshape(f,w);z=torch.randn(n,d,device='cuda');prior=torch.rand(g,n,device='cuda');prior/=prior.sum(1,keepdim=True)
 distance=(z[:,None,:]-z[members[labels]]).square().sum(-1).contiguous();qualified=torch.rand(g,n,device='cuda')>.8;qualified[0]=False;geometry=dict(width=w,members=members,distances=distance)
 loss,owner=hist.history_loss(prior,labels,geometry,qualified,f)
 ds=distance[None].expand(g,-1,-1).masked_fill(~qualified[:,members[labels]],float('inf'));minimum=ds.min(-1).values
 own=torch.where((ds==minimum[:,:,None])&qualified[:,members[labels]],members[labels][None],2147483647).min(-1).values
 own=torch.where(qualified,torch.arange(n,device='cuda')[None],own);valid=own<2147483647
 expected=torch.zeros(g,f,device='cuda').scatter_add_(1,labels[None].expand(g,-1),torch.where(valid,prior*minimum,0))
 assert torch.equal(owner,torch.where(valid,own,-1).int())
 torch.testing.assert_close(loss,expected,rtol=2e-6,atol=2e-6)
 pop=torch.bincount(labels,minlength=f).float();actual=drop.drop_norm(prior,z,geometry,f,drop.prepare(pop))
 ref=torch.stack([(prior[:,labels==i,None]*z[None,labels==i]).sum(1).norm(dim=-1) for i in range(f)],1)
 torch.testing.assert_close(actual,ref,rtol=2e-6,atol=2e-6)
 checks.append(dict(width=w,loss_max_error=float((loss-expected).abs().max()),drop_max_error=float((actual-ref).abs().max()),owners_exact=True))
q=quality_record({},'Synthetic wide-cluster kernel numerical test; no pose or reconstruction inference')
(out/'result.json').write_text(json.dumps(dict(complete=True,checks=checks,quality=q),indent=2)+'\n');(out/'quality.md').write_text(quality_markdown(q));print(checks)
