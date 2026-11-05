import pytest
import torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA Triton numerical tests')

@pytest.mark.parametrize('probes',[1,4])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_probability_matches_full_key_softmax(probes,dtype):
 from saf3r.triton.current_probe_probability import probability
 torch.manual_seed(71);torch.backends.cuda.matmul.allow_tf32=False
 q=torch.randn(2,79,16,device='cuda',dtype=dtype);k=torch.randn_like(q);patch=torch.arange(79,device='cuda');patch=patch[patch%5!=0]
 actual,pos=probability(q,k,patch,32,probes,.25)
 reference=(q[:,pos.flatten()].float()@k.float().transpose(1,2)*.25).softmax(-1)[:,:,patch].reshape(2,len(pos),probes,-1).mean((0,2))
 torch.testing.assert_close(actual,reference,rtol=2e-5,atol=1e-6)

def test_full_stats_and_A_mass():
 import triton
 from saf3r.patch.current_probe_states import _base,select
 labels=torch.tensor([0,0,0,1,1,1,2,2],device='cuda');reps=torch.tensor([0,3,6],device='cuda');pop=torch.tensor([3.,3.,2.],device='cuda')
 members=torch.tensor([[0,1,2,-1],[3,4,5,-1],[6,7,-1,-1]],device='cuda',dtype=torch.int32)
 features=torch.tensor([[.1,.2],[.4,.1],[.3,.3],[1.,.2],[.8,.3],[1.,1.],[.1,2.],[.1,2.2]],device='cuda');norm=features.square().sum(-1);dr=(features-features[reps[labels]]).square().sum(-1)
 prior=torch.tensor([[.1,.2,.1,.07,.21,.02,.12,.18],[.03,.1,.27,.08,.12,.1,.1,.2]],device='cuda');g,n=prior.shape;f=len(reps)
 pi=torch.empty((g,f),device='cuda');e=torch.empty((g,f,4),device='cuda');cost=torch.empty_like(e);obj=torch.empty_like(e)
 _base[(g,1)](prior,members,norm,dr,pop,pi,e,cost,obj,n,f,4,4,num_warps=4,enable_fp_fusion=False)
 for c in range(f):
  torch.testing.assert_close(pi[:,c],prior[:,labels==c].sum(-1));torch.testing.assert_close(e[:,c,0],(prior[:,labels==c]*norm[labels==c]).sum(-1));torch.testing.assert_close(e[:,c,1],(prior[:,labels==c]*dr[labels==c]).sum(-1))
 old=torch.tensor([[0,1,3,-1],[1,4,6,-1]],device='cuda',dtype=torch.int32)
 geometry=dict(members=members,norm2=norm,rep_distance=dr,probe_member_width=4,rep_history_execution='prior_residual')
 ids,valid,w,record=select(prior,pi,old,None,None,reps,labels,pop,geometry,4,16)
 assert not bool(record['implementation_validation_error'].any());assert ids.shape==(2,4)
 for i in range(g):
  live=ids[i]>=0;assert len(ids[i,live].unique())==int(live.sum());mass=torch.zeros(f,device='cuda').scatter_add_(0,labels[ids[i,live].long()],prior[i,ids[i,live].long()]*w[i,live]);keep=mass>0
  torch.testing.assert_close(mass[keep],pi[i,keep],rtol=1e-5,atol=1e-6)

def test_no_evidence_attention_matches_reference():
 from saf3r.patch.current_probe_reuse import main_attention
 torch.manual_seed(129);h,n,r,d=2,79,67,16;gs=32
 q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16);k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 ids=torch.tensor([[8,17,29,-1],[5,16,33,45],[7,31,41,60]],device='cuda',dtype=torch.int32);special=torch.tensor([0,1,2],device='cuda');weights=torch.tensor([[2.,1.,3.,0.],[1.,4.,1.,1.],[1.,1.,2.,1.]],device='cuda')
 out,_=main_attention(q,k,v,ids,special,.25,weights.log(),query_group_size=gs,block_queries=32)
 refs=[]
 for group in range(3):
  selected=torch.cat((special,ids[group,ids[group]>=0].long()));bias=torch.cat((torch.zeros(3,device='cuda'),weights[group,ids[group]>=0].log()))
  logits=q[:,group*gs:min((group+1)*gs,r)].float()@k[:,selected].float().transpose(1,2)*.25+bias
  refs.append(logits.softmax(-1)@v[:,selected].float())
 reference=torch.cat(refs,1);relative=(out.float()-reference).norm()/reference.norm();assert relative<.005
