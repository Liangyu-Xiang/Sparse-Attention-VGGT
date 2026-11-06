import pytest,torch
from saf3r.triton.teacher_history_raw_fast import attention
from saf3r.patch.current_probe_reuse import main_attention
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('n,probes',[(512,4),(259,4),(356,4),(480,4),(512,2),(356,2),(259,1)])
def test_av_and_unbiased_probe_partition(n,probes,h=2,d=32,a=17):
 torch.manual_seed(127);device='cuda';g=(n+255)//256;q=torch.randn(h,n,d,device=device,dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q);specials=torch.tensor([0,5],device=device);patches=torch.tensor([j for j in range(n) if j not in [0,5]],device=device);ids=torch.stack([patches[torch.randperm(len(patches),device=device)[:a]] for _ in range(g)]).int();weights=torch.exp(torch.rand(g,a,device=device)*2+30)
 ref,_=main_attention(q,k,v,ids,specials,d**-.5,weights.log());out,e=attention(q,k,v,ids,specials,weights,d**-.5,probes,True);bare,_=attention(q,k,v,ids,specials,weights,d**-.5,probes,False)
 from saf3r.triton.teacher_history_attention import attention as original_evidence
 old_out,old_e=original_evidence(q,k,v,ids,specials,weights,d**-.5,probes,True)
 torch.testing.assert_close(e['encoded'],old_e['encoded'],atol=0,rtol=0);torch.testing.assert_close(e['shifts'],old_e['shifts'],atol=0,rtol=0)
 torch.testing.assert_close(out,ref,atol=.002,rtol=.002);torch.testing.assert_close(out,bare,atol=.002,rtol=.002)
 pos=(torch.arange(g,device=device)[:,None]*256+torch.arange(probes,device=device)[None]*(256//probes)+32).clamp_max(n-1);keys=torch.cat((specials[None].expand(g,-1),ids),1).long();raw=torch.einsum('hgpd,hgad->hgpa',q[:,pos].float(),k[:,keys].float())*d**-.5
 torch.testing.assert_close(e['raw_lse'],raw.logsumexp(-1),rtol=2e-5,atol=2e-5)
 encoded=e['encoded'].float();indices=torch.arange(encoded.shape[-1],device=device)//128;shift=e['shifts'][:,:,:,indices];decoded=encoded+shift;decoded=decoded.permute(1,0,2,3)
 probability=(decoded-e['raw_lse'][:,:,:,None]).exp();torch.testing.assert_close(probability,raw.softmax(-1),atol=2e-5,rtol=.005)

@pytest.mark.parametrize('n',[512,356])
def test_actual_head_dimension_multitile(n):
 test_av_and_unbiased_probe_partition(n,4,h=16,d=64,a=257)
