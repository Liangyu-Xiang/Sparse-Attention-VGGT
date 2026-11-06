import pytest,torch
from saf3r.triton.teacher_history_probability_tiles import attention
from saf3r.patch.current_probe_reuse import main_attention
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('n,probes',[(512,4),(259,4),(356,4),(480,4),(512,2),(356,2),(259,1)])
def test_av_and_unbiased_probe_partition(n,probes):
 torch.manual_seed(127);device='cuda';h,d,a=2,32,17;g=(n+255)//256;q=torch.randn(h,n,d,device=device,dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q);specials=torch.tensor([0,5],device=device);patches=torch.tensor([j for j in range(n) if j not in [0,5]],device=device);ids=torch.stack([patches[torch.randperm(len(patches),device=device)[:a]] for _ in range(g)]).int();weights=torch.exp(torch.rand(g,a,device=device)*2+30)
 ref,_=main_attention(q,k,v,ids,specials,d**-.5,weights.log());out,e=attention(q,k,v,ids,specials,weights,d**-.5,probes,True);bare,_=attention(q,k,v,ids,specials,weights,d**-.5,probes,False)
 torch.testing.assert_close(out,ref,atol=.002,rtol=.002);torch.testing.assert_close(out,bare,atol=.002,rtol=.002)
 pos=(torch.arange(g,device=device)[:,None]*256+torch.arange(probes,device=device)[None]*(256//probes)+32).clamp_max(n-1);keys=torch.cat((specials[None].expand(g,-1),ids),1).long();raw=torch.einsum('hgpd,hgad->hgpa',q[:,pos].float(),k[:,keys].float())*d**-.5
 torch.testing.assert_close(e['raw_lse'],raw.logsumexp(-1),rtol=2e-5,atol=2e-5)
 encoded=e['encoded'][...,:len(specials)+a].float();indices=torch.arange(encoded.shape[-1],device=device)//e['block_keys'];shift=e['shifts'][:,:,:,indices];inverse=torch.cat((torch.ones(g,len(specials),device=device),e['inverse_weights']),1)
 probability=(encoded*inverse[:,None,None,:]*torch.exp(shift-e['raw_lse'].permute(1,0,2)[:,:,:,None])).permute(1,0,2,3)
 torch.testing.assert_close(probability,raw.softmax(-1),atol=2e-5,rtol=.005)
