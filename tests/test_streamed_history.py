from types import SimpleNamespace
import pytest,torch
from saf3r.patch.current_probe_reuse import main_attention
from saf3r.patch import streamed_probability_history as hist

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

@pytest.mark.parametrize('n,a',[(259,17),(512,31)])
def test_inline_scores_reuse_unbiased_qk_without_changing_av(n,a):
 torch.manual_seed(73);h,d,gs=2,32,256;g=(n+gs-1)//gs
 q=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q)
 specials=torch.tensor([0,5],device='cuda');patches=torch.tensor([j for j in range(n) if j not in [0,5]],device='cuda')
 ids=torch.stack([patches[torch.randperm(len(patches),device='cuda')[:a]] for _ in range(g)]).int().contiguous()
 bias=torch.randn(g,a,device='cuda')*.5;pos=(torch.arange(g,device='cuda')*gs+gs//8).clamp_max(n-1)
 dense=q[:,pos].float()@k.float().transpose(1,2)/d**.5;lz=dense.logsumexp(-1).contiguous()
 plain,_=main_attention(q,k,v,ids,specials,d**-.5,bias)
 result,e=main_attention(q,k,v,ids,specials,d**-.5,bias,history_logz=lz)
 torch.testing.assert_close(result,plain,atol=.002,rtol=.002)
 keys=torch.cat((specials[None].expand(g,-1),ids),1).long()
 expected=dense.softmax(-1).mean(0).gather(1,keys)
 torch.testing.assert_close(e['scores'],expected,rtol=2e-5,atol=2e-7)
 assert not bool(e['clipped'].any()) and e['scores'].shape==(g,a+len(specials))

def test_cache_completion_transfer_is_headless_and_mass_preserving():
 torch.manual_seed(51);g,n,r,a,s=2,37,3,11,2;device='cuda'
 grid=torch.arange(n,device=device)%r;sizes=torch.bincount(grid,minlength=r).float()
 ids=torch.stack([torch.randperm(n,device=device)[:a] for _ in range(g)]).int()
 base=torch.rand(g,r,device=device);base=base/(base*sizes).sum(1,keepdim=True)*.8
 obs=dict(base=base,coarse=base*sizes)
 state=SimpleNamespace(patches=torch.arange(n,device=device),specials=torch.arange(s,device=device),grid=grid,grid_sizes=sizes,regions=r,streamed_history_cache=None)
 score=torch.rand(g,s+a,device=device)*.02
 cache=hist.finalize(state,obs,0,ids,score);assert cache['conditional'].shape==(g,a) and cache['conditional'].dtype==torch.float16
 region=grid[ids.long()];mass=torch.zeros(g,r,device=device).scatter_add_(1,region,score[:,s:]);count=torch.zeros_like(mass).scatter_add_(1,region,torch.ones(g,a,device=device));full=mass+(sizes-count)*base
 torch.testing.assert_close(cache['conditional'].float(),score[:,s:]/full.gather(1,region),atol=2e-4,rtol=1e-3)
 state.streamed_history_cache=cache;state.history_ids=ids
 prior=hist.prior(state,obs,0,g)
 totals=torch.zeros(g,r,device=device).scatter_add_(1,grid[None].expand(g,-1),prior)
 torch.testing.assert_close(totals,obs['coarse'],atol=2e-4,rtol=1e-3)
 # Changing historical probabilities MUST change the next-layer result.
 state.streamed_history_cache=dict(cache,conditional=cache['conditional']*.5)
 altered=hist.prior(state,obs,0,g)
 assert (altered-prior).abs().max()>1e-4

def test_current_observation_is_pooled_only_and_head_permutation_invariant():
 torch.manual_seed(77);h,n,d,gs=3,257,32,256;device='cuda';specials=torch.tensor([0,1],device=device);patches=torch.arange(2,n,device=device);grid=torch.arange(len(patches),device=device)%4
 state=SimpleNamespace(config=SimpleNamespace(query_group_size=gs),groups=2,regions=4,grid=grid,patches=patches,specials=specials,grid_sizes=torch.bincount(grid,minlength=4).float())
 q=torch.randn(h,n,d,device=device,dtype=torch.bfloat16);k=torch.randn_like(q)
 a=hist.observe(state,q,k,d**-.5);perm=torch.tensor([2,0,1],device=device);b=hist.observe(state,q[perm],k[perm],d**-.5)
 torch.testing.assert_close(a['coarse'],b['coarse'])
 assert a['logz'].shape==(h,2) and a['coarse'].shape==(2,4)
