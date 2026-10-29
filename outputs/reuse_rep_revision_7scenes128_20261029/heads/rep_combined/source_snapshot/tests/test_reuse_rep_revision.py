import pytest
import torch
from saf3r.patch.reuse_cosine_distortion import relative_logit_prior,cosine_states
from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,ReuseExploreState,ReuseResidualExploration,budget_states
from saf3r.models.official_vggt.layers.attention import Attention


def test_relative_prior_cross_grid_and_full_observation():
    grid=torch.tensor([0,0,1,1,1]);labels=torch.tensor([0,1,0,1,1])
    p=torch.tensor([[.3,.7],[.8,.2]])
    ids=torch.tensor([[0,2,-1],[0,1,2]])
    delta=torch.tensor([[2.,-1.,0.],[0.,0.,0.]])
    a,pi=relative_logit_prior(p,grid,torch.tensor([2.,3.]),ids,delta,labels,2)
    torch.testing.assert_close(torch.zeros_like(p).scatter_add_(1,grid[None].expand(2,-1),a),p)
    torch.testing.assert_close(pi.sum(1),torch.ones(2))
    torch.testing.assert_close(a[0,0]/a[0,1],torch.exp(torch.tensor(2.)))
    torch.testing.assert_close(a[1],p[1,grid]/torch.tensor([2.,2.,3.,3.,3.]))
    full=torch.arange(5)[None].repeat(2,1)
    a,_=relative_logit_prior(p,grid,torch.tensor([2.,3.]),full,torch.full((2,5),-30.),labels,2)
    torch.testing.assert_close(torch.zeros_like(p).scatter_add_(1,grid[None].expand(2,-1),a),p)


def test_rep_gain_equivalent_and_drop_negative_similarity_unclipped():
    z=torch.tensor([[1.,0.],[-1.,0.],[-1.,0.]])
    pi=torch.tensor([[1.]]);labels=torch.zeros(3,dtype=torch.long)
    errors,costs=cosine_states(pi,z,torch.tensor([0]),labels,torch.tensor([3.]),torch.full((1,1,4),-1),torch.zeros(1,1,dtype=torch.bool),torch.ones(1,3)/3,1.)
    assert errors[0,0,0]==1 and errors[0,0,1]>1
    gain=errors[:,:,1:2]-errors
    assert gain[0,0,0]>0
    a,_=budget_states(errors,costs,1)
    b,_=budget_states(-gain,costs,1,lambda_upper=errors[:,:,0].amax(1)+1e-12)
    assert torch.equal(a,b)


@pytest.mark.parametrize('mode',['all_probability','all_relative_logit'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all_observed_edges_including_seed_reps_are_history(mode,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    torch.set_num_threads(1);torch.manual_seed(12)
    cfg=ReuseExploreConfig(state_loss='cosine_distortion',drop_multiplier=1.,history_mode=mode,rep_gain=True,capture=True,source_width=4,window_groups=2)
    state=ReuseExploreState(cfg);state.reset(1,8,16,device)
    for layer in range(3):
        attention=Attention(dim=32,num_heads=2).to(device).eval()
        wrapped=ReuseResidualExploration(attention,state,layer).eval()
        with torch.inference_mode(),torch.autocast(device,enabled=device=='cuda',dtype=torch.bfloat16):
            out=wrapped(torch.randn(1,133,32,device=device))
        assert torch.isfinite(out).all()
        for i,w in enumerate(state.captured[layer]['windows']):
            valid=w['ids']>=0
            assert torch.equal(w['genuine'],valid)
            for row in w['ids']:
                assert len(row[row>=0].unique())==(row>=0).sum()
            start=i*2
            assert torch.equal(state.history_ids[start:start+len(valid)]>=0,valid)
        if mode=='all_relative_logit':
            assert torch.isfinite(state.history_relative).all()
        if layer==1:assert state.records[-1]['observed_history_mean']>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')
def test_fused_relative_logits_match_raw_qk_without_compensation():
    from saf3r.triton.reuse_explore_av import weighted_attention
    torch.manual_seed(17);device='cuda';h,r,n,d=2,45,41,16
    q=torch.randn(h,r,d,device=device,dtype=torch.float16)
    k=torch.randn(h,n,d,device=device,dtype=torch.float16);v=torch.randn_like(k)
    ids=torch.tensor([[1,3,5,-1],[2,7,11,17]],device=device,dtype=torch.int32)
    bias=torch.tensor([[0.,3.,1.,-torch.inf],[5.,0.,2.,1.]],device=device)
    grid=torch.arange(n,device=device)%3
    pooled=torch.randn(h,3,d,device=device,dtype=torch.float16)
    special=torch.tensor([0],device=device,dtype=torch.long)
    out,mass,delta=weighted_attention(q,k,v,ids,bias,special,.25,relative_reference=(pooled,grid))
    refout,refmass=weighted_attention(q,k,v,ids,bias,special,.25)
    torch.testing.assert_close(out,refout);torch.testing.assert_close(mass,refmass)
    for g in range(2):
        qm=q[:,g*32:min(r,(g+1)*32)].float().mean(1)
        valid=ids[g]>=0;ii=ids[g,valid].long()
        expected=((qm[:,None]*(k[:,ii].float()-pooled[:,grid[ii]].float())).sum(-1)*.25).mean(0)
        torch.testing.assert_close(delta[g,valid],expected,atol=8e-4,rtol=4e-3)
