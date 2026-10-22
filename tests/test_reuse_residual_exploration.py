import pytest
import torch
from saf3r.patch.reuse_residual_exploration import (ReuseExploreConfig,ReuseExploreState,
    ReuseResidualExploration,canonical_layout,budget_states,top_history,reference_attention)
from saf3r.patch.region_history_attention import prepare_qkv
from saf3r.models.official_vggt.layers.attention import Attention


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_budget_prefers_low_error_and_is_feasible_without_fixed_history_ratio():
    errors=torch.tensor([[[10.,1.,0.],[.01,.005,0.]]])
    costs=torch.tensor([[[0.,1.,4.],[0.,1.,4.]]])
    chosen,count=budget_states(errors,costs,4,20)
    assert count<=4 and chosen.tolist()==[[2,0]]
    chosen,count=budget_states(torch.zeros_like(errors),costs,4,20)
    assert count==0 and chosen.tolist()==[[0,0]]


def test_stability_uses_measured_mass_not_old_edge_counts():
    labels=torch.tensor([0,0,0,1,1,1]);reps=torch.tensor([0,3]);grid=torch.tensor([0,0,0,1,1,1])
    ids=torch.tensor([[1,2,4,5]],dtype=torch.int32);w=torch.tensor([[.6,.001,.3,.001]])
    h,u,discard=top_history(ids,w,torch.tensor([[.01,1.]]),grid,labels,reps,2)
    assert h.tolist()==[[[1,2],[4,5]]]
    torch.testing.assert_close(u,torch.tensor([[[.006,.00001],[.3,.001]]]))
    h,u,_=top_history(ids,w,torch.tensor([[0.,1.]]),grid,labels,reps,2)
    assert (h[0,0]==-1).all() and (u[0,0]==0).all()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_weighted_kernel_mass_tails_and_noncontiguous_query_window(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    torch.manual_seed(17)
    dtype=torch.bfloat16 if device=='cuda' else torch.float32
    q=torch.randn(2,101,32,device=device,dtype=dtype)[:,3:38]
    k=torch.randn(2,73,32,device=device,dtype=dtype);v=torch.randn_like(k)
    ids=torch.tensor([[5,7,9,11,13,15,17,-1],[6,8,10,12,-1,-1,-1,-1]],device=device,dtype=torch.int32)
    pop=torch.tensor([[8,1,2,3,1,1,1,0],[2,1,1,3,0,0,0,0]],device=device,dtype=torch.float32)
    special=torch.arange(5,device=device)
    expected,mass=reference_attention(q,k,v,ids,pop.log(),special,32**-.5)
    if device=='cuda':
        from saf3r.triton.reuse_explore_av import weighted_attention
        out,actual=weighted_attention(q,k,v,ids,pop.log(),special,32**-.5)
        torch.testing.assert_close(out.float(),expected.float(),atol=.015,rtol=.025)
        torch.testing.assert_close(actual,mass,atol=.001,rtol=.02)
    assert (mass[ids<0]==0).all() and (mass.sum(1)<1).all()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all24_weighted_support_population_no_duplicates_history_and_reset(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    torch.manual_seed(5)
    c=ReuseExploreConfig(capture=True,source_width=4,window_groups=2)
    s=ReuseExploreState(c);s.reset(1,8,16,device);grid=s.grid.clone()
    for layer in range(24):
        original=Attention(dim=32,num_heads=2).to(device).eval()
        wrapper=ReuseResidualExploration(original,s,layer).eval()
        x=torch.randn(1,133,32,device=device)
        with torch.inference_mode(),torch.autocast(device,enabled=device=='cuda',dtype=torch.bfloat16):
            actual=wrapper(x)
            q,k,v=prepare_qkv(original,x,None);q=q.to(v.dtype);k=k.to(v.dtype)
            parts=[]
            for wi,window in enumerate(s.captured[layer]['windows']):
                ids=window['ids'];pop=window['population'];real=window['genuine']
                first=wi*c.window_groups*32;end=min(133,first+len(ids)*32)
                expected,mass=reference_attention(q[:,first:end],k,v,ids,pop.log(),s.specials,original.scale)
                parts.append(expected)
                for row in range(len(ids)):
                    valid=ids[row]>=0;selected=ids[row][valid].long()
                    assert len(selected.unique())==len(selected) and len(selected)<=s.budget
                    assert (pop[row][valid]>=1).all()
                    for rep_id,weight in zip(ids[row][~real[row]&valid],pop[row][~real[row]&valid]):
                        labels=s.fine.labels
                        cluster=labels[rep_id.long()-5]
                        members=(labels==cluster).sum()
                        nreal=(labels[ids[row][real[row]].long()-5]==cluster).sum()
                        assert weight==members-nreal
            expected=torch.cat(parts,dim=1)
            expected=original.proj_drop(original.proj(expected.transpose(0,1).reshape(1,133,32)))
        torch.testing.assert_close(actual.float(),expected.float(),atol=.015 if device=='cuda' else 1e-6,rtol=.03 if device=='cuda' else 1e-5)
        assert torch.isfinite(actual).all() and torch.equal(s.grid,grid)
        record=s.records[-1]
        torch.testing.assert_close(record['reuse_edges_mean']+record['representatives_mean']+record['exact_edges_mean'],
                                   record['counts_mean'])
        for g,h in enumerate(s.history_ids):
            wi=g//c.window_groups;row=g%c.window_groups
            window=s.captured[layer]['windows'][wi]
            allowed=window['ids'][row][window['genuine'][row]]
            assert set((h[h>=0]+5).tolist()).issubset(set(allowed.tolist()))
        assert (s.previous_grid.sum(1).float()-1).abs().max()<.01
    s.reset(1,8,16,device)
    assert s.previous_grid is None and s.history_ids is None and s.previous_layer==-1 and not s.records
