"""Graph selection must depend on refreshed statistics, not unused Value storage."""
import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_shared_graph_omits_unused_values_and_matches_eager_after_refresh():
    from saf3r.patch.reuse_window_graph import Window,eager
    from saf3r.triton.reuse_shared_value_states import prepare_geometry,prepare
    from saf3r.triton.reuse_drop_moment import prepare as bins
    torch.manual_seed(395)
    device='cuda';g,n,f,d=4,120,12,128
    labels=torch.arange(n,device=device)//10;reps=torch.arange(0,n,10,device=device)
    population=torch.full((f,),10.,device=device);grid=torch.arange(n,device=device)//20
    prior=torch.rand(g,n,device=device);prior/=prior.sum(1,keepdim=True)
    pi=torch.zeros(g,f,device=device).scatter_add_(1,labels[None].expand(g,-1),prior)
    uniform=torch.full((g,6),1/n,device=device)
    old=torch.arange(0,n,3,device=device,dtype=torch.int32)[None].expand(g,-1).contiguous()
    allowed=torch.rand(g,f,device=device)>.4
    z=torch.randn(n,d,device=device)
    def geometry(values):
        geom=prepare_geometry(values,labels,reps,population,min_width=16)
        prepare(values,labels,reps,grid,6,geom)
        geom.update(drop_bins=bins(population),shared_value_allow_observed_rep=True,
                    shared_value_weighting='cluster_population',fuse_state_kernel=True)
        return geom
    geom=geometry(z)
    window=Window(prior,pi,old,allowed,z,reps,labels,population,geom,32,16,1.,bucket_bins='stable',uniform=uniform)
    assert window.constants[0].numel()==0
    for factor in [1.,1.7]:
        fresh=z*factor;geom=geometry(fresh)
        window.update_constants(fresh,reps,labels,population,geom)
        actual=window(prior,pi,old,allowed,uniform=uniform)
        expected=eager(prior,pi,old,allowed,fresh,reps,labels,population,geom,32,16,1.,uniform=uniform)
        for a,b in zip(actual[:3],expected[:3]):torch.testing.assert_close(a,b)
