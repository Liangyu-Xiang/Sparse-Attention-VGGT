"""Fixed-order correction and log-cache calibration math; graph-capture safety."""
import pytest
import torch
from saf3r.triton import reuse_shared_value_states as shared
from saf3r.triton.reuse_drop_moment import prepare
from saf3r.patch.reuse_inference_ops import gate
from scripts.validate_shared_value_loss import example

@pytest.fixture
def fixed_order():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    previous=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:yield
    finally:torch.use_deterministic_algorithms(previous)

def test_fixed_corrections_math_repeat_and_graph(fixed_order):
    z,labels,grid,reps,pop,u,prior,old,pi,*_=example('cuda')
    old=old.int();old[:,-2:]=-1
    context=gate(prior,pi,pop,old,labels,reps)
    geo=shared.prepare_geometry(z,labels,reps,pop,min_width=64)
    shared.prepare(z,labels,reps,grid,u.shape[1],geo)
    first=shared.corrections(prior,old,u,labels,geo,context)
    second=shared.corrections(prior,old,u,labels,geo,context)
    for x,y in zip(first,second):torch.testing.assert_close(x,y,atol=0,rtol=0)
    expected=[torch.zeros_like(first[0],dtype=torch.float64) for _ in range(3)]
    for g in range(len(old)):
        for key in old[g].tolist():
            if key<0:continue
            c=int(labels[key]);r=int(grid[key]);prob=prior[g,key].double();delta=prob-u[g,r].double()
            a=geo['shared_token_drop'][key].double();b=geo['shared_token_rep'][key].double()
            expected[0][g,c]+=delta*a;expected[1][g,c]+=delta*b
            if context['qualified'][g,key]:expected[2][g,c]+=prob*a
    for x,y in zip(first,expected):torch.testing.assert_close(x.double(),y,atol=2e-5,rtol=2e-6)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=shared.corrections(prior,old,u,labels,geo,context)
    prior.mul_(.75)
    graph.replay()
    eager=shared.corrections(prior,old,u,labels,geo,context)
    for x,y in zip(captured,eager):torch.testing.assert_close(x,y,atol=0,rtol=0)

def test_history_calibration_fixed_reduction_and_fp16_cache(fixed_order):
    from saf3r.triton.reuse_probability_history import store
    torch.manual_seed(93)
    g,p,a,r,n=3,1,31,7,97
    grid=torch.arange(n,device='cuda')%r;sizes=torch.bincount(grid,minlength=r).float()
    ids=torch.stack([torch.randperm(n,device='cuda')[:a] for _ in range(g)]).int();ids[:,-3:]=-1
    scores=torch.randn(g,p,a,device='cuda');missing=torch.randn(g,p,r,device='cuda')
    caches=[torch.empty_like(scores,dtype=torch.float16) for _ in range(2)]
    results=[store(scores,missing,ids,grid,sizes,c) for c in caches]
    for x,y in zip(results[0],results[1]):torch.testing.assert_close(x,y,atol=0,rtol=0)
    torch.testing.assert_close(*caches,atol=0,rtol=0)
    uniform,counts,den=results[0]
    for group in range(g):
        for region in range(r):
            selected=(ids[group]>=0)&(grid[ids[group].long().clamp_min(0)]==region)
            terms=scores[group,0,selected].double().exp().sum()+(sizes[region]-counts[group,region]).double()*missing[group,0,region].double().exp()
            torch.testing.assert_close(den[group,0,region].double(),terms.log(),atol=2e-6,rtol=2e-6)
