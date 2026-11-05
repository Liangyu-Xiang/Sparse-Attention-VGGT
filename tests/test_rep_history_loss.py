import pytest,torch,triton
from saf3r.patch import rep_history_loss as op

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('mode',[1,2])
def test_history_formula_support_weights_and_attention(mode):
    from saf3r.patch import reuse_inference_ops as rule
    from saf3r.patch.shared_value_repair import execution_weights
    from saf3r.triton.reuse_direct_sparse_attention import direct_sparse_attention
    dev='cuda';g,n,f=5,3,1
    p=torch.tensor([[.1,.7,.2]],device=dev).expand(g,-1).contiguous()
    z=torch.tensor([[0.,0.],[2.,0.],[-1.,1.]],device=dev);dist=z.square().sum(-1)
    labels=torch.zeros(n,device=dev,dtype=torch.long);reps=torch.tensor([0],device=dev);pop=torch.tensor([3.],device=dev)
    histories=[[0],[1],[0,1],[0,1,2],[]]
    qual=torch.zeros(g,n,device=dev,dtype=torch.bool)
    for row,ids in enumerate(histories):qual[row,ids]=True
    original_qual=qual.clone();m=qual.sum(1,dtype=torch.int32)[:,None];seen=qual[:,:1].clone()
    rep=(p*dist).sum(1)[:,None];kept=(p*dist*qual).sum(1)[:,None]
    dh=(dist[None]*qual).sum(1)[:,None]*(m==1);pi=p.sum(1)[:,None]
    legacy=(p*z.square().sum(-1)*(~qual)).sum(1)[:,None]
    errors=torch.stack((rep,rep,legacy,torch.zeros_like(rep)),-1);cost=torch.stack((torch.zeros_like(rep),torch.ones_like(rep),m.float(),pop[None].expand(g,-1)),-1)
    objective=errors-rep[:,:,None];amount=torch.empty_like(m)
    op._repair[(g,1)](errors,cost,objective,pi,kept,dh,m,seen,reps,pop,qual,amount,n,f,256,mode,num_warps=4,enable_fp_fusion=False)
    for row,H in enumerate(histories):
        if not H:assert torch.isinf(errors[row,0,2]);continue
        expected=(p[row]*dist*(~original_qual[row])).sum() if mode==1 else (rep[row,0]+(pi[row,0]-2*p[row,H[0]])*dist[H[0]]).clamp_min(0) if len(H)==1 else legacy[row,0]
        torch.testing.assert_close(errors[row,0,2],expected,atol=3e-7,rtol=1e-6)
        support=sorted(set(H)|{0}) if mode==1 else H
        assert torch.where(qual[row])[0].tolist()==support
        assert int(cost[row,0,2])==len(support)
    assert torch.equal(errors[0,0,2],errors[0,0,1]),'H={Rep}: raw loss identity before any selectable-state gate'
    if mode==1:
        assert errors[2,0,2]<=errors[0,0,2] and errors[3,0,2]==0
    else:
        assert errors[1,0,2]<errors[1,0,1],'Single dominant historical key improves approximate loss'
        assert torch.equal(qual,original_qual) and torch.equal(amount,m)
    selected=torch.full((g,f),5,device=dev,dtype=torch.long);selected[-1]=0
    context=dict(qualified=qual,eligible_count=amount,remove_rep=torch.zeros_like(m,dtype=torch.bool))
    ids,_,record=rule.pack(selected,None,reps,labels,pop,3,context)
    assert not bool(context['pack_error'].any()) and not bool(record['native_cost_count_error'].any())
    assert bool((record['counts']<=3).all())
    for row in ids:
        keptids=row[row>=0];assert keptids.unique().numel()==keptids.numel()
    if mode==1:
        weights=torch.empty(ids.shape,device=dev);op._weights_A[(g,1)](ids,labels,reps,selected,pop,amount,weights,n,f,3,256,num_warps=4)
        for row,H in enumerate(histories[:-1]):
            support=ids[row][ids[row]>=0];w=weights[row][ids[row]>=0]
            torch.testing.assert_close(w.sum(),pop[0])
            assert bool((w[support!=0]==1).all())
            assert float(w[support==0][0])==n-(len(support)-1)
    else:weights=execution_weights('cluster_population',selected,ids,reps,labels,pop,p,pi,amount)
    # H={Rep} and Rep have exactly equal support, cost and execution weight.
    assert ids[0,0]==0 and bool((ids[0,1:]==-1).all()) and weights[0,0]==3
    torch.manual_seed(2026);q=torch.randn(2,g*32,16,device=dev,dtype=torch.float16)
    k=torch.randn(2,4,16,device=dev,dtype=torch.float16);v=torch.randn_like(k)
    actual,_=direct_sparse_attention(q,k,v,ids,torch.tensor([3],device=dev),16**-.5,logit_bias=weights.clamp_min(1e-30).log())
    for row in range(g):
        valid=ids[row]>=0;support=torch.cat((torch.tensor([3],device=dev),ids[row,valid].long()))
        bias=torch.cat((torch.zeros(1,device=dev),weights[row,valid].log()))
        qq=q[:,row*32:(row+1)*32].float();logits=qq@k[:,support].float().transpose(-1,-2)*16**-.5+bias
        expected=logits.softmax(-1)@v[:,support].float()
        torch.testing.assert_close(actual[:,row*32:(row+1)*32].float(),expected,atol=1.2e-3,rtol=3e-3)
    assert bool(torch.isfinite(actual).all())

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('mode',['legacy','A','B'])
def test_graph_replay_and_budget(mode):
    from saf3r.triton import reuse_shared_value_states as shared
    from saf3r.patch.reuse_window_graph import Window,eager
    device='cuda';torch.manual_seed(42);n,f,g=12,4,2
    labels=torch.arange(n,device=device)//3;reps=torch.arange(f,device=device)*3;pop=torch.full((f,),3.,device=device)
    z=torch.randn(n,4,device=device);grid=torch.arange(n,device=device)%2
    geometry=shared.prepare_geometry(z,labels,reps,pop,min_width=64);shared.prepare(z,labels,reps,grid,2,geometry)
    geometry.update(rep_history_loss=mode,rep_history_diagnostics=True,shared_value_weighting='cluster_population',shared_value_allow_observed_rep=True,drop_bins=[(4,torch.arange(f,device=device,dtype=torch.int32))],bounded_bins=True)
    prior=torch.rand(g,n,device=device);prior/=prior.sum(1,keepdim=True)
    uniform=torch.zeros(g,2,device=device)
    # All Keys are observed; coarse zero plus historical residual recovers prior exactly.
    old=torch.arange(n,device=device,dtype=torch.int32)[None].expand(g,-1).contiguous()
    pi=torch.zeros(g,f,device=device).scatter_add_(1,labels[None].expand(g,-1),prior)
    allow=torch.ones(g,f,device=device,dtype=torch.bool)
    window=Window(prior,pi,old,allow,z,reps,labels,pop,geometry,5,16,1,'stable',uniform=uniform)
    for iteration in range(2):
        if iteration:
            old[:]=reps.repeat(3)[:n];old[:,f:]=-1  # valid history remains unique
            prior=prior.flip(1).contiguous();pi.zero_().scatter_add_(1,labels[None].expand(g,-1),prior)
        result=window(prior,pi,old,allow,uniform=uniform)
        expected=eager(prior,pi,old,allow,z,reps,labels,pop,geometry,5,16,1,uniform=uniform)
        assert torch.equal(result[0],expected[0]);torch.testing.assert_close(result[2],expected[2])
        if mode=='legacy':
            native=eager(prior,pi,old,allow,z,reps,labels,pop,dict(geometry,rep_history_diagnostics=False),5,16,1,uniform=uniform)
            assert torch.equal(native[0],result[0]);torch.testing.assert_close(native[2],result[2])
        assert not bool(result[3]['implementation_validation_error'].any())
        assert bool((result[3]['counts']<=5).all())
        assert bool(torch.isfinite(result[2]).all())
