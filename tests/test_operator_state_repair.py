"""Compare state losses to actual normalized outputs and verify emitted real Keys."""
import pytest
import torch
import triton
from saf3r.patch import operator_state_repair as op


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('calibrated',[False,True])
def test_normalized_losses_match_actual_operators_and_equal_supports(calibrated):
    torch.manual_seed(17);device='cuda';g,f,n,d=2,3,9,4
    labels=torch.tensor([0,0,1,1,1,2,2,2,2],device=device)
    reps=torch.tensor([0,2,5],device=device);alt=torch.tensor([1,3,6],device=device)
    pop=torch.tensor([2.,3.,4.],device=device)
    members=torch.tensor([[0,1,-1,-1],[2,3,4,-1],[5,6,7,8]],device=device,dtype=torch.int32)
    p=torch.rand(g,n,device=device);sp=torch.rand(g,1,device=device)
    total=p.sum(1)+sp.sum(1);p/=total[:,None];sp/=total[:,None]
    y=torch.randn(n,d,device=device);sy=torch.randn(1,d,device=device)
    qual=torch.zeros(g,n,device=device,dtype=torch.bool);qual[:,reps]=True
    qual[:,7]=True;hc=torch.zeros(g,f,device=device,dtype=torch.int32).scatter_add_(1,labels[None].expand(g,-1),qual.int())
    pi=torch.zeros(g,f,device=device).scatter_add_(1,labels[None].expand(g,-1),p)
    a=y.square().sum(-1);r=(y-y[reps[labels]]).square().sum(-1)
    b=(y-y[alt[labels]]).square().sum(-1);pair=(y-(y[reps[labels]]+y[alt[labels]])*.5).square().sum(-1)
    stats=torch.empty(g,f,2*d+7,device=device)
    op._statistics[(g,1)](p,labels,members,y,qual,a,r,b,pair,stats,n,f,4,d,8,4,num_warps=4,enable_fp_fusion=False)
    binned=torch.empty_like(stats)
    # Fixed-capacity graph slots include garbage IDs beyond each live count.
    for size,live_ids in [(2,[0]),(4,[1,2])]:
        clusters=torch.tensor(live_ids+[999]*(4-len(live_ids)),device=device,dtype=torch.int32)
        count_ptr=torch.tensor([len(live_ids)],device=device,dtype=torch.int32)
        op._statistics_capacity[(g,1)](p,labels,members,y,qual,a,r,b,pair,binned,clusters,count_ptr,n,f,4,d,2,size,True,num_warps=4,enable_fp_fusion=False)
    torch.testing.assert_close(binned,stats,atol=2e-6,rtol=2e-5)
    mu=(p@y+sp@sy)/(p.sum(1)+sp.sum(1))[:,None]
    variance=((p@y.square().sum(-1)+sp@sy.square().sum(-1))/(p.sum(1)+sp.sum(1))-mu.square().sum(-1)).clamp_min(0)
    allowed=torch.tensor([[True,False,True],[False,True,False]],device=device)
    e=torch.empty(g,f,6,device=device);cost=torch.empty_like(e);obj=torch.empty_like(e);state_weights=torch.empty_like(e)
    op._losses[(g,1)](stats,p,pi,pop,reps,alt,y,hc,allowed,mu,p.sum(1)+sp.sum(1),variance,e,cost,obj,n,f,d,6,True,0.,32,calibrated,state_weights,num_warps=4,enable_fp_fusion=False)
    for row in range(g):
        for c in range(f):
            all_ids=torch.where(labels==c)[0];h=all_ids[qual[row,all_ids]]
            supports=[[],[int(reps[c])],h.tolist(),all_ids.tolist(),[int(alt[c])],[int(reps[c]),int(alt[c])]]
            for state,ids in enumerate(supports):
                if state==3 and not allowed[row,c]:assert torch.isinf(e[row,c,state]);continue
                weights=torch.ones(n,device=device);weights[all_ids]=0
                if ids:
                    w=float(pop[c])/len(ids)
                    if calibrated and state!=3:w*=float((pi[row,c]/(p[row,ids].sum()*w)).clamp(.25,4.))
                    weights[ids]=w
                    if calibrated:torch.testing.assert_close(state_weights[row,c,state],torch.tensor(w,device=device),atol=2e-5,rtol=2e-5)
                predicted=(p[row]*weights)@y+sp[row]@sy
                predicted/=((p[row]*weights).sum()+sp[row].sum())
                expected=(predicted-mu[row]).square().sum()
                torch.testing.assert_close(e[row,c,state],expected,atol=3e-6,rtol=3e-4)
                assert int(cost[row,c,state])==len(ids)
    torch.testing.assert_close(e[:,:2,1],e[:,:2,2],atol=2e-6,rtol=3e-4)
    selected=torch.tensor([[4,5,0],[1,0,5]],device=device,dtype=torch.int64)
    new_qual=torch.empty_like(qual);amount=torch.empty_like(hc);packed=torch.empty_like(selected)
    op._qualify[(g,1)](selected,labels,reps,alt,qual,new_qual,hc,amount,packed,n,f,256,num_warps=4)
    from saf3r.patch import reuse_inference_ops as rule
    context=dict(qualified=new_qual,eligible_count=amount,remove_rep=torch.zeros_like(hc,dtype=torch.bool))
    ids,_,record=rule.pack(packed,None,reps,labels,pop,3,context)
    assert ids[0].tolist()==[1,2,3]
    assert ids[1].tolist()==[0,5,6]
    assert bool((record['counts']==3).all())
    assert bool((record['native_cost_count_error']==0).all())
    if calibrated:
        emitted=torch.empty(ids.shape,device=device)
        op._emit_weights[(g,1)](ids,labels,selected,state_weights,emitted,n,f,ids.shape[1],6,256,num_warps=4)
        expected=torch.stack([state_weights[row,labels[ids[row]],selected[row,labels[ids[row]]]] for row in range(g)])
        torch.testing.assert_close(emitted,expected)
