import math
import copy
import pytest
import torch
from saf3r.patch.adaptive_block_value import (block_mean, select_block_mask, audit_contributions,
    audit_error, audit_mask_error, ranked_mask, match_history, audited_decisions, AdaptiveBlockState,
    AdaptiveBlockValueAttention, install_adaptive_block_value)
from saf3r.models.official_vggt.layers.attention import Attention
from saf3r.patch.value_clustering_gpu import stable_cluster_mean


@pytest.mark.parametrize('device',['cpu','cuda'] if torch.cuda.is_available() else ['cpu'])
def test_block_mask_is_unique_exact_and_covers_unselected_value_regions(device):
    torch.manual_seed(5)
    p = torch.rand(7,20,device=device)
    order = torch.randperm(20,device=device)
    history = torch.stack([torch.randperm(20,device=device)[:6] for _ in range(7)])
    for a in [0,1,4,9]:
        ix,origin = select_block_mask(p,order,9,a,history)
        assert all(len(r.unique())==9 for r in ix)
        assert ((origin>0).sum(-1)==a).all()
        assert ((origin==1).sum(-1)<=a//2).all()


@pytest.mark.parametrize('device',['cpu','cuda'] if torch.cuda.is_available() else ['cpu'])
def test_segmented_centers_are_repeatable_and_match_reference(device):
    torch.manual_seed(10)
    x=torch.randn(2048,64,device=device)
    labels=torch.randint(0,31,(2048,),device=device)
    expected=torch.stack([x[labels==k].mean(0) for k in range(31)])
    first=stable_cluster_mean(x,labels,31)
    torch.testing.assert_close(first,expected,atol=1e-6,rtol=1e-5)
    for _ in range(5):
        assert torch.equal(stable_cluster_mean(x,labels,31),first)


def test_exact_probe_contributions_reproduce_mask_softmax():
    torch.manual_seed(7)
    q=torch.randn(2,13,8);kp=torch.randn(2,17,8);vp=torch.randn_like(kp)
    ks=torch.randn(2,3,8);vs=torch.randn_like(ks)
    probe=torch.tensor([1,5,9])
    c=audit_contributions(q,kp,vp,ks,vs,probe,4,.3)
    ids=torch.tensor([[0,2],[1,3],[0,4]])
    for head in range(2):
        nmse,_=audit_error(c,head,ids)
        actual=[]
        for j,blocks in enumerate(ids):
            ix=(blocks[:,None]*4+torch.arange(4)).flatten();ix=ix[ix<17]
            keys=torch.cat((kp[head,ix],ks[head]));vals=torch.cat((vp[head,ix],vs[head]))
            actual.append((q[head,probe[j]]@keys.T*.3).softmax(-1)@vals)
        actual=torch.stack(actual)
        expected=(actual-c[-1][head]).square().sum()/c[-1][head].square().sum()
        torch.testing.assert_close(nmse,expected)


def test_batched_auditor_matches_production_selection():
    torch.manual_seed(35)
    scores=torch.rand(3,7,20)
    order=torch.randperm(20)
    history=torch.stack([torch.stack([torch.randperm(20)[:6] for _ in range(7)]) for _ in range(3)])
    ranks=scores.argsort(-1,descending=True).argsort(-1)
    a=torch.tensor([0,4,9])
    for use_old in [False,True]:
        hm=torch.zeros_like(scores,dtype=torch.bool).scatter_(-1,history,True) if use_old else None
        mask=ranked_mask(ranks,order,9,a,hm)
        for h in range(3):
            ids,_=select_block_mask(scores[h],order,9,int(a[h]),history[h] if use_old else None)
            expected=torch.zeros_like(mask[h]).scatter_(-1,ids,True)
            assert torch.equal(mask[h],expected)


def test_vectorized_errors_match_single_head_auditor():
    torch.manual_seed(36)
    q=torch.randn(3,13,8);kp=torch.randn(3,19,8);vp=torch.randn_like(kp)
    ks=torch.randn(3,3,8);vs=torch.randn_like(ks)
    c=audit_contributions(q,kp,vp,ks,vs,torch.tensor([1,5,9]),4,.3)
    scores=torch.rand(3,3,5)
    masks=ranked_mask(scores.argsort(-1,descending=True).argsort(-1),torch.arange(5),3,torch.tensor([0,1,3]))
    nmse,p90=audit_mask_error(c,masks)
    for head in range(3):
        ids=masks[head].nonzero(as_tuple=True)[1].reshape(3,3)
        e,t=audit_error(c,head,ids)
        torch.testing.assert_close(nmse[head],e)
        torch.testing.assert_close(p90[head],t)


def test_head_matching_uses_current_relevance_not_head_number():
    histories=[torch.tensor([[0,1],[0,1]]),torch.tensor([[2,3],[2,3]])]
    p=torch.tensor([[[.01,.01,.49,.49],[.01,.01,.49,.49]],
                    [[.49,.49,.01,.01],[.49,.49,.01,.01]]])
    matching,enrichment=match_history(p,histories,torch.tensor([0,1]))
    assert matching==[1,0]
    assert all(e>1 for e in enrichment)


def test_budget_guard_expands_instead_of_claiming_fixed_ten_percent():
    state=AdaptiveBlockState(audit_queries=4,audit_nmse=1e-9,audit_p90=1e-9,
                             budget_candidates=(.1,.5,1.))
    # Omitted blocks all have unique, nonzero contributions: any strict subset
    # fails the very tight guard. The final fallback must be exactly dense.
    q=torch.zeros(2,8,8); k=torch.zeros(2,640,8)
    v=torch.arange(640*8).float().reshape(1,640,8).expand(2,-1,-1)
    c=audit_contributions(q,k,v,k[:,:0],v[:,:0],torch.tensor([2,6]),64,.3)
    routing=torch.rand(2,2,10)
    stats={'concentration':torch.tensor([.2,.8])}
    result=audited_decisions(state,routing,torch.arange(10),stats,c,torch.tensor([0,1]),
                             [0,1],[0.,0.],640,0,10,False)
    assert all(r['dense_fallback'] and r['rho']==1 for r in result)
    state.audit_guard=False
    fixed=audited_decisions(state,routing,torch.arange(10),stats,c,torch.tensor([0,1]),
                            [0,1],[0.,0.],640,0,10,False)
    assert all(not r['dense_fallback'] and r['blocks']==1 and not r['audit_passed'] for r in fixed)


def test_value_only_layer_keeps_validated_discovery_history(monkeypatch):
    import saf3r.patch.adaptive_block_value as module
    state=AdaptiveBlockState(retention=.5,audit_guard=False,query_block=32,key_block=32,
                             audit_queries=4,validation_queries=4)
    state.reset(1,1,100)
    original=Attention(dim=8,num_heads=2).eval()
    def value_only(*args):
        return [dict(rho=.5,blocks=2,a_blocks=0,alpha=0.,use_history=False,
                     audit_nmse=0.,audit_p90=0.,dense_fallback=False,audit_passed=True) for _ in range(2)]
    monkeypatch.setattr(module,'audited_decisions',value_only)
    x=torch.randn(1,105,8)
    with torch.inference_mode():
        AdaptiveBlockValueAttention(original,state,0).eval()(x)
        assert all(hist.shape[-1]==0 for hist in state.history)
        for layer in [1,2]:
            AdaptiveBlockValueAttention(original,state,layer).eval()(x)
            assert all(hist.shape[-1]==2 for hist in state.history)
            assert all(count==2 for count in state.records[-1]['history_blocks_per_head'])
            assert len(state.records[-1]['independent_validation']['output_nmse'])==2


def test_full_retention_recovers_dense_and_all_24_layers_reset(monkeypatch):
    from test_layer_value_attention import ToyAggregator
    model=torch.nn.Module();model.aggregator=ToyAggregator().eval()
    reference=copy.deepcopy(model)
    state=install_adaptive_block_value(model,retention=1.,audit_queries=4)
    sdpa=torch.nn.functional.scaled_dot_product_attention
    def batch_sdpa(q,k,v,**kwargs):
        assert q.ndim==k.ndim==v.ndim==4
        return sdpa(q,k,v,**kwargs)
    monkeypatch.setattr(torch.nn.functional,'scaled_dot_product_attention',batch_sdpa)
    with torch.inference_mode():
        for frames in [2,1]:
            images=torch.randn(1,frames,3,28,28)
            torch.testing.assert_close(model.aggregator(images),reference.aggregator(images))
            assert len(state.records)==24
    assert all(c['dense_fallback'] for c in state.records[-1]['decisions'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
@pytest.mark.parametrize('head_dim',[32,64])
@pytest.mark.parametrize('query_block',[32,64])
@pytest.mark.parametrize('key_block',[32,64,128])
def test_tensor_core_block_attention_exact_mask_and_same_pass_evidence(dtype,head_dim,query_block,key_block):
    from saf3r.triton.adaptive_block_attention import block_attention
    torch.manual_seed(8)
    q=torch.randn(3,137,head_dim,device='cuda',dtype=dtype)
    npatch=2*key_block+43 if key_block>=64 else 2*key_block+19
    kp=torch.randn(3,npatch,head_dim,device='cuda',dtype=dtype);vp=torch.randn_like(kp)
    ks=torch.randn(3,7,head_dim,device='cuda',dtype=dtype);vs=torch.randn_like(ks)
    heads=torch.tensor([0,2],device='cuda',dtype=torch.int32)
    counts=torch.tensor([2,3],device='cuda',dtype=torch.int32)
    nq=math.ceil(137/query_block)
    patterns=[[0,2,-1],[1,2,-1],[0,1,-1]]
    masks=torch.tensor([[patterns[i%3] for i in range(nq)],[[0,1,2]]*nq],device='cuda',dtype=torch.int32)
    out,evidence=block_attention(q,kp,vp,ks,vs,masks,counts,heads,query_block=query_block,key_block=key_block,scale=.2)
    for slot,head in enumerate([0,2]):
        for block in range(nq):
            qr=slice(block*query_block,min((block+1)*query_block,137))
            ids=masks[slot,block,:int(counts[slot])].long()
            ix=(ids[:,None]*key_block+torch.arange(key_block,device='cuda')).flatten();ix=ix[ix<npatch]
            keys=torch.cat((kp[head,ix],ks[head]));values=torch.cat((vp[head,ix],vs[head]))
            logits=q[head,qr].float()@keys.float().T*.2
            expected=(logits.softmax(-1)@values.float()).to(dtype)
            torch.testing.assert_close(out[head,qr],expected,rtol=.02,atol=.006)
            for j,bid in enumerate(ids):
                span=(ix//key_block==bid)
                lse=logits[:,:len(ix)][:,span].logsumexp(-1).mean()
                torch.testing.assert_close(evidence[slot,block,j],lse,rtol=1e-4,atol=1e-4)
