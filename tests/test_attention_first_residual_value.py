"""Independent Attention-first selector: causal allocation and coverage contracts."""
import math
import pytest
import torch

from saf3r.patch.attention_first_residual_value import (
    AttentionFirstConfig, AttentionFirstState, allocate_budget, choose_attention,
    residual_value, reference_union, compress_history, AttentionFirstResidualValue,
    install_attention_first_residual_value,
)


def example(device='cpu'):
    labels=torch.arange(12,device=device).repeat_interleave(2)
    x=torch.stack((torch.arange(24,device=device).float(),torch.ones(24,device=device)),1)
    reps=torch.arange(0,24,2,device=device)
    sums=torch.zeros(12,2,device=device).index_add_(0,labels,x)
    sizes=torch.bincount(labels,minlength=12)
    return x,labels,reps,sums,sizes


def test_budget_is_causal_not_layer_table():
    cfg=AttentionFirstConfig()
    assert allocate_budget(20,0,0,0,None,cfg)[:2]==(0,20)
    assert allocate_budget(20,1,0,0,None,cfg)[:2]==(19,1)
    base=allocate_budget(100,.5,0,0,(.5,0),cfg)[0]
    assert allocate_budget(100,.5,.8,0,(.5,0),cfg)[0]>base
    assert allocate_budget(100,.5,0,1,(.5,1),cfg)[0]<base
    # Flat current probabilities do not inherit a forced minimum A budget.
    assert allocate_budget(20,0,0,0,(1,0),cfg)[:2]==(0,20)
    assert allocate_budget(1,1,1,0,(1,0),cfg)[:2]==(0,1)
    with pytest.raises(ValueError):allocate_budget(20,float('nan'),0,0,None,cfg)


def test_paired_speed_requires_identical_input_hardware_and_protocol():
    from scripts.evaluate_attention_first_residual_value import matched_speedup
    common=dict(dataset='7scenes',scene='chess',frames=2,input={'frames':[0,9]},
                hardware={'gpu':'test'},protocol={'warmup':1,'repeats':3})
    dense=dict(common,mode='dense',median_seconds=2.)
    sparse=dict(common,mode='attention_first',median_seconds=1.)
    assert matched_speedup(sparse,dense)==2.
    with pytest.raises(ValueError):matched_speedup(dict(sparse,frames=3),dense)
    with pytest.raises(ValueError):matched_speedup(sparse,dict(dense,mode='other'))


@pytest.mark.parametrize('ka',[0,3,7,12])
def test_attention_history_first_exact_unique_budget(ka):
    table=torch.arange(12).reshape(3,4)
    parents=torch.arange(12)//4
    scores=torch.tensor([[.1,.8,.1],[.9,.05,.05]])
    history=torch.tensor([[1,5],[6,9]],dtype=torch.int32) if ka>=3 else None
    out,seen=choose_attention(scores,table,torch.tensor([4,4,4]),parents,history,ka,12)
    assert out.shape==(2,ka)
    for g in range(2):
        assert len(out[g].unique())==ka
        assert seen[g].sum()==ka
        if history is not None:assert torch.equal(out[g,:2],history[g])
    if ka==3:assert out[0,2]==4  # Explore after retained [1,5], without V exclusion.


def test_boundary_uniform_sampling_and_padding():
    out,_=choose_attention(torch.tensor([[1.,0.]]),torch.tensor([[0,1,2,3,4],[5,6,-1,-1,-1]]),
        torch.tensor([5,2]),torch.tensor([0,0,0,0,0,1,1]),None,3,7)
    assert out.tolist()==[[0,2,4]]
    with pytest.raises(ValueError):
        choose_attention(torch.ones(1,1),torch.arange(3)[None],torch.tensor([3]),
            torch.zeros(3,dtype=torch.long),torch.tensor([[0,1]]),1,3)


@pytest.mark.parametrize('ka,kv',[(0,12),(1,11),(4,8),(7,1),(8,4)])
def test_residual_covers_structures_not_only_deduplicates_tokens(ka,kv):
    x,labels,reps,sums,sizes=example()
    attention=torch.arange(ka).reshape(1,ka).repeat(2,1)
    out,cert=residual_value(x,labels,reps,sums,sizes,attention,kv)
    for g in range(2):
        assert len(out[g].unique())==kv
        assert set(out[g].tolist()).isdisjoint(attention[g].tolist())
        covered=set(labels[attention[g]].tolist())
        assert set(labels[out[g].long()].tolist()).isdisjoint(covered)
        assert (cert['populations'][g]>0).all()
        assert cert['populations'][g].sum()==2*(12-len(covered))
        untouched=cert['uncovered'][g]
        assert set(cert['merged_labels'][g,untouched].tolist())==set(range(kv))


def test_residual_mean_uses_population_weights_and_real_reps():
    x=torch.tensor([[0.]]*8+[[8.],[20.]])
    labels=torch.tensor([0]*8+[1,2]);reps=torch.tensor([0,8,9])
    sums=torch.tensor([[0.],[8.],[20.]]);sizes=torch.tensor([8,1,1])
    out,cert=residual_value(x,labels,reps,sums,sizes,torch.empty(1,0,dtype=torch.long),1)
    # Full-member mean=2.8 => representative0; unweighted mean9.33 would select8.
    assert out.item()==0
    assert cert['populations'].item()==10


def test_unified_softmax_and_history_tail_excludes_specials():
    torch.manual_seed(42)
    q=torch.randn(2,35,8);k=torch.randn(2,150,8);v=torch.randn_like(k)
    specials=torch.tensor([0,2,4]);ids=torch.arange(5,135).repeat(2,1)
    out,e,lse=reference_union(q,k,v,specials,ids,.3)
    expected=torch.empty_like(out)
    for row in range(35):
        keys=torch.cat((specials,ids[row//32]))
        z=(q[:,row:row+1]@k[:,keys].transpose(-1,-2))*.3
        expected[:,row:row+1]=z.softmax(-1)@v[:,keys]
    torch.testing.assert_close(out,expected)
    history=compress_history(e,lse,ids,len(specials),80)
    assert history.shape==(2,80)
    for row in history:
        assert len(row.unique())==80
        assert set(row.tolist()).isdisjoint(specials.tolist())
        assert ((row>=5)&(row<135)).all()


def test_gate_population_prior_flat_q_and_reset():
    s=AttentionFirstState(AttentionFirstConfig(retentions=(.5,)*24))
    s.reset(1,4,4,'cpu')
    features=torch.randn(16,4);s.fine.update(features,0)
    q=torch.zeros(2,s.n,4);pooled=torch.randn(2,s.parent_count,4)
    a,b,signals=s.gate(q,pooled,features,0,.5)
    assert a==0 and b==s.budgets[0]
    assert signals['concentration']<1e-6
    s.history=torch.ones(math.ceil(s.n/32),2,dtype=torch.int32)
    s.previous_layer=3
    s.reset(2,4,4,'cpu')
    assert s.history is None and s.previous_layer==-1 and s.previous_signals is None
    assert len(s.records)==0


def test_all_layers_causal_history_budget_shrink_and_reset():
    from torch import nn
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads=2;self.head_dim=8;self.scale=8**-.5
            self.qkv=nn.Linear(16,48);self.q_norm=nn.Identity();self.k_norm=nn.Identity()
            self.rope=None;self.proj=nn.Identity();self.proj_drop=nn.Identity()
    torch.manual_seed(23)
    cfg=AttentionFirstConfig(retentions=(.4,)*10+(.2,)*14,source_width=2,capture=True,window_groups=2)
    state=AttentionFirstState(cfg);state.reset(2,8,8,'cpu')
    original=Attention().eval()
    with torch.inference_mode():
        for layer in range(24):
            module=AttentionFirstResidualValue(original,state,layer).eval()
            output=module(torch.randn(1,state.n,16)*3)
            assert output.shape==(1,state.n,16) and torch.isfinite(output).all()
            record=state.records[-1]
            assert record['attention_keys']+record['value_keys']==state.budgets[layer]
            assert record['history_used']<=math.floor(.8*record['attention_keys'])
            for window in state.captured[layer]['windows']:
                for a,v in zip(window['attention'],window['value']):
                    assert set(a.tolist()).isdisjoint(v.tolist())
                for ids in window['union']:
                    assert len(ids.unique())==state.budgets[layer]
                    assert set(ids.tolist()).isdisjoint(state.specials.tolist())
    assert state.history is None and state.previous_layer==23
    assert any(row['attention_keys'] for row in state.records)
    with pytest.raises(ValueError):module(torch.zeros(1,state.n,16))
    state.reset(2,8,8,'cpu')
    assert state.previous_layer==-1 and not state.captured
    fake=nn.Module();fake.fused_shared_av_state=object()
    with pytest.raises(ValueError):install_attention_first_residual_value(fake)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires idle CUDA GPU')
def test_cuda_selector_and_residual_parity():
    torch.manual_seed(7)
    table=torch.arange(24).reshape(3,8)
    scores=torch.rand(2,3);hist=torch.tensor([[2,4],[11,16]],dtype=torch.int32)
    args=(scores,table,torch.tensor([8,8,8]),torch.arange(24)//8,hist,13,24)
    cpu,_=choose_attention(*args)
    gpu,_=choose_attention(*(a.cuda() if torch.is_tensor(a) else a for a in args))
    assert torch.equal(cpu,gpu.cpu())
    args2=(*example(),cpu,2)
    expected,cert=residual_value(*args2)
    actual,got=residual_value(*(a.cuda() if torch.is_tensor(a) else a for a in args2))
    assert torch.equal(expected,actual.cpu())
    torch.testing.assert_close(cert['populations'],got['populations'].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Requires idle CUDA GPU')
def test_bf16_final_kernel_special_union_tail():
    from saf3r.triton.fused_shared_av import shared_av
    torch.manual_seed(11)
    q=torch.randn(2,35,64,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(2,160,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    specials=torch.tensor([0,2,4],device='cuda');ids=torch.arange(5,135,device='cuda').repeat(2,1).int()
    expected,_,_=reference_union(q,k,v,specials,ids,.125)
    actual,e,lse=shared_av(q,k,v,k[:,specials].contiguous(),v[:,specials].contiguous(),ids,0,scale=.125,evidence=True)
    torch.testing.assert_close(actual,expected,atol=.015,rtol=.025)
    history=compress_history(e,lse,ids,len(specials),80)
    assert all(len(row.unique())==80 for row in history)
