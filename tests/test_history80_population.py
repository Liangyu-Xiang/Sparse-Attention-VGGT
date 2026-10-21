import pytest
import torch
from saf3r.patch.value_region_av import residual_population_log_weights
from saf3r.patch.fused_shared_av import reference_attention


def test_residual_mass_conserves_population_and_specials_are_unit():
    labels=torch.tensor([0,0,0,1,1,2])
    ids=torch.tensor([[1,4,-1],[1,2,4]])
    b=residual_population_log_weights(torch.bincount(labels),labels,ids,2)
    torch.testing.assert_close(b.exp(),torch.tensor([[2.,1.,1.,1.,1.],[1.,1.,1.,1.,1.]]))
    torch.testing.assert_close(b[:,:3].exp().sum(1)+(ids>=0).sum(1),torch.tensor([6.,6.]))


def test_no_attention_and_no_value_boundaries():
    labels=torch.tensor([0,0,1]);empty=torch.empty((2,0),dtype=torch.int32)
    b=residual_population_log_weights(torch.tensor([2,1]),labels,empty,3)
    torch.testing.assert_close(b.exp(),torch.tensor([[2.,1.,1.,1.,1.]]).repeat(2,1))
    b=residual_population_log_weights(torch.empty(0),labels,empty,3)
    assert b.shape==(2,3) and not b.any()


def test_incoming_history_schedule_validation_and_middle_only_config():
    from saf3r.patch.value_region_av import ValueRegionAVConfig
    from scripts.calibrate_history80_middle import config_for
    with pytest.raises(ValueError):ValueRegionAVConfig(incoming_history_fractions=[.8]*23)
    with pytest.raises(ValueError):ValueRegionAVConfig(incoming_history_fractions=[float('nan')]*24)
    c=config_for(dict(alpha=.4,history=.4))
    assert c.attention_ratios[:10]==(0.,)+(.2,)*9 and c.attention_ratios[17:]==(.2,)*7
    assert c.attention_ratios[10:17]==(.4,)*7
    assert c.incoming_history_fractions[:10]==(.8,)*10 and c.incoming_history_fractions[17:]==(.8,)*7
    assert c.incoming_history_fractions[10:17]==(.4,)*7 and c.population_compensation
    assert c.retentions==(.2,)*24


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_identical_member_keys_and_values_match_explicit_dense(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    from saf3r.triton.fused_shared_av import shared_av
    torch.manual_seed(13);h,n,r,d=2,9,39,32
    dtype=torch.bfloat16 if device=='cuda' else torch.float32
    labels=torch.tensor([0,0,0,1,1,1,1],device=device)
    cluster_k=torch.randn(h,2,d,device=device,dtype=dtype)
    cluster_v=torch.randn_like(cluster_k)
    k=torch.cat((cluster_k[:,labels],torch.randn(h,2,d,device=device,dtype=dtype)),1)
    v=torch.cat((cluster_v[:,labels],torch.randn(h,2,d,device=device,dtype=dtype)),1)
    q=torch.randn(h,r,d,device=device,dtype=dtype)
    common=torch.tensor([0,3,7,8],device=device)
    ids=torch.tensor([[1,4,-1],[1,2,4]],device=device,dtype=torch.int32)
    bias=residual_population_log_weights(torch.bincount(labels),labels,ids,2)
    if device=='cuda':
        out,e,lse=shared_av(q,k.contiguous(),v.contiguous(),k[:,common].contiguous(),v[:,common].contiguous(),
            ids,2,evidence=True,common_log_weights=bias)
        cpu_ref=reference_attention(q,k,v,common,ids,2,d**-.5,bias)
        for a,b in zip((out,e,lse),cpu_ref):torch.testing.assert_close(a,b,atol=.015,rtol=.015)
    else:out,e,lse=reference_attention(q,k,v,common,ids,2,d**-.5,bias)
    dense=torch.nn.functional.scaled_dot_product_attention(q[None],k[None],v[None])[0]
    assert (out.float()-dense.float()).norm()/dense.float().norm()<(.01 if device=='cuda' else 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_weighted_gpu_tail_tiles_and_evidence_match_reference():
    from saf3r.triton.fused_shared_av import shared_av
    torch.manual_seed(19);h,n,r,d=2,177,45,32
    q=torch.randn(h,r,d,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    common=torch.arange(71,device='cuda');ids=torch.arange(80,147,device='cuda').repeat(2,1).int()
    ids[-1,-1]=-1
    bias=torch.cat((torch.rand(2,66,device='cuda')*3,torch.zeros(2,5,device='cuda')),1)
    actual=shared_av(q,k,v,k[:,common].contiguous(),v[:,common].contiguous(),ids,66,
        evidence=True,common_log_weights=bias)
    reference=reference_attention(q,k,v,common,ids,66,d**-.5,bias)
    for a,b in zip(actual,reference):torch.testing.assert_close(a,b,atol=.015,rtol=.015)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_compensated_wrapper_all24_history_and_support_mass(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    from saf3r.patch.value_region_av import ValueRegionAVAttention,ValueRegionAVState,ValueRegionAVConfig
    from saf3r.patch.region_history_attention import prepare_qkv
    from saf3r.models.official_vggt.layers.attention import Attention
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        config=ValueRegionAVConfig(population_compensation=True,capture=True,parent_only=True,
            history_fraction=.8,source_width=4,region_child_width=2,window_groups=2,
            incoming_history_fractions=(.8,)*10+(.4,)*7+(.8,)*7)
        state=ValueRegionAVState(config);state.reset(1,8,16,device)
        for layer in range(24):
            original=Attention(dim=32,num_heads=2).to(device).eval()
            wrapper=ValueRegionAVAttention(original,state,layer).eval()
            x=torch.randn(1,133,32,device=device)
            with torch.inference_mode(),torch.autocast(device,enabled=device=='cuda',dtype=torch.bfloat16):
                actual=wrapper(x)
                capture=state.captured[layer];common=capture['common'];att=capture['attention']
                labels=capture['value_labels'];pop=torch.bincount(labels)
                # Captured Attention IDs are full-token IDs, convert to patch-local.
                patch_att=(att.long()-5).masked_fill(att<0,-1)
                bias=residual_population_log_weights(pop,labels,patch_att,5)
                assert torch.isfinite(bias).all()
                torch.testing.assert_close(bias.exp().sum(1)+(att>=0).sum(1),
                    torch.full((len(att),),133.,device=device))
                q,k,v=prepare_qkv(original,x,None)
                reference=reference_attention(q.to(v.dtype),k.to(v.dtype),v,common,att,len(pop),original.scale,bias)[0]
                reference=original.proj(reference.transpose(0,1).reshape(1,133,32))
                torch.testing.assert_close(actual,reference,atol=.02 if device=='cuda' else 1e-5,rtol=.03)
            assert state.records[-1]['population_compensation']
            if layer<23:
                expected=min(state.total[layer],int(config.incoming_history_fractions[layer+1]*state.ka[layer+1]))
                assert state.records[-1]['history_out']==expected
    finally:torch.set_num_threads(previous)
