import pytest
import torch
from scripts.compare_deep_head_rollouts import head_output_metrics,original_attention
from types import SimpleNamespace


@pytest.mark.parametrize('chunk',[1,7,100])
def test_full_head_metrics_streaming_and_query_scopes(chunk):
    torch.manual_seed(0)
    reference=torch.randn(1,18,8)
    actual=reference.clone().reshape(1,18,2,4)
    actual[:,:,0]*=2
    actual[:,:,1]*=-1
    metrics,aggregate=head_output_metrics(actual.reshape(1,18,8),reference,9,heads=2,chunk=chunk)
    for scope,count in [('all_queries',18),('patch_queries',8),('special_queries',10)]:
        a=actual[0];b=reference.reshape(18,2,4)
        mask=torch.ones(18,dtype=torch.bool) if scope=='all_queries' else torch.arange(18)%9>=5
        if scope=='special_queries':mask=~mask
        for h in (0,1):
            assert metrics[h][scope]['queries']==count
            assert metrics[h][scope]['relative_l2_frobenius']==pytest.approx(1 if h==0 else 2,abs=1e-6)
            assert metrics[h][scope]['mean_per_query_cosine']==pytest.approx(1 if h==0 else -1,abs=1e-6)
        expected=(a[mask]-b[mask]).norm()/b[mask].norm()
        assert aggregate[scope]['relative_l2_frobenius']==pytest.approx(float(expected),abs=1e-6)
        assert aggregate[scope]['queries']==count*2


def test_zero_reference_is_not_invented_similarity():
    zero=torch.zeros(1,9,8)
    metrics,_=head_output_metrics(zero,zero,9,heads=2)
    assert metrics[0]['all_queries']['relative_l2_frobenius'] is None
    assert metrics[0]['all_queries']['mean_per_query_cosine'] is None


def test_unwrap_supported_actual_projection_modules():
    original=object()
    assert original_attention(original) is original
    assert original_attention(SimpleNamespace(original=original)) is original
    assert original_attention(SimpleNamespace(orig_module=original)) is original


def test_projection_input_matches_true_individual_heads_without_reordering():
    av=torch.arange(2*9*4).reshape(1,2,9,4).float()+1
    projected_input=av.transpose(1,2).reshape(1,9,8)
    metrics,_=head_output_metrics(projected_input,projected_input.clone(),9,heads=2)
    assert all(m['all_queries']['relative_l2_frobenius']==0 for m in metrics)
    torch.testing.assert_close(projected_input.reshape(1,9,2,4).permute(0,2,1,3),av)
