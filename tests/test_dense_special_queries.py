import torch
from saf3r.patch.dense_special_queries import correct_output,query_ids


def test_dense_camera_correction_matches_direct_attention_and_preserves_other_queries():
    torch.manual_seed(81)
    h,n,d=2,42,8
    q,k,v=[torch.randn(h,n,d) for _ in range(3)]
    specials=torch.tensor([0,1,2,3,4,21,22,23,24,25])
    output=torch.randn_like(q);original=output.clone()
    actual,ids=correct_output(q,k,v,output,specials,1,d**-.5)
    expected=(q[:,ids]@k.transpose(-1,-2)*d**-.5).softmax(-1)@v
    torch.testing.assert_close(actual[:,ids],expected)
    other=torch.ones(n,dtype=torch.bool);other[ids]=False
    assert torch.equal(actual[:,other],original[:,other])
    assert ids.tolist()==[0,21]


def test_disabled_dense_query_path_is_exact_noop():
    q=torch.randn(2,21,8);out=torch.randn_like(q)
    actual,ids=correct_output(q,q,q,out,torch.arange(5),0,8**-.5)
    assert actual is out and ids.numel()==0
    assert query_ids(torch.arange(5),5).tolist()==list(range(5))


def test_anchor_queries_are_dense_without_duplicate_camera_or_changed_other_rows():
    torch.manual_seed(82)
    q,k,v=[torch.randn(2,42,8) for _ in range(3)]
    output=torch.randn_like(q);before=output.clone()
    specials=torch.tensor([0,1,2,3,4,21,22,23,24,25])
    actual,ids=correct_output(q,k,v,output,specials,1,8**-.5,anchor_tokens=21)
    assert ids.tolist()==list(range(22))
    expected=(q[:,ids]@k.transpose(-1,-2)*8**-.5).softmax(-1)@v
    torch.testing.assert_close(actual[:,ids],expected)
    assert torch.equal(actual[:,22:],before[:,22:])
