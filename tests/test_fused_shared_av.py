import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('attention_count',[0,7,65])
def test_one_softmax_tail_tiles_and_same_pass_patch_mass(attention_count):
    from saf3r.triton.fused_shared_av import shared_av
    torch.manual_seed(123)
    h,n,r,d=2,167,45,32
    q=torch.randn(h,r+5,d,device='cuda',dtype=torch.bfloat16)[:,:r]
    k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    value_count=17;common=torch.arange(22,device='cuda')
    ids=torch.stack([torch.arange(30,30+attention_count,device='cuda'),torch.arange(60,60+attention_count,device='cuda')]).int()
    if attention_count:ids[-1,-1]=-1
    out,e,lse=shared_av(q,k,v,k[:,common].contiguous(),v[:,common].contiguous(),ids,value_count,evidence=True)
    reference=[]
    for row in range(r):
        support=torch.cat((common,ids[row//32].long()))
        valid=support>=0;safe=support.clamp_min(0)
        z=(q[:,row,None].float()*k[:,safe].float()).sum(-1)*d**-.5
        p=z.masked_fill(~valid[None],-torch.inf).softmax(-1)
        reference.append((p[:,:,None]*v[:,safe].float()).sum(1))
        # History evidence excludes all special keys, but uses the full denominator.
        for slot in range(e.shape[-1]):
            if slot==0:
                expected=p[:,:value_count].sum(1)
            else:
                start=22+(slot-1)*64;expected=p[:,start:start+64].sum(1)
            torch.testing.assert_close((e[:,row,slot]-lse[:,row]).exp(),expected,atol=3e-4,rtol=.01)
    reference=torch.stack(reference,1)
    assert (out.float()-reference).norm()/reference.norm()<.01
    assert torch.isfinite(out).all()
