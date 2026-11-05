import pytest,torch
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA absent')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('shape',[(1,4,131,64),(3,2,37,64)])
def test_fused_rope_matches_intermediate_rounding_and_tail(dtype,shape):
    from saf3r.models.official_vggt.layers.rope import RotaryPositionEmbedding2D
    from saf3r.triton.rope_fast import apply
    torch.manual_seed(51)
    b,h,n,d=shape;x=torch.randn(b,h,n,d,device='cuda',dtype=dtype)
    pos=torch.randint(31,(b,n,2),device='cuda');pos[:,0]=0
    rope=RotaryPositionEmbedding2D();reference=rope(x,pos)
    cos,sin=rope._compute_frequency_components(d//2,int(pos.max())+1,x.device,x.dtype)
    result=apply(x,pos,cos,sin)
    assert torch.equal(result,reference)


def test_noncontiguous_qkv_view():
    from saf3r.models.official_vggt.layers.rope import RotaryPositionEmbedding2D
    from saf3r.triton.rope_fast import apply
    x=torch.randn(2,29,3,4,64,device='cuda',dtype=torch.bfloat16)[:,:,0].permute(0,2,1,3)
    pos=torch.randint(11,(2,29,2),device='cuda');rope=RotaryPositionEmbedding2D();expected=rope(x,pos)
    cos,sin=rope._compute_frequency_components(32,11,x.device,x.dtype)
    assert torch.equal(apply(x,pos,cos,sin),expected)
