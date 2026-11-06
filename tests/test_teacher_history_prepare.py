import types,torch,pytest
from saf3r.patch.head_value_sketch import projection,features
from saf3r.patch.teacher_guided_history import prepare_keys
from saf3r.triton.teacher_history_prepare import prepare
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('n',[1024,1027,1088])
def test_fused_key_preparation(n):
 torch.manual_seed(131);h,d=16,64;k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);special=torch.tensor([0,1,2,3,4],device='cuda');patch=torch.arange(5,n,device='cuda');mat=projection(h,d,4,'cuda');s=types.SimpleNamespace(config=types.SimpleNamespace(history_key_block_size=64,teacher_history_samples=8,teacher_history_fused_prepare=False),patches=patch,specials=special,_value_sketch_matrix=mat)
 ref=prepare_keys(s,k);bank,anchors,z=prepare(k,patch,special,mat);torch.testing.assert_close(bank,ref['bank'],rtol=0,atol=0)
 valid=ref['valid'];torch.testing.assert_close(anchors[:,valid],ref['anchor_keys'][:,valid],rtol=0,atol=0);torch.testing.assert_close(z,features(k,mat)[patch],rtol=1e-6,atol=1e-6)
