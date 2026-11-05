import json
from dataclasses import replace
from pathlib import Path
import pytest,torch
from torch import nn
from saf3r.patch.native_prefix_reuse import prefix_density

def test_prefix_density_counts_unique_keys_and_all_heads():
    assert prefix_density([dict(mode='all_to_first',head_ids=list(range(16)))],14,64)==1/14
    assert prefix_density([dict(mode='global_to_frame',head_ids=list(range(16)))],14,64)==1/14
    groups=[dict(mode='stride',stride=128,include_as=True,head_ids=list(range(16)))]
    n=14*69;ids=set(range(0,n,128))|set(range(69))|{f*69+i for f in range(14) for i in range(5)}
    assert prefix_density(groups,14,64)==len(ids)/n

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_prefix_tail_order_reset_history_cleanup_and_statistics_equivalence():
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,install_reuse_residual_exploration
    from saf3r.models.official_vggt.layers.attention import Attention
    class Block(nn.Module):
        def __init__(self):super().__init__();self.attn=Attention(dim=256,num_heads=16)
    class Aggregator(nn.Module):
        def __init__(self):
            super().__init__();self.global_blocks=nn.ModuleList([Block() for _ in range(24)]);self.register_buffer('tokens',torch.randn(1,14*69,256))
        def forward(self,images):
            x=self.tokens.clone()
            for block in self.global_blocks:x=x+.01*block.attn(x)
            return x
    class Model(nn.Module):
        def __init__(self):super().__init__();self.aggregator=Aggregator()
        def forward(self,images):return self.aggregator(images[None])
    torch.manual_seed(714);old=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
    try:
        c=json.loads((Path(__file__).resolve().parents[1]/'configs/evaluation/eth3d_repair_20261105/native9_sketch4.json').read_text())
        model=Model().cuda().eval();state=install_reuse_residual_exploration(model,ReuseExploreConfig(**c));images=torch.zeros(14,3,112,112,device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):reference=model(images)
        assert [r['layer'] for r in state.records]==list(range(24))
        assert all(r['native_prefix'] for r in state.records[:9])
        assert all(not r['native_prefix'] for r in state.records[9:])
        assert state.previous_layer==14 and state.ready_edges is None
        state.config=replace(state.config,collect_statistics=False)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):actual=model(images)
        assert state.records==[] and state.ready_edges is None
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(old)
