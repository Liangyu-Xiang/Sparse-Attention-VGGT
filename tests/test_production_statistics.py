"""Report aggregation must not affect predictions, routing, or validation."""
import json
from pathlib import Path
import pytest,torch

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_statistics_switch_preserves_attention_outputs_and_history_exactly():
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,ReuseExploreState,ReuseResidualExploration
    from saf3r.models.official_vggt.layers.attention import Attention
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/evaluation/eth3d_repair_20261105/anchor_sketch4.json').read_text())
    config['layer_retentions']=[.2]*24
    old=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
    try:
        rows=[]
        for collect in [True,False]:
            torch.manual_seed(713)
            state=ReuseExploreState(ReuseExploreConfig(**(config|{'collect_statistics':collect})))
            state.reset(16,8,8,'cuda');x=torch.randn(1,state.n,64,device='cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for layer in range(3):x=ReuseResidualExploration(Attention(dim=64,num_heads=2).cuda().eval(),state,layer).eval()(x)
            assert len(state.records)==(3 if collect else 0)
            rows.append((x.clone(),state.history_ids.clone(),state.previous_grid.clone()))
        for a,b in zip(*rows):torch.testing.assert_close(a,b,rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(old)
