import json
from pathlib import Path
import torch,pytest


def test_live_anchor_support_budget_camera_and_history_reset(monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from saf3r.patch.reuse_residual_exploration import ReuseExploreConfig,ReuseExploreState,ReuseResidualExploration
    from saf3r.models.official_vggt.layers.attention import Attention
    from saf3r.triton import reuse_direct_sampled_probability as kernel
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'configs/evaluation/integrated_20261105/main.json').read_text())
    c.update(protect_first_frame=True,layer_retentions=[.16,.20]+[.16]*22)
    state=ReuseExploreState(ReuseExploreConfig(**c));state.reset(32,8,8,'cuda');original=kernel.direct_sparse_attention_sampled
    calls=[]
    def check(q,k,v,ids,specials,*args,**kw):
        kernel.validate_support_indices(ids,specials,state.n)
        assert set(range(69))<=set(specials.cpu().tolist())
        assert bool(((ids<0)|(ids>=74)).all())
        assert ids.shape[1]+len(specials)<=int(state.n*state.config.layer_retentions[state.previous_layer+1])
        calls.append(len(ids));return original(q,k,v,ids,specials,*args,**kw)
    monkeypatch.setattr(kernel,'direct_sparse_attention_sampled',check)
    x=torch.randn(1,state.n,64,device='cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for layer in range(3):
            x=ReuseResidualExploration(Attention(dim=64,num_heads=2).cuda().eval(),state,layer).eval()(x)
            assert torch.isfinite(x).all()
            row=state.records[-1]
            assert row['dense_special_query_count']==32
            assert row['protected_patch_keys']==64 and row['semantic_special_keys']==160
            assert state.history_ids.shape[1]==state.ready_edges.shape[-1]==state.budget
    assert calls
    state.reset(33,8,8,'cuda')
    assert state.history_ids is None and len(state.semantic_specials)==165
