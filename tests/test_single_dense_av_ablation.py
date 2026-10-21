from types import SimpleNamespace
import pytest
import torch
from torch import nn
from saf3r.patch.single_dense_av_ablation import SingleDenseOutputWithSparseState,apply_single_dense_ablation


def test_baseline_reproduction_gate_does_not_accept_missing_or_drifted_scores():
    from scripts.run_single_dense_history80 import reproduced
    from scripts.quality_metrics import FIELDS
    a={k:.5 for k in FIELDS}
    assert reproduced(a,dict(a))
    b=dict(a);b['f1']+=.001
    assert not reproduced(a,b)
    b=dict(a);b['auc03']=None
    assert not reproduced(a,b)


def test_shadow_runs_first_history_unchanged_by_dense_and_dense_output_returned():
    events=[]
    class Original(nn.Module):
        def forward(self,x,pos=None):events.append('dense');return x+3
    class Sparse(nn.Module):
        def __init__(self):super().__init__();self.original=Original();self.history=None
        def forward(self,x,pos=None):events.append('sparse');self.history=17;return x+1
    sparse=Sparse();wrapper=SingleDenseOutputWithSparseState(sparse,10,False)
    x=torch.zeros(1,3,4)
    torch.testing.assert_close(wrapper(x),x+3)
    assert events==['sparse','dense'] and sparse.history==17 and wrapper.calls==1


@pytest.mark.parametrize('target',[0,10,23])
def test_actual24_layer_state_chain_and_only_target_dense_output(target):
    from saf3r.patch.value_region_av import ValueRegionAVConfig,ValueRegionAVState,ValueRegionAVAttention
    from saf3r.models.official_vggt.layers.attention import Attention
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(22)
        c=ValueRegionAVConfig(parent_only=True,population_compensation=True,history_fraction=.8,
            source_width=4,region_child_width=2)
        state=ValueRegionAVState(c);state.reset(1,8,16,'cpu')
        blocks=nn.ModuleList()
        for layer in range(24):
            block=nn.Module();block.attn=ValueRegionAVAttention(Attention(32,2).eval(),state,layer).eval();blocks.append(block)
        model=SimpleNamespace(aggregator=SimpleNamespace(global_blocks=blocks))
        wrapper=apply_single_dense_ablation(model,target)
        assert sum(isinstance(b.attn,SingleDenseOutputWithSparseState) for b in blocks)==1
        x=torch.randn(1,133,32)
        with torch.inference_mode():
            for layer,block in enumerate(blocks):
                if layer==target:
                    expected=wrapper.sparse.original(x);actual=block.attn(x)
                    torch.testing.assert_close(actual,expected)
                else:actual=block.attn(x)
                assert state.previous_layer==layer and len(state.records)==layer+1
                x=x+actual
        assert wrapper.calls==1 and wrapper.comparison['all_queries']['queries']==133
        assert len(state.records)==24 and all(r['population_compensation'] for r in state.records)
        assert state.history is None
        with pytest.raises(ValueError):apply_single_dense_ablation(model,target)
    finally:torch.set_num_threads(previous)
