"""One Dense OUTPUT intervention while preserving the original sparse state chain.

The shadow sparse operator still runs and sends its own history. Dense weights
never initialize clustering or history. This is a diagnostic, not a fast path.
"""
from torch import nn


class SingleDenseOutputWithSparseState(nn.Module):
    def __init__(self,sparse,layer,measure=True):
        super().__init__();self.sparse=sparse;self.layer=layer;self.measure=measure
        self.calls=0;self.comparison=None

    def forward(self,x,pos=None):
        sparse_output=self.sparse(x,pos=pos)
        # This call must not replace or modify the shadow's labels/history.
        dense_output=self.sparse.original(x,pos=pos)
        if self.measure:
            from scripts.compare_deep_head_rollouts import head_output_metrics
            state=self.sparse.state
            _,self.comparison=head_output_metrics(sparse_output.detach(),dense_output.detach(),
                state.height*state.width+5,heads=1)
        self.calls+=1
        return dense_output


def apply_single_dense_ablation(model,layer,measure=True):
    from .value_region_av import ValueRegionAVAttention
    if not 0<=layer<24:raise ValueError('One zero-based global layer0–23 required')
    blocks=model.aggregator.global_blocks
    if len(blocks)!=24 or not all(isinstance(b.attn,ValueRegionAVAttention) for b in blocks):
        raise ValueError('Expected the original24-layer ValueRegion sparse scheme')
    wrapper=SingleDenseOutputWithSparseState(blocks[layer].attn,layer,measure).eval()
    blocks[layer].attn=wrapper
    return wrapper
