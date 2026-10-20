"""Controlled diagnostic: sparse state evolves everywhere, Dense outputs elsewhere.

Inactive layers still maintain the selected scheme's causal state. They discard
its sparse output and execute original Dense attention. This deliberate extra
work is timed, not a deployment-speed optimization or a Dense-initialized policy.
"""
from torch import nn

STAGES=dict(all=tuple(range(24)),shallow=tuple(range(10)),middle=tuple(range(10,17)),deep=tuple(range(17,24)))


class DenseOutputWithSparseState(nn.Module):
    def __init__(self,sparse):
        super().__init__();self.sparse=sparse

    def forward(self,x,pos=None):
        self.sparse(x,pos=pos)
        return self.sparse.original(x,pos=pos)


def apply_stage_ablation(model,stage):
    active=STAGES[stage]
    for layer,block in enumerate(model.aggregator.global_blocks):
        if layer not in active:block.attn=DenseOutputWithSparseState(block.attn).eval()
    return active
