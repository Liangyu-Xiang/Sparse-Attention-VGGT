import math
import pytest
import torch
from torch import nn

from scripts.diagnose_attention_first import output_errors,masks_from_capture
from saf3r.patch.av_stage_ablation import STAGES,apply_stage_ablation,DenseOutputWithSparseState
from scripts.run_attention_first_campaign import worst_scenes


def test_full_output_error_definitions():
    dense=torch.tensor([[[1.,2.],[3.,4.],[0.,0.]]])
    identical=output_errors(dense,dense,chunk=1)
    assert identical['relative_l2']==0 and identical['cosine']==pytest.approx(1)
    assert identical['mean_token_cosine']==pytest.approx(1)
    scaled=output_errors(2*dense,dense,chunk=1)
    assert scaled['relative_l2']==pytest.approx(1) and scaled['cosine']==pytest.approx(1)
    negative=output_errors(-dense,dense)
    assert negative['relative_l2']==pytest.approx(2) and negative['cosine']==pytest.approx(-1)
    orthogonal=output_errors(torch.tensor([[0.,1.]]),torch.tensor([[1.,0.]]))
    assert orthogonal['relative_l2']==pytest.approx(math.sqrt(2)) and orthogonal['cosine']==0


def test_masks_original_coordinates_tail_and_shift():
    old=dict(common=torch.tensor([0]),attention=torch.tensor([[1,-1],[2,3]]),query_offset=1)
    mask=masks_from_capture(old,35,'old')
    assert mask[:,0].all()
    assert mask[1,1] and not mask[0,1] and mask[0,2] and mask[0,3]
    new=dict(specials=torch.tensor([0]),windows=[dict(union=torch.tensor([[1,2],[3,4]]))])
    mask=masks_from_capture(new,35,'attention_first')
    assert mask[:32,1:3].all() and mask[32:,3:5].all() and mask[:,0].all()


def test_ablation_preserves_state_but_returns_exact_dense_inactive():
    class Dense(nn.Module):
        def forward(self,x,pos=None):return x+pos
    class Sparse(nn.Module):
        def __init__(self):super().__init__();self.original=Dense();self.calls=0
        def forward(self,x,pos=None):self.calls+=1;return x-pos
    model=nn.Module();model.aggregator=nn.Module()
    model.aggregator.global_blocks=nn.ModuleList([nn.Module() for _ in range(24)])
    original=[]
    for block in model.aggregator.global_blocks:block.attn=Sparse();original.append(block.attn)
    apply_stage_ablation(model,'middle')
    for layer,block in enumerate(model.aggregator.global_blocks):
        result=block.attn(torch.tensor(3.),pos=torch.tensor(2.))
        assert result.item()==(1 if 10<=layer<=16 else 5)
        assert original[layer].calls==1
    assert set(STAGES['shallow'])|set(STAGES['middle'])|set(STAGES['deep'])==set(range(24))


def test_prior_worst_scene_selection_is_explicit_and_frozen(tmp_path):
    import json
    from scripts.run_attention_first_campaign import SCENES
    for index,scene in enumerate(SCENES):
        for method in ('dense','region'):
            p=tmp_path/method/scene/'result.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps(dict(metrics=dict(pose=dict(auc30=.9-(.01*index if method=='region' else 0)),
                recon_unposed=dict(fscore=.8)))))
    rows=worst_scenes(tmp_path)
    assert [row['scene'] for row in rows]==['stairs','redkitchen','pumpkin']
