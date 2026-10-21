from types import SimpleNamespace
import json
import pytest
import torch
from torch import nn
from scripts.evaluate_full_global_output_comparison import FullOutputCapture,history80_config,METHODS
from scripts.run_full_global_output_comparison import dataset_summary


def fake_scene(error,cosine,quality=True):
    methods={}
    metric=dict(relative_l2_frobenius=error,mean_per_query_cosine=cosine,flattened_cosine=cosine)
    scopes={s:dict(metric) for s in ('all_queries','patch_queries','special_queries')}
    for mode in METHODS:
        comparisons={c:scopes for c in ('rollout_vs_dense','same_input_vs_dense')}
        methods[mode]=dict(layers=[dict(layer=l,module_output=comparisons,preprojection=comparisons,
            heads=[dict(head=h,**comparisons) for h in range(16)]) for l in range(24)])
    return dict(outputs_complete=True,quality_complete=quality,methods=methods,
        quality={m:dict(quality=dict(values={k:error for k in ('auc03','auc30','acc','comp','cd','f1')})) for m in METHODS})


def test_dataset_macro_is_equal_scene_not_energy_or_token_weighted():
    x=dataset_summary(['a','b'],{'a':fake_scene(1,.9),'b':fake_scene(3,.7)})
    assert x['outputs_complete'] and x['quality_complete']
    assert all(r['relative_l2_frobenius']==2 for r in x['layer_macro'])
    assert all(r['mean_per_query_cosine']==pytest.approx(.8) for r in x['head_macro'])
    assert x['quality_macro']['history80']['f1']==2


def test_partial_macro_and_quality_missing_are_explicit():
    x=dataset_summary(['a','b'],{'a':fake_scene(1,.9,quality=False)})
    assert not x['outputs_complete'] and not x['quality_complete']
    assert x['missing_outputs']==['b'] and x['missing_quality']==['a','b']
    assert x['quality_macro']=={} and x['output_scenes']==['a']


def test_full_saf3r_and_history80_configs_frozen():
    import yaml
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    c=history80_config()
    assert c.parent_only and c.history_fraction==.8 and not c.parent_descriptors
    assert c.attention_ratios==(0.,)+(.2,)*9+(.8,)*7+(.2,)*7
    cfg=yaml.safe_load((root/'configs/evaluation/eval_saf3r_vggt.yaml').read_text())['model']
    assert cfg['patch_module']['topk']==4 and cfg['patch_module']['lazy_dino_topk']
    p=json.loads((root/cfg['sparse_config_path']).read_text())
    assert set(p)=={str(i) for i in range(24)}
    for groups in p.values():assert sorted(h for g in groups for h in g['head_ids'])==list(range(16))


def test_consolidation_rejects_incomplete_retry_without_creating_report(tmp_path):
    from scripts.consolidate_full_global_output_comparison import consolidate
    original=tmp_path/'original';original.mkdir()
    (original/'manifest.json').write_text(json.dumps(dict(scenes={'eth3d':['a']})))
    retry=tmp_path/'retry';scene=retry/'eth3d/a';scene.mkdir(parents=True)
    (scene/'comparison.json').write_text(json.dumps(dict(outputs_complete=True,quality_complete=False)))
    destination=tmp_path/'final'
    with pytest.raises(ValueError,match='Incomplete paired'):
        consolidate(original,retry,destination)
    assert not destination.exists()


def test_capture_all24_av_and_postprojection_are_different_positions(tmp_path):
    class Attention(nn.Module):
        def __init__(self):super().__init__();self.proj=nn.Identity()
        def forward(self,x):return self.proj(x)*2
    blocks=nn.ModuleList()
    for _ in range(24):
        b=nn.Module();b.attn=Attention();blocks.append(b)
    model=SimpleNamespace(aggregator=SimpleNamespace(global_blocks=blocks))
    out=tmp_path/'dense';capture=FullOutputCapture(model,'dense',out,out,2,1,1)
    x=torch.ones(1,12,1024)
    for l,b in enumerate(blocks):
        x=b.attn(x)
        av=torch.load(out/'dense_outputs'/f'AV_L{l:02d}.pt',weights_only=True)
        module=torch.load(out/'dense_outputs'/f'OUT_L{l:02d}.pt',weights_only=True)
        torch.testing.assert_close(module,av*2)
    capture.close()
    assert len(capture.layers)==24
    for l in capture.layers:
        assert len(l['heads'])==16
        assert l['module_output']['rollout_vs_dense']['all_queries']['relative_l2_frobenius']==0
        assert l['heads'][0]['rollout_vs_dense']['all_queries']['mean_per_query_cosine']==pytest.approx(1)
