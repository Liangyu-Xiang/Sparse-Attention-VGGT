import json
import math
import pytest
import torch
from saf3r.patch.av_internal_gate import InternalAVGate
from saf3r.patch.value_region_av import ValueRegionAVConfig,ValueRegionAVState,ValueRegionAVAttention
from saf3r.models.official_vggt.layers.attention import Attention
from saf3r.patch.region_history_attention import prepare_qkv


PARAMETERS=dict(version=1,feature='log_sqrt_head_kl_times_group_js',midpoint=-4.5,
    temperature=.74,min_alpha=.2,max_alpha=.8,alpha_step=.05,groups=4)


def test_gate_mapping_is_monotone_and_finite_without_layer_input():
    gate=InternalAVGate(PARAMETERS)
    values=[gate.alpha(x,x)[0] for x in [0.,1e-6,.001,.01,.1,1.]]
    assert values==sorted(values) and values[0]==.2 and values[-1]==.8
    with pytest.raises(ValueError):InternalAVGate(dict(PARAMETERS,temperature=0))
    with pytest.raises(ValueError):ValueRegionAVConfig(adaptive_gate='unused')


def test_dynamic_capacity_reserves_low_alpha_even_if_nominal_schedule_is_high(tmp_path):
    path=tmp_path/'gate.json';path.write_text(json.dumps(PARAMETERS))
    c=ValueRegionAVConfig(parent_only=True,adaptive_gate=str(path),source_width=4,region_child_width=2,
        attention_ratios=(0.,)+(.8,)*23,retentions=(.05,)+(.2,)*23)
    s=ValueRegionAVState(c);s.reset(1,8,16,'cpu')
    assert s.value.base_count>=max(p-round(.2*p) for p in s.total[1:])


def test_unsupervised_calibration_does_not_consume_layer_labels_or_errors(tmp_path):
    from scripts.calibrate_av_internal_gate import fit
    for scene in ('a','b'):
        path=tmp_path/'signals/eth3d'/scene/'result.json';path.parent.mkdir(parents=True)
        rows=[dict(layer=i,head_parent_kl=.001 if i<12 else .1,
                   query_group_js=.0001 if i<12 else .05,candidates=[dict(output_nmse=100)]) for i in range(24)]
        path.write_text(json.dumps(dict(dataset='eth3d',protocol=dict(ground_truth_used=False),
            input=dict(shape=[32,3,10,10]),layers=rows)))
    fit(tmp_path/'signals',tmp_path/'one.json')
    for path in (tmp_path/'signals/eth3d').glob('*/result.json'):
        data=json.loads(path.read_text())
        for i,row in enumerate(data['layers']):row.update(layer=100-i,candidates=[dict(output_nmse=-999)])
        path.write_text(json.dumps(data))
    fit(tmp_path/'signals',tmp_path/'two.json')
    a=json.loads((tmp_path/'one.json').read_text());b=json.loads((tmp_path/'two.json').read_text())
    for key in ('centers','midpoint','temperature'):assert a[key]==b[key]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_dynamic_all24_history_unique_output_and_fresh_input(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    path=tmp_path/'gate.json';path.write_text(json.dumps(PARAMETERS))
    c=ValueRegionAVConfig(parent_only=True,adaptive_gate=str(path),source_width=4,region_child_width=2,
                          capture=True,window_groups=2)
    s=ValueRegionAVState(c);s.reset(1,8,16,device);torch.manual_seed(5)
    try:
        for layer in range(24):
            o=Attention(dim=32,num_heads=2).to(device).eval();w=ValueRegionAVAttention(o,s,layer).eval()
            x=torch.randn(1,133,32,device=device);incoming=None if s.history is None else s.history.clone()
            with torch.inference_mode(),torch.autocast(device,enabled=device=='cuda',dtype=torch.bfloat16):
                y=w(x);d=s.captured[layer];mask=torch.zeros((133,133),device=device,dtype=torch.bool)
                for g,a in enumerate(d['attention']):
                    ids=torch.cat((d['common'],a[a>=0].long()))
                    assert len(ids.unique())==len(ids) and len(ids)>=math.floor(.2*133)
                    assert set(range(5)).issubset(ids.tolist())
                    if incoming is not None:
                        h=incoming[g];h=h[h>=0].long()
                        assert set(s.patches[h].tolist()).issubset(ids.tolist())
                    mask[g*32:min(133,(g+1)*32),ids]=True
                q,k,v=prepare_qkv(o,x,None);q=q.to(v.dtype);k=k.to(v.dtype)
                with torch.autocast(device,enabled=False):
                    z=(q.float()@k.float().transpose(-1,-2))*o.scale
                    expected=(z.masked_fill(~mask[None],-torch.inf).softmax(-1)@v.float()).to(v.dtype)
                expected=o.proj_drop(o.proj(expected.transpose(0,1).reshape(1,133,32)))
            torch.testing.assert_close(y.float(),expected.float(),rtol=.025 if device=='cuda' else 1e-5,
                                       atol=.012 if device=='cuda' else 1e-6)
            assert s.records[-1]['gate_seed']==(layer==0)
            if layer:assert s.records[-1]['attention_edges_target']>=s.records[-1]['gate_incoming_history_cap']
            assert s.policy_parent_keys is None
        s.reset(1,8,16,device)
        assert s.history is None and s.policy_info=={} and s.previous_layer==-1
    finally:torch.set_num_threads(old_threads)
