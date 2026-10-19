from copy import deepcopy
from scripts.report_fused_shared_av import quality_gates
import json
import pytest


def result():
    return dict(scene='chess',metrics=dict(pose=dict(auc30=.85),depth={'Abs Rel':.1},recon_unposed=dict(fscore=.5)))


def test_unavailable_quality_never_passes():
    assert not quality_gates([])['passed']
    a=result();del a['metrics']['depth']
    assert not quality_gates([(a,a)])['passed']


def test_strict_gates_and_scene_guardrails():
    a=result();b=deepcopy(a)
    assert quality_gates([(a,b)])['passed']
    b['metrics']['pose']['auc30']-=.0021
    assert not quality_gates([(a,b)])['passed']
    b=deepcopy(a);b['metrics']['depth']['Abs Rel']=.1011
    assert not quality_gates([(a,b)])['passed']
    b=deepcopy(a);b['metrics']['recon_unposed']['fscore']-=.0051
    assert not quality_gates([(a,b)])['passed']
    b=deepcopy(a);b['metrics']['pose']['auc30']-=.02
    c=deepcopy(a);c['scene']='fire';d=deepcopy(c);d['metrics']['pose']['auc30']+=.02
    assert not quality_gates([(a,b),(c,d)])['passed']


def test_expensive_stages_refuse_failed_or_non128_gate(tmp_path):
    from scripts.run_fused_shared_av_stages import require_gate
    path=tmp_path/'gate.json'
    path.write_text(json.dumps(dict(frames=[128],speed_passed=False,memory_passed=True,warmed_protocol_valid=True)))
    with pytest.raises(ValueError):require_gate(path)
    path.write_text(json.dumps(dict(frames=[50],speed_passed=True,memory_passed=True,warmed_protocol_valid=True)))
    with pytest.raises(ValueError):require_gate(path)
    path.write_text(json.dumps(dict(frames=[128],speed_passed=True,memory_passed=True,warmed_protocol_valid=True)))
    require_gate(path)
