from types import SimpleNamespace
import pytest,torch
from saf3r.patch.protected_frame_keys import protected_layout,check_protected_budget
from saf3r.patch.dense_special_queries import query_ids


def test_first_frame_protection_is_disjoint_and_preserves_camera_ids():
    frames,h,w=14,25,37;n=frames*(h*w+5)
    patches,protected,semantic,active=protected_layout(frames,h,w,'cpu',True)
    assert active==13 and len(patches)==13*925
    assert len(protected)==14*5+925 and len(semantic)==14*5
    assert set(range(930))<=set(protected.tolist())
    assert not set(patches.tolist())&set(protected.tolist())
    assert sorted(patches.tolist()+protected.tolist())==list(range(n))
    assert query_ids(semantic,1).tolist()==list(range(0,n,930))
    cfg=SimpleNamespace(retention=.2,layer_retentions=[.1]*6+[.22]*12+[.1]*6)
    check_protected_budget(cfg,n,len(protected),active)
    assert all(int(r*n)-len(protected)+len(protected)<=r*n for r in cfg.layer_retentions)


def test_disabled_protection_matches_original_domain():
    p,s,semantic,f=protected_layout(2,3,7,'cpu',False)
    assert f==2 and s is semantic and len(p)==42 and len(s)==10


def test_infeasible_anchor_never_increases_retention():
    cfg=SimpleNamespace(retention=.1,layer_retentions=None)
    with pytest.raises(ValueError):check_protected_budget(cfg,2*930,935,1)
