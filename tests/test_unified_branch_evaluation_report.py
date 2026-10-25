import json
from pathlib import Path
from scripts.run_unified_branch_7scenes import report,SCENES,OLD_METHODS,ROOT


BASELINE=ROOT/'outputs/global_attention/reuse_explore_20261021/quality128'


def test_old_results_reused_and_new_macro_requires_all_seven(tmp_path):
    report(tmp_path,BASELINE)
    d=json.loads((tmp_path/'comparison.json').read_text())
    assert len(d['per_run'])==35
    assert [m['complete'] for m in d['macro']]==[7,7,7,7,0]
    for scene in SCENES[:-1]:
        path=tmp_path/'quality128'/scene/'result.json';path.parent.mkdir(parents=True)
        original=json.loads((BASELINE/scene/'reuse_explore/result.json').read_text())
        # Fixture only, not an experiment: ensure even a copied numeric old
        # speedup is not misreported as new paired performance.
        original['quality']['speedup_vs_vggt']=123
        path.write_text(json.dumps(original))
    report(tmp_path,BASELINE)
    d=json.loads((tmp_path/'comparison.json').read_text())
    assert d['macro'][-1]['complete']==6
    assert not d['macro'][-1]['quality']['complete']
    new=[r for r in d['per_run'] if r['method']=='unified']
    assert all(r['quality']['speedup_vs_vggt'] is None for r in new)
    assert all(r['quality']['speedup_unavailable'] for r in new)
    assert 'New vs Dense end-to-end speedup is N/A' in (tmp_path/'comparison.md').read_text()


def test_complete_macro_has_six_metrics_but_no_new_speed_claim(tmp_path):
    for scene in SCENES:
        path=tmp_path/'quality128'/scene/'result.json';path.parent.mkdir(parents=True)
        path.write_text((BASELINE/scene/'reuse_explore/result.json').read_text())
    report(tmp_path,BASELINE)
    d=json.loads((tmp_path/'comparison.json').read_text())
    last=d['macro'][-1]
    assert last['complete']==7 and last['quality']['complete']
    assert len(last['quality']['values'])==6
    assert last['quality']['speedup_vs_vggt'] is None
    assert len(last['delta_vs_local_dense'])==6
