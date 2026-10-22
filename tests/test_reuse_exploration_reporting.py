import json
from scripts.run_reuse_exploration_eval import report,SCENES,METHODS
from scripts.quality_metrics import quality_record


def fake_result(method):
    q=quality_record(dict(pose=dict(auc03=.4,auc30=.9),recon_unposed=dict(acc=.02,comp=.04,overall=.03,fscore=.8)))
    q.update(speedup_vs_vggt=1. if method=='dense' else .5,speedup_unavailable=None)
    return dict(method=method,quality=q,timing=dict(median_seconds=1. if method=='dense' else 2.,
               samples=[dict(peak_allocated_gib=8.)]*3))


def test_missing_scenes_never_produce_macro_quality(tmp_path):
    out=tmp_path/'heads/dense';out.mkdir(parents=True)
    data=fake_result('dense');(out/'result.json').write_text(json.dumps(data))
    (out/'forward.json').write_text(json.dumps(data))
    report(tmp_path)
    summary=json.loads((tmp_path/'summary.json').read_text())
    dense=next(r for r in summary['macro'] if r['method']=='dense')
    assert dense['quality_scenes_complete']==1 and dense['values'] is None
    assert 'N/A' in (tmp_path/'report.md').read_text()


def test_all_seven_equal_scene_macro_units_and_failed_speed_gate(tmp_path):
    for scene in SCENES:
        for method in METHODS:
            out=tmp_path/scene/method;out.mkdir(parents=True)
            data=fake_result(method)
            (out/'result.json').write_text(json.dumps(data));(out/'forward.json').write_text(json.dumps(data))
    report(tmp_path)
    summary=json.loads((tmp_path/'summary.json').read_text())
    new=next(r for r in summary['macro'] if r['method']=='reuse_explore')
    assert abs(new['values']['auc03']-.4)<1e-12
    assert abs(new['values']['cd']-.03)<1e-12
    assert new['speedup_ratio_of_mean_scene_medians']==.5
    assert new['auc30_f1_gates_pass'] and not new['twofold_speed_gate_pass']
    assert new['memory_gate_pass']
